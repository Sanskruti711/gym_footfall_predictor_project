# scripts/retrain_model.py
"""
Appends N new synthetic days to the DB (continuing from where the data left off),
then retrains the model on the full updated dataset.

Run from the PROJECT ROOT:
    python scripts/retrain_model.py
"""
import sys
import os

# Make sure sibling scripts are importable regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import sqlite3
import numpy as np
import pandas as pd

from data_generator import generate_data, MAX_CAPACITY, assign_time_block
from train_model import main as train_main, DB_PATH, TABLE_NAME, FEATURE_COLS


# ── helpers ──────────────────────────────────────────────────────────────────

def get_existing_df() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df


def build_new_rows(existing_df: pd.DataFrame, num_new_days: int) -> pd.DataFrame:
    """
    Generate num_new_days of new rows that:
      - continue day_index from max+1
      - inherit correct event flags based on absolute day position
      - compute previous_day_occupancy and rolling_3day_avg from REAL existing data
    """
    max_day = int(existing_df["day_index"].max())
    total_days_after = max_day + 1 + num_new_days  # total calendar length after append

    # Generate a fresh full dataset up to the new total length so event flags
    # (exam windows, holidays, new_term) are positioned correctly on the calendar
    full_df = generate_data(num_days=total_days_after)

    # Only keep the new days
    new_df = full_df[full_df["day_index"] > max_day].copy()

    # ── Fix previous_day_occupancy and rolling_3day_avg using REAL history ──
    # Build lookup: (day_index, time_block) → mean people_count
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    day_block_mean = (
        combined.groupby(["day_index", "time_block"])["people_count"]
        .mean()
    )

    def prev_occ(row):
        key = (row["day_index"] - 1, row["time_block"])
        return day_block_mean.get(key, np.nan)

    def rolling_avg(row):
        vals = [
            day_block_mean.get((row["day_index"] - d, row["time_block"]), np.nan)
            for d in [1, 2, 3]
        ]
        valid = [v for v in vals if not np.isnan(v)]
        return round(float(np.mean(valid)), 1) if valid else np.nan

    new_df["previous_day_occupancy"] = new_df.apply(prev_occ, axis=1)
    new_df["rolling_3day_avg"]       = new_df.apply(rolling_avg, axis=1)

    # Fallback: if still NaN (e.g. first new day has no prior), use block mean
    block_fallback = existing_df.groupby("time_block")["people_count"].mean()
    new_df["previous_day_occupancy"] = new_df.apply(
        lambda r: block_fallback.get(r["time_block"], 30.0)
        if pd.isna(r["previous_day_occupancy"]) else r["previous_day_occupancy"],
        axis=1,
    ).round(1)
    new_df["rolling_3day_avg"] = new_df.apply(
        lambda r: block_fallback.get(r["time_block"], 30.0)
        if pd.isna(r["rolling_3day_avg"]) else r["rolling_3day_avg"],
        axis=1,
    ).round(1)

    return new_df


def append_to_db(new_df: pd.DataFrame):
    conn = sqlite3.connect(DB_PATH)
    new_df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
    conn.close()


# ── main ─────────────────────────────────────────────────────────────────────

def retrain(num_new_days: int = 3):
    print(f"── Step 1: Loading existing data from DB ──")
    existing_df = get_existing_df()
    print(f"   Existing rows : {len(existing_df):,}")
    print(f"   Max day_index : {existing_df['day_index'].max()}")

    print(f"\n── Step 2: Generating {num_new_days} new day(s) ──")
    new_df = build_new_rows(existing_df, num_new_days)
    print(f"   New rows generated : {len(new_df):,}")

    print(f"\n── Step 3: Appending new rows to DB ──")
    append_to_db(new_df)
    print(f"   Done. Total rows now: {len(existing_df) + len(new_df):,}")

    print(f"\n── Step 4: Retraining model on full dataset ──")
    train_main()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Append new data and retrain GymIQ model")
    parser.add_argument("--days", type=int, default=3, help="Number of new days to append (default: 3)")
    args = parser.parse_args()
    retrain(num_new_days=args.days)