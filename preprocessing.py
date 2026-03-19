# preprocessing.py
"""
Preprocessing utilities for the GymIQ gym_footfall dataset.
Called by train_model.py before model training.
"""
import pandas as pd
import numpy as np

FEATURE_COLS = [
    "day_of_week",
    "is_weekend",
    "time_block",
    "month",
    "week_of_year",
    "sin_month",
    "cos_month",
    "sin_week",
    "cos_week",
    "exam_period",
    "is_pre_exam_week",
    "is_new_year_jan",
    "is_summer",
    "special_event",
    "is_holiday",
    "sports_or_challenge",
    "is_new_term",
    "temperature_c",
    "previous_day_occupancy",
    "rolling_3day_avg",
]
TARGET_COL = "people_count"


def preprocess_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw gym_footfall DataFrame before model training.

    Steps:
      1. Keep only required feature + target columns
      2. Enforce numeric dtypes (coerce bad values to NaN)
      3. Drop rows with any missing values
      4. Clip target to valid range [0, 80] (gym capacity)
      5. Clip temperature to plausible range [5, 45]
      6. Reset index

    Returns a clean DataFrame ready for train/test split.
    """
    df = raw_df.copy()

    required_cols = FEATURE_COLS + [TARGET_COL]
    keep_cols = ["day_index"] + required_cols

    # 1. Keep only required columns (+ day_index for chronological split)
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"preprocess_df: missing columns: {missing}")
    df = df[keep_cols]

    # 2. Enforce numeric types
    df = df.apply(pd.to_numeric, errors="coerce")

    # 3. Drop rows with missing values
    before = len(df)
    df = df.dropna(subset=required_cols)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  [preprocess] Dropped {dropped} rows with missing values")

    # 4. Clip target to gym capacity range
    df[TARGET_COL] = df[TARGET_COL].clip(0, 80)

    # 5. Clip temperature to plausible range
    df["temperature_c"] = df["temperature_c"].clip(5, 45)

    # 6. Reset index
    df = df.reset_index(drop=True)

    print(f"  [preprocess] Clean rows: {len(df):,}  |  Features: {len(FEATURE_COLS)}")
    return df