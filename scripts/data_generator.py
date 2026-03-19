# scripts/data_generator.py
import sys
import sqlite3
import numpy as np
import pandas as pd

# Fix Windows cp1252 terminal encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MAX_CAPACITY = 80   # approximate gym capacity
NUM_DAYS     = 365  # one full year — captures all seasonal cycles


# ── Time helpers ──────────────────────────────────────────────────────────────

def assign_time_block(hour: int) -> int:
    """
    0 → 06:00–08:00  Early morning
    1 → 08:00–12:00  Morning
    2 → 12:00–16:00  Afternoon
    3 → 16:00–22:00  Evening
    """
    if 6 <= hour < 8:    return 0
    elif 8 <= hour < 12: return 1
    elif 12 <= hour < 16: return 2
    else:                 return 3


def day_to_month(day_index: int) -> int:
    """Map day_index (0-364) to calendar month (1-12)."""
    months = [31,28,31,30,31,30,31,31,30,31,30,31]
    cumulative = 0
    for m, days in enumerate(months, start=1):
        cumulative += days
        if day_index < cumulative:
            return m
    return 12


def day_to_week_of_year(day_index: int) -> int:
    """Map day_index (0-364) to week of year (1-52)."""
    return min(int(day_index // 7) + 1, 52)


def circular_encode(value: float, max_val: float) -> tuple[float, float]:
    """
    Encode a cyclic value (e.g. month 1-12) as (sin, cos) so that
    the model understands Dec → Jan as continuous, not a 12 → 1 jump.
    """
    angle = 2 * np.pi * value / max_val
    return round(np.sin(angle), 4), round(np.cos(angle), 4)


def drift_factor(day_index: int, num_days: int) -> float:
    """Simulate growing gym membership: 1.0 (start) → 1.15 (end)."""
    progress = day_index / max(1, num_days - 1)
    return 1.0 + 0.15 * progress


def seasonal_temp(day_index: int) -> float:
    """
    Realistic temperature curve for a year starting in January.
    Coldest in Jan (day 0), hottest around day 182 (July), cold again by Dec.
    Returns offset in °C.
    """
    # Shift so day 0 = Jan = coldest. Peak at day ~182 (July)
    angle = 2 * np.pi * (day_index - 182) / 365
    return 8 * np.sin(angle)   # ±8°C seasonal swing


# ── Main generator ────────────────────────────────────────────────────────────

def generate_data(num_days: int = NUM_DAYS) -> pd.DataFrame:
    np.random.seed(42)

    # ── Calendar-aware event windows ──────────────────────────────────────────
    # Exam periods: 4 per year (Jan finals, Apr finals, Jul re-sits, Oct mids)
    exam_days = (
        set(range(14,  28))  |   # mid-Jan finals
        set(range(105, 119)) |   # mid-Apr finals
        set(range(196, 207)) |   # mid-Jul re-sits
        set(range(287, 298))     # mid-Oct mids
    )

    # Pre-exam stress weeks (one week before each exam window)
    pre_exam_days = (
        set(range(7,   14)) |
        set(range(98,  105)) |
        set(range(189, 196)) |
        set(range(280, 287))
    )

    # Holiday clusters: winter break, spring break, summer, Diwali
    holiday_days = (
        set(range(0,   7))   |   # New Year week
        set(range(355, 365)) |   # Christmas/winter break
        set(range(120, 128)) |   # Spring break
        set(range(210, 225)) |   # Summer break
        set(range(295, 302))     # Diwali / autumn break
    )

    # New term starts: Jan, Apr, Sep
    new_term_days = (
        set(range(7,   17)) |    # Jan term start (after NY week)
        set(range(128, 138)) |   # Apr/spring term
        set(range(245, 255))     # Sep term start
    )

    # Special events: competitions, open days
    special_event_days = {20, 55, 90, 150, 200, 260, 320}

    # Sports / gym challenges
    sports_challenge_days = (
        set(range(30,  33)) |    # January challenge
        set(range(170, 174)) |   # Summer challenge
        set(range(330, 334))     # Year-end challenge
    )

    # New Year resolution window: Jan 1–21 (day 0–20)
    new_year_jan_days = set(range(0, 21))

    # Summer quiet period: Jun–Jul (approx day 152–212)
    summer_days = set(range(152, 213))

    # ── Pass 1: raw rows ──────────────────────────────────────────────────────
    rows = []

    for day in range(num_days):
        day_of_week  = day % 7
        is_weekend   = 1 if day_of_week in [5, 6] else 0
        month        = day_to_month(day)
        week_of_year = day_to_week_of_year(day)

        # Circular time encodings
        sin_month, cos_month = circular_encode(month, 12)
        sin_week,  cos_week  = circular_encode(week_of_year, 52)

        # Event flags
        exam_period         = 1 if day in exam_days            else 0
        is_pre_exam_week    = 1 if day in pre_exam_days        else 0
        is_holiday          = 1 if day in holiday_days         else 0
        is_new_term         = 1 if day in new_term_days        else 0
        special_event       = 1 if day in special_event_days   else 0
        sports_or_challenge = 1 if day in sports_challenge_days else 0
        is_new_year_jan     = 1 if day in new_year_jan_days    else 0
        is_summer           = 1 if day in summer_days          else 0

        day_drift    = drift_factor(day, num_days)
        season_temp  = seasonal_temp(day)

        for hour in range(6, 23):
            time_block  = assign_time_block(hour)

            # ── Temperature ──────────────────────────────────────────────
            base_temp = np.random.normal(26, 3) + season_temp
            if hour < 9:   base_temp -= 4
            elif hour > 19: base_temp -= 2
            temperature_c = float(np.clip(base_temp, 5, 45))

            # ── Base crowd by time block ──────────────────────────────────
            block_base  = [25, 35, 15, 45][time_block]
            block_noise = [4,  5,  4,  8 ][time_block]
            base = float(block_base)

            # Weekend
            base += 5 if is_weekend else 0

            # ── Seasonal effects (the new stuff) ──────────────────────────
            # New Year resolutions: +18 in Jan (everyone joins the gym lol)
            if is_new_year_jan:
                base += 18

            # Summer quieter: students travel, outdoor activities replace gym
            if is_summer:
                base -= 10

            # Pre-exam: students panic-study, avoid gym
            if is_pre_exam_week:
                base -= 8

            # Exam: even less gym time
            if exam_period:
                base -= 12 if time_block in [1, 3] else -5

            # Holidays
            if is_holiday and not special_event:
                base -= 16

            # Temperature comfort
            if temperature_c < 12 or temperature_c > 38:
                base -= 8
            elif 18 <= temperature_c <= 30:
                base += 3

            # Events / challenges / new term
            if special_event:       base += 15
            if sports_or_challenge: base += 12
            if is_new_term:         base += 8

            # Block noise
            base += np.random.normal(0, block_noise)

            people_count = float(np.clip(base * day_drift, 0, MAX_CAPACITY))

            rows.append({
                "day_index"          : day,
                "day_of_week"        : day_of_week,
                "is_weekend"         : is_weekend,
                "month"              : month,
                "week_of_year"       : week_of_year,
                "sin_month"          : sin_month,
                "cos_month"          : cos_month,
                "sin_week"           : sin_week,
                "cos_week"           : cos_week,
                "time_block"         : time_block,
                "exam_period"        : exam_period,
                "is_pre_exam_week"   : is_pre_exam_week,
                "temperature_c"      : round(temperature_c, 1),
                "special_event"      : special_event,
                "is_holiday"         : is_holiday,
                "sports_or_challenge": sports_or_challenge,
                "is_new_term"        : is_new_term,
                "is_new_year_jan"    : is_new_year_jan,
                "is_summer"          : is_summer,
                "previous_day_occupancy": 0.0,   # filled in pass 2
                "rolling_3day_avg"   : 0.0,       # filled in pass 2
                "people_count"       : round(people_count, 1),
            })

    df = pd.DataFrame(rows)

    # ── Pass 2: fill lag features from real data ──────────────────────────────
    day_block_mean = (
        df.groupby(["day_index", "time_block"])["people_count"]
        .mean()
        .rename("_dbm")
    )
    df = df.join(day_block_mean, on=["day_index", "time_block"])

    def prev_occ(row):
        return day_block_mean.get((row["day_index"] - 1, row["time_block"]), np.nan)

    def roll3(row):
        vals = [day_block_mean.get((row["day_index"] - d, row["time_block"]), np.nan)
                for d in [1, 2, 3]]
        valid = [v for v in vals if not np.isnan(v)]
        return round(float(np.mean(valid)), 1) if valid else np.nan

    df["previous_day_occupancy"] = df.apply(prev_occ, axis=1)
    df["rolling_3day_avg"]       = df.apply(roll3,    axis=1)

    # Fallback for day 0 — pandas 3.0 safe syntax
    df["previous_day_occupancy"] = df["previous_day_occupancy"].fillna(df["_dbm"]).round(1)
    df["rolling_3day_avg"]       = df["rolling_3day_avg"].fillna(df["_dbm"]).round(1)
    df.drop(columns=["_dbm"], inplace=True)

    return df


def save_to_sqlite(df: pd.DataFrame, db_path="data/project.db", table_name="gym_footfall"):
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved {len(df):,} rows --> {db_path} :: {table_name}")


if __name__ == "__main__":
    df = generate_data(num_days=NUM_DAYS)
    save_to_sqlite(df)
    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nNew Year Jan rows (sample):")
    print(df[df["is_new_year_jan"]==1][["day_index","month","people_count"]].head(5))
    print(f"\nSummer rows (sample):")
    print(df[df["is_summer"]==1][["day_index","month","people_count"]].head(5))
    print(df.describe())