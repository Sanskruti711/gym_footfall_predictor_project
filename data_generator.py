# data_generator.py
import sqlite3
import numpy as np
import pandas as pd

def generate_data(num_days=60):
    np.random.seed(42)
    rows = []

    # days 0 .. num_days-1
    for day in range(num_days):
        day_of_week = day % 7  # 0-6
        is_weekend = 1 if day_of_week in [5, 6] else 0

        # simple flags
        exam_period = 1 if 20 <= day <= 30 else 0  # fake exam window
        is_holiday = 1 if day % 30 == 0 else 0
        is_new_term = 1 if day < 7 else 0
        special_event = 1 if day in [10, 25] else 0
        sports_or_challenge = 1 if day in [15, 40] else 0

        for hour in range(6, 23):  # 6 to 22
            # temperature: colder at morning/evening, hotter mid-day
            base_temp = np.random.normal(28, 4)
            if hour < 9:
                base_temp -= 4
            elif hour > 19:
                base_temp -= 2
            temperature_c = max(10, min(40, base_temp))

            # previous_day_occupancy (fake, will refine later)
            previous_day_occ = np.random.uniform(10, 80)

            # base occupancy
            occ = 20.0

            # hour effect
            if 17 <= hour <= 20:
                occ += 30  # evening peak
            elif 6 <= hour <= 9:
                occ += 15  # morning good
            elif 21 <= hour <= 22:
                occ -= 5  # late

            # weekday/weekend
            if is_weekend:
                occ -= 10
            else:
                occ += 5

            # exams reduce
            if exam_period:
                occ -= 20

            # holidays reduce unless special event
            if is_holiday and not special_event:
                occ -= 25

            # temperature effect
            if temperature_c < 15 or temperature_c > 35:
                occ -= 10  # too cold/hot
            else:
                occ += 5   # comfy

            # special events / sports / new term
            if special_event:
                occ += 20
            if sports_or_challenge:
                occ += 15
            if is_new_term:
                occ += 10

            # previous day influence
            occ += 0.3 * (previous_day_occ - 50)

            # noise
            occ += np.random.normal(0, 8)

            # clip 0-100
            occ = max(0, min(100, occ))

            rows.append({
                "day_index": day,
                "hour": hour,
                "day_of_week": day_of_week,
                "exam_period": exam_period,
                "temperature_c": round(temperature_c, 1),
                "is_weekend": is_weekend,
                "special_event": special_event,
                "is_holiday": is_holiday,
                "sports_or_challenge": sports_or_challenge,
                "is_new_term": is_new_term,
                "previous_day_occupancy": round(previous_day_occ, 1),
                "occupancy_percentage": round(occ, 1)
            })

    df = pd.DataFrame(rows)
    return df

def save_to_sqlite(df, db_path="project.db", table_name="gym_footfall"):
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

if __name__ == "__main__":
    df = generate_data(num_days=60)
    save_to_sqlite(df)
    print("Data generated and saved to project.db, table gym_footfall")
