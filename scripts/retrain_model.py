# retrain_model.py
import sqlite3
import numpy as np
import pandas as pd
from data_generator import generate_data
from train_model import train_and_evaluate, DB_PATH, TABLE_NAME

def append_new_days(num_new_days=3):
    # get current max day_index
    conn = sqlite3.connect(DB_PATH)
    df_existing = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    max_day = df_existing["day_index"].max()
    conn.close()

    # generate data for num_new_days after max_day
    new_df = generate_data(num_days=num_new_days)
    # shift day_index so it continues from max_day + 1
    new_df["day_index"] = new_df["day_index"] + max_day + 1

    conn = sqlite3.connect(DB_PATH)
    new_df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
    conn.close()
    print(f"Appended {num_new_days} new days of data.")

if __name__ == "__main__":
    append_new_days(num_new_days=3)
    train_and_evaluate()
