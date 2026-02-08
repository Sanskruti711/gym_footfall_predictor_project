# train_model.py
import sqlite3
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

DB_PATH = "project.db"
TABLE_NAME = "gym_footfall"

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df

def train_and_evaluate():
    df = load_data()

    feature_cols = [
        "hour",
        "day_of_week",
        "exam_period",
        "temperature_c",
        "is_weekend",
        "special_event",
        "is_holiday",
        "sports_or_challenge",
        "is_new_term",
        "previous_day_occupancy"
    ]
    target_col = "occupancy_percentage"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1) Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lr = lin_reg.predict(X_test)
    rmse_lr = mean_squared_error(y_test, y_pred_lr) ** 0.5

    # 2) Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rmse_rf = mean_squared_error(y_test, y_pred_rf) **0.5

    print(f"LinearRegression RMSE: {rmse_lr:.2f}")
    print(f"RandomForestRegressor RMSE: {rmse_rf:.2f}")

    # choose best
    if rmse_rf <= rmse_lr:
        best_model = rf
        best_name = "RandomForestRegressor"
        best_rmse = rmse_rf
    else:
        best_model = lin_reg
        best_name = "LinearRegression"
        best_rmse = rmse_lr

    # save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/model_{timestamp}.pkl"
    joblib.dump(best_model, model_path)

    # save metrics
    metrics = {
        "model_name": best_name,
        "rmse": best_rmse,
        "timestamp": timestamp,
        "model_path": model_path
    }

    with open("model_history.json", "a") as f:
        f.write(json.dumps(metrics) + "\n")

    print(f"Saved best model: {best_name} with RMSE={best_rmse:.2f}")
    print(f"Model file: {model_path}")

if __name__ == "__main__":
    train_and_evaluate()
