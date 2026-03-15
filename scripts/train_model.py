# train_model.py
import sqlite3
import json
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from preprocessing import preprocess_df, FEATURE_COLS, TARGET_COL

import joblib

DB_PATH = "project.db"
TABLE_NAME = "gym_footfall"


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-5
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)


def train_and_evaluate():
    # 1) Load and preprocess
    df_raw = load_data()
    df = preprocess_df(df_raw)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # 2) Train–test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3) Define models
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=5, random_state=42),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
    }

    results = []

    # 4) Train and evaluate each model
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_test, y_pred)
        mape_val = mape(y_test, y_pred)

        results.append(
            {
                "name": name,
                "model": mdl,
                "rmse": rmse,
                "mae": mae,
                "mape": mape_val,
            }
        )

        print(f"{name} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape_val:.1f}%")

    # 5) Choose best by RMSE (you can also consider MAPE)
    best = min(results, key=lambda r: r["rmse"])
    best_model = best["model"]
    best_name = best["name"]
    best_rmse = best["rmse"]
    best_mae = best["mae"]
    best_mape = best["mape"]

    # 6) Save best model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/model_{timestamp}.pkl"
    joblib.dump(best_model, model_path)

    # 7) Append metrics to history log
    metrics = {
        "model_name": best_name,
        "rmse": best_rmse,
        "mae": best_mae,
        "mape": best_mape,
        "timestamp": timestamp,
        "model_path": model_path,
    }

    with open("model_history.json", "a") as f:
        f.write(json.dumps(metrics) + "\n")

    print(
        f"Saved best model: {best_name} "
        f"with RMSE={best_rmse:.2f}, MAPE={best_mape:.1f}%"
    )
    print(f"Model file: {model_path}")


if __name__ == "__main__":
    train_and_evaluate()


