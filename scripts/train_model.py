# scripts/train_model.py
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

DB_PATH    = "data/project.db"
TABLE_NAME = "gym_footfall"
MODELS_DIR = "models"

# ── Features ────────────────────────────────────────────────────────────────
# Includes the three new columns added by the improved data generator
FEATURE_COLS = [
    # ── Day / time ──
    "day_of_week",
    "is_weekend",
    "time_block",
    # ── Seasonal / temporal ──
    "month",              # raw month (1-12)
    "week_of_year",       # raw week (1-52)
    "sin_month",          # circular encoding — captures Dec→Jan continuity
    "cos_month",
    "sin_week",           # circular encoding — week-level seasonality
    "cos_week",
    # ── Event flags ──
    "exam_period",
    "is_pre_exam_week",   # panic-study week before exams
    "is_new_year_jan",    # New Year resolution crowd spike
    "is_summer",          # quieter Jun-Jul period
    "special_event",
    "is_holiday",
    "sports_or_challenge",
    "is_new_term",
    # ── Context ──
    "temperature_c",
    "previous_day_occupancy",
    "rolling_3day_avg",
]
TARGET_COL = "people_count"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2):
    """
    Split by day_index (chronological) instead of random shuffling.
    Prevents future data leaking into training — critical for time-series.
    """
    cutoff = int(df["day_index"].max() * (1 - test_frac))
    train  = df[df["day_index"] <= cutoff]
    test   = df[df["day_index"] >  cutoff]
    return train, test


def robust_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error that avoids division by near-zero.
    Clips denominator to at least 1 person so we never get inf/astronomical %.
    """
    denom = np.maximum(np.abs(y_true), 1.0)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def prediction_interval(model, X: pd.DataFrame):
    """
    Estimate 10th/90th percentile bounds.
    Works for RF and GradientBoosting (tree ensembles).
    Falls back to ±15% proxy for Ridge.
    """
    estimators = getattr(model, "estimators_", None)
    if estimators is not None:
        try:
            # GradientBoosting: estimators_ is shape (n_iter, 1) — flatten it
            flat = [t[0] if hasattr(t, "__len__") else t for t in estimators]
            tree_preds = np.array([t.predict(X) for t in flat])
            lower = np.percentile(tree_preds, 10, axis=0)
            upper = np.percentile(tree_preds, 90, axis=0)
            return lower, upper
        except Exception:
            pass
    # Fallback for non-tree models
    pred = model.predict(X)
    return pred * 0.85, pred * 1.15


def save_history(record: dict):
    """
    Persist model metadata to the same SQLite DB (model_history table).
    Much easier to query in the Streamlit app than a JSONL file.
    """
    conn = sqlite3.connect(DB_PATH)
    pd.DataFrame([record]).to_sql(
        "model_history", conn, if_exists="append", index=False
    )
    conn.close()


# ── Candidate models ─────────────────────────────────────────────────────────

def build_candidates():
    """
    5 models covering linear baseline, tree ensembles, boosting:
      1. RandomForest      — strong baseline, handles non-linearity well
      2. ExtraTrees        — more randomised than RF, often faster & competitive
      3. GradientBoosting  — sequential boosting, good on tabular data
      4. XGBoost           — industry-standard boosting, usually top performer
      5. Ridge             — fast linear baseline, sanity check
    """
    return [
        (
            "RandomForest",
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "n_estimators"     : [200, 300, 400],
                "max_depth"        : [None, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf" : [1, 2, 4],
                "max_features"     : ["sqrt", "log2"],
            },
        ),
        (
            "ExtraTrees",
            ExtraTreesRegressor(random_state=42, n_jobs=-1),
            {
                "n_estimators"     : [200, 300, 400],
                "max_depth"        : [None, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf" : [1, 2, 4],
                "max_features"     : ["sqrt", "log2"],
            },
        ),
        (
            "GradientBoosting",
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators" : [100, 200, 300],
                "max_depth"    : [3, 4, 5],
                "learning_rate": [0.05, 0.1, 0.15],
                "subsample"    : [0.7, 0.85, 1.0],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        (
            "XGBoost",
            XGBRegressor(random_state=42, n_jobs=-1,
                         verbosity=0, objective="reg:squarederror"),
            {
                "n_estimators" : [100, 200, 300],
                "max_depth"    : [3, 4, 6],
                "learning_rate": [0.05, 0.1, 0.2],
                "subsample"    : [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "reg_alpha"    : [0, 0.1, 1.0],
            },
        ),
        (
            "Ridge",
            Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
            {"ridge__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]},
        ),
    ]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load & validate -----------------------------------------------------
    df = load_data()

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"These feature columns are missing from the DB — "
            f"did you regenerate the data? Missing: {missing}"
        )

    train_df, test_df = chronological_split(df, test_frac=0.2)
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[TARGET_COL]

    print(f"Train rows: {len(X_train):,}  |  Test rows: {len(X_test):,}\n")

    # 2. Train & evaluate all candidates ------------------------------------
    results = []  # list of (name, fitted_model, rmse, mape, params_str)

    for name, estimator, param_dist in build_candidates():
        print(f"── Training {name} ──")

        search = RandomizedSearchCV(
            estimator,
            param_distributions=param_dist,
            n_iter=15,
            scoring="neg_root_mean_squared_error",
            cv=5,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_

        y_pred = best.predict(X_test)
        rmse   = float(mean_squared_error(y_test, y_pred) ** 0.5)
        mape   = robust_mape(y_test.values, y_pred)

        print(f"   RMSE: {rmse:.2f}  |  MAPE: {mape:.2f}%  |  Best params: {search.best_params_}")
        results.append((name, best, rmse, mape, str(search.best_params_)))

    # 3. Pick winner (lowest RMSE) ------------------------------------------
    results.sort(key=lambda x: x[2])
    print("\n── Model comparison (sorted by RMSE) ──")
    for name, _, rmse, mape, _ in results:
        winner_tag = " ✅ WINNER" if name == results[0][0] else ""
        print(f"   {name:<22} RMSE={rmse:.2f}  MAPE={mape:.2f}%{winner_tag}")

    best_name, best_model, best_rmse, best_mape, best_params = results[0]

    # 4. Feature importance (tree models only) ------------------------------
    # Get the underlying estimator if it's a Pipeline
    inner = best_model.named_steps.get("ridge", best_model) if hasattr(best_model, "named_steps") else best_model
    if hasattr(inner, "feature_importances_"):
        importance = dict(zip(FEATURE_COLS, inner.feature_importances_.round(4)))
        importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        print(f"\nFeature importances ({best_name}):")
        for feat, score in importance_sorted.items():
            print(f"  {feat:<28} {score:.4f}")
        top_features = str(list(importance_sorted.keys())[:3])
    else:
        # Ridge: use coefficients as proxy for importance
        coefs = dict(zip(FEATURE_COLS, np.abs(inner.coef_).round(4)))
        top_features = str(sorted(coefs, key=coefs.get, reverse=True)[:3])
        print(f"\nRidge coefficients (absolute): {coefs}")

    # 5. Prediction interval ------------------------------------------------
    lower, upper       = prediction_interval(best_model, X_test)
    avg_interval_width = float(np.mean(upper - lower))
    print(f"\nAvg 80% prediction interval width: {avg_interval_width:.1f} people")

    # 6. Save best model ----------------------------------------------------
    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_{timestamp}.pkl"
    model_path     = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(best_model, model_path)
    print(f"Saved best model ({best_name}) → {model_path}")

    # 7. Persist to history -------------------------------------------------
    record = {
        "timestamp"          : timestamp,
        "model_name"         : best_name,
        "model_path"         : model_path,
        "rmse"               : round(best_rmse, 4),
        "mape"               : round(best_mape, 4),
        "avg_interval_width" : round(avg_interval_width, 2),
        "best_params"        : best_params,
        "features"           : str(FEATURE_COLS),
        "train_rows"         : len(X_train),
        "test_rows"          : len(X_test),
        "top_features"       : top_features,
        # Store all 5 model RMSEs for dashboard comparison chart
        "all_model_rmses"    : str({n: round(r, 4) for n, _, r, _, _ in results}),
    }
    save_history(record)
    print("History saved to DB → table 'model_history'")


if __name__ == "__main__":
    main()