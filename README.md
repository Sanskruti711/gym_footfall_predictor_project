# College Gym Footfall Predictor

This project predicts hourly crowd levels (occupancy percentage) in a college gym using synthetic data stored in a SQLite database and traditional machine learning models. It demonstrates how to move from a basic model to a small, production‑like pipeline with SQL, Python scripts, model versioning, and a Streamlit dashboard.

---

## Data generation

To (re)create the SQLite database with synthetic gym data:

```bash
python data_generator.py
```

This script:
- Generates synthetic hourly occupancy data.
- Stores it in `project.db` under the `gym_footfall` table.

---

## Model training

To train and save a model:

```bash
python train_model.py
```

This script:
- Loads data from `project.db` (table `gym_footfall`).
- Preprocesses it using `preprocess_df`.
- Trains three models: `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`.
- Prints RMSE, MAE, and MAPE for each model.
- Selects the best model based on RMSE.
- Saves the best model into the `models/` folder.
- Appends its metrics and metadata to `model_history.json`.

---

## Model history and versioning

Every time `train_model.py` is run, a new model is trained and saved into the `models/` folder, and a JSON line is appended to `model_history.json`. Each line looks like:

```json
{"model_name": "RandomForestRegressor", "rmse": 9.55, "mae": 7.67, "mape": 4140396.75, "timestamp": "20260222_192330", "model_path": "models/model_20260222_192330.pkl"}
```

This file acts as a simple model registry and lets you track:
- Which model type was used.
- When it was trained (`timestamp`).
- Its performance metrics (RMSE, MAE, MAPE).
- Where the corresponding `.pkl` file is stored.

---

## Streamlit dashboard

To run the interactive dashboard:

```bash
streamlit run app.py
```

The app:
- Loads the most recent model from the `models/` folder using `model_history.json`.
- Shows current model info (file path, name, RMSE, timestamp).
- Lets you choose date/time and context (exam period, holiday, temperature, etc.).
- Displays the predicted gym occupancy for that hour and a simple crowd‑level label.
```