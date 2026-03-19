# 🏋️ GymIQ — Smart Gym Footfall Predictor

> A machine learning pipeline that predicts how crowded a college gym will be across four daily time slots, served through an animated Streamlit dashboard.

---

## 📌 Project Overview

College gyms suffer from highly uneven footfall. Students arrive during peak hours, face overcrowding, and leave without training. **GymIQ** solves this by predicting crowd levels for any day, time slot, and set of conditions — so users can plan smarter visits.

The system trains **5 regression models**, auto-selects the best performer, and displays predictions with an 80% confidence interval on a live gauge chart.

---

## 🗂️ Project Structure

```
gym_footfall_predictor_project/
├── scripts/
│   ├── data_generator.py      # Generate 365 days of synthetic footfall data
│   ├── train_model.py         # Train 5 models, compare, save best
│   └── retrain_model.py       # Append new data + trigger retrain
├── dashboard/
│   └── app.py                 # Streamlit dashboard (4 pages)
├── data/
│   └── project.db             # SQLite DB (gym_footfall + model_history tables)
├── models/                    # Saved .pkl model files (timestamped)
├── preprocessing.py           # Data preprocessing utilities
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/Sanskruti711/gym_footfall_predictor_project
cd gym_footfall_predictor_project

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

### Step 1 — Generate Data
```bash
python scripts/data_generator.py
```
Generates **6,205 rows** of synthetic gym footfall data across 365 days and saves to `data/project.db`.

### Step 2 — Train Models
```bash
python scripts/train_model.py
```
Trains **5 models** with RandomizedSearchCV, evaluates on a chronological test split, prints a comparison table, and saves the best model as a timestamped `.pkl` file.

### Step 3 — Launch Dashboard
```bash
streamlit run dashboard/app.py
```

### Retrain with New Data
```bash
python scripts/retrain_model.py --days 7
```
Appends 7 new days to the DB and retrains automatically.

---

## 🤖 Models Compared

| Model | Why Included |
|---|---|
| **XGBoost** ✅ | Industry-standard boosting; regularisation; typically best on tabular data |
| **Ridge Regression** | Fast linear baseline; sanity check |
| **GradientBoosting** | Sequential boosting; strong tabular performance |
| **ExtraTrees** | More randomised than RF; often competitive |
| **RandomForest** | Strong ensemble baseline; interpretable importances |

The winner is selected automatically by lowest RMSE on the test set.

---

## 📊 Features Used (21 total)

| Category | Features |
|---|---|
| Time | `day_of_week`, `is_weekend`, `time_block` |
| Seasonal | `month`, `week_of_year`, `sin_month`, `cos_month`, `sin_week`, `cos_week` |
| Events | `exam_period`, `is_pre_exam_week`, `is_new_year_jan`, `is_summer`, `special_event`, `is_holiday`, `sports_or_challenge`, `is_new_term` |
| Context | `temperature_c`, `previous_day_occupancy`, `rolling_3day_avg` |

**Key design decisions:**
- **Circular encoding** (`sin`/`cos`) for month and week so the model understands Dec → Jan as continuous
- **Chronological train/test split** (not random) to prevent future data leaking into training
- **Lag features** computed from actual data, not random noise

---

## 📈 Dashboard Pages

| Page | Description |
|---|---|
| **Welcome** | Today's crowd by time slot + best/worst slot recommendation |
| **Plan Visit** | Prediction tool with animated gauge + 80% confidence interval |
| **Trends** | Heatmap, exam vs normal comparison, daily crowd trend |
| **Data & Model** | Dataset snapshot, 5-model comparison chart, feature importances, training history |

---

## 🔄 Data Drift Handling

A `drift_factor()` function linearly increases base crowd by up to 15% over the dataset duration, simulating growing gym membership. The retrain pipeline appends real-position data (correct exam/holiday flags based on calendar position) and recomputes lag features from existing DB records before retraining.

---

## 📦 Tech Stack

`Python` · `Streamlit` · `scikit-learn` · `XGBoost` · `pandas` · `numpy` · `Plotly` · `SQLite` · `joblib`

---

## 📄 Documentation

Full technical documentation is available in `GymIQ_Technical_Documentation.docx`, covering data generation, feature engineering, model training rationale, evaluation metrics, and pipeline architecture.

---

*Sanskruti Sonawane · STSE203 · March 2026*