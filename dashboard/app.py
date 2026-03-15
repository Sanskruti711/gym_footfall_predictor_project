# app.py
import os
import sqlite3
import json
from glob import glob

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

DB_PATH = "project.db"
TABLE_NAME = "gym_footfall"

FEATURE_COLS = [
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

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df

def get_latest_model_path():
    model_files = glob("models/model_*.pkl")
    if not model_files:
        return None
    # sort by filename (timestamp inside)
    model_files.sort()
    return model_files[-1]

def load_latest_metrics():
    if not os.path.exists("model_history.json"):
        return None
    last_line = None
    with open("model_history.json", "r") as f:
        for line in f:
            if line.strip():
                last_line = line
    if last_line:
        return json.loads(last_line)
    return None

st.title("College Gym Footfall Prediction Dashboard")

df = load_data()

st.subheader("Data Overview")
st.write(df.head(15))

st.subheader("Basic Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.write("Average occupancy by hour")
    hour_group = df.groupby("hour")["occupancy_percentage"].mean().reset_index()
    fig1, ax1 = plt.subplots()
    ax1.plot(hour_group["hour"], hour_group["occupancy_percentage"], marker="o")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Avg occupancy %")
    ax1.set_title("Avg occupancy by hour")
    st.pyplot(fig1)

with col2:
    st.write("Average occupancy by day_of_week")
    dow_group = df.groupby("day_of_week")["occupancy_percentage"].mean().reset_index()
    fig2, ax2 = plt.subplots()
    ax2.bar(dow_group["day_of_week"], dow_group["occupancy_percentage"])
    ax2.set_xlabel("Day of week (0=Mon)")
    ax2.set_ylabel("Avg occupancy %")
    ax2.set_title("Avg occupancy by day_of_week")
    st.pyplot(fig2)

st.subheader("Model Info")
latest_model_path = get_latest_model_path()
metrics = load_latest_metrics()

if latest_model_path is None:
    st.error("No trained model found. Run train_model.py first.")
else:
    st.success(f"Using model file: {latest_model_path}")
    if metrics:
        st.write(f"Model name: {metrics['model_name']}")
        st.write(f"RMSE: {metrics['rmse']:.2f}")
        st.write(f"Trained at: {metrics['timestamp']}")

    model = joblib.load(latest_model_path)

    st.subheader("Make a Prediction")

    hour = st.slider("Hour (24h)", 6, 22, 18)
    day_of_week = st.slider("Day of week (0=Mon .. 6=Sun)", 0, 6, 2)
    exam_period = st.selectbox("Exam period?", ["No", "Yes"])
    exam_period_val = 1 if exam_period == "Yes" else 0

    temperature_c = st.slider("Temperature (°C)", 10.0, 40.0, 28.0, 0.5)
    is_weekend = 1 if day_of_week in [5, 6] else 0

    special_event = st.selectbox("Special event?", ["No", "Yes"])
    special_event_val = 1 if special_event == "Yes" else 0

    is_holiday = st.selectbox("Holiday?", ["No", "Yes"])
    is_holiday_val = 1 if is_holiday == "Yes" else 0

    sports_or_challenge = st.selectbox("Sports/gym challenge?", ["No", "Yes"])
    sports_or_challenge_val = 1 if sports_or_challenge == "Yes" else 0

    is_new_term = st.selectbox("New term period?", ["No", "Yes"])
    is_new_term_val = 1 if is_new_term == "Yes" else 0

    previous_day_occupancy = st.slider("Previous day's same-hour occupancy (%)", 0.0, 100.0, 50.0, 1.0)

    if st.button("Predict crowd"):
        input_data = np.array([[
            hour,
            day_of_week,
            exam_period_val,
            temperature_c,
            is_weekend,
            special_event_val,
            is_holiday_val,
            sports_or_challenge_val,
            is_new_term_val,
            previous_day_occupancy
        ]])

        pred = model.predict(input_data)[0]
        pred = max(0, min(100, pred))

        if pred < 30:
            label = "Low"
        elif pred < 70:
            label = "Medium"
        else:
            label = "High"

        st.write(f"Predicted occupancy: **{pred:.1f}%**")
        st.write(f"Crowd level: **{label}**")
