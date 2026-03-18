# dashboard/app.py
import os
import sqlite3
import json
from glob import glob

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px

# ---- PAGE CONFIG & THEME TWEAKS ----
st.set_page_config(
    page_title="GymIQ · Footfall Analytics",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .metric-container {
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        background-color: #111827;
        border: 1px solid #1f2937;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- CONSTANTS ----
DB_PATH = "data/project.db"
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
    "previous_day_occupancy",
]

# ---- HELPERS ----
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df


def get_latest_model_path():
    model_files = glob("models/model_*.pkl")
    if not model_files:
        return None
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

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("## 🏋️‍♀️ GymIQ")
    st.markdown("##### Footfall Analytics Dashboard")
    st.markdown("---")
    st.markdown("**Filters**")

    day_filter = st.selectbox(
        "Day of Week",
        ["All", "0 Mon", "1 Tue", "2 Wed", "3 Thu", "4 Fri", "5 Sat", "6 Sun"],
    )

    exam_filter = st.selectbox("Exam Period", ["Both", "Exam", "Normal"])

    st.markdown("---")
    page = st.radio(
        "Main Menu",
        ["Home", "Forecast", "Model Intelligence"],
        index=0,
    )

# ---- DATA & MODEL ----
df = load_data()

latest_model_path = get_latest_model_path()
metrics = load_latest_metrics()
model = None
if latest_model_path is not None:
    model = joblib.load(latest_model_path)

# apply simple filters to df for visuals
df_vis = df.copy()
if day_filter != "All":
    df_vis = df_vis[df_vis["day_of_week"] == int(day_filter[0])]
if exam_filter != "Both":
    flag = 1 if exam_filter == "Exam" else 0
    df_vis = df_vis[df_vis["exam_period"] == flag]

# ---- HOME PAGE ----
if page == "Home":
    st.markdown("### 🏠 Home · Gym Descriptive Analytics")

    total_sessions = len(df_vis)
    peak_occ = df_vis["occupancy_percentage"].max()
    avg_occ = df_vis["occupancy_percentage"].mean()
    median_occ = df_vis["occupancy_percentage"].median()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Sessions", f"{total_sessions:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Peak Occupancy", f"{peak_occ:.0f} %")
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Avg Occupancy", f"{avg_occ:.1f} %")
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Median Occupancy", f"{median_occ:.1f} %")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("----")

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Hourly Footfall Trend")
        df_hour = (
            df_vis.groupby("hour")["occupancy_percentage"]
            .mean()
            .reset_index()
            .sort_values("hour")
        )
        fig_hour = px.line(
            df_hour,
            x="hour",
            y="occupancy_percentage",
            markers=True,
            color_discrete_sequence=["#00C0F2"],
            labels={"hour": "Hour", "occupancy_percentage": "Avg occupancy %"},
            template="plotly_dark",
        )
        fig_hour.update_traces(line=dict(width=3))
        st.plotly_chart(fig_hour, use_container_width=True)

    with right:
        st.subheader("Average by Day of Week")
        df_day = (
            df_vis.groupby("day_of_week")["occupancy_percentage"]
            .mean()
            .reset_index()
            .sort_values("day_of_week")
        )
        fig_day = px.bar(
            df_day,
            x="day_of_week",
            y="occupancy_percentage",
            color="occupancy_percentage",
            color_continuous_scale="Turbo",
            labels={
                "day_of_week": "Day (0=Mon)",
                "occupancy_percentage": "Avg occupancy %",
            },
            template="plotly_dark",
        )
        fig_day.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_day, use_container_width=True)

    st.markdown("----")
    st.subheader("Sample Data")
    st.dataframe(df_vis.head(20))

# ---- FORECAST PAGE ----
elif page == "Forecast":
    st.markdown("### 🔮 Footfall Forecast")

    if model is None:
        st.error("No trained model found. Run train_model.py first.")
    else:
        col_form, col_chart = st.columns([1, 2])

        with col_form:
            st.markdown("#### Input Scenario")

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

            previous_day_occupancy = st.slider(
                "Previous day's same-hour occupancy (%)",
                0.0,
                100.0,
                50.0,
                1.0,
            )

            input_array = np.array(
                [
                    [
                        hour,
                        day_of_week,
                        exam_period_val,
                        temperature_c,
                        is_weekend,
                        special_event_val,
                        is_holiday_val,
                        sports_or_challenge_val,
                        is_new_term_val,
                        previous_day_occupancy,
                    ]
                ]
            )

            pred = model.predict(input_array)[0]
            pred = max(0, min(100, pred))

            if pred < 30:
                label = "Low"
            elif pred < 70:
                label = "Medium"
            else:
                label = "High"

            st.markdown("### ")
            st.metric("Predicted Occupancy", f"{pred:.1f} %", label)

        with col_chart:
            st.markdown("#### Next 6 Hours (simple scenario)")
            hours_ahead = list(range(hour, min(hour + 6, 23)))
            preds = []
            for h in hours_ahead:
                arr = input_array.copy()
                arr[0, 0] = h
                preds.append(max(0, min(100, model.predict(arr)[0])))
            df_fore = pd.DataFrame({"hour": hours_ahead, "predicted_occupancy": preds})
            fig_fore = px.line(
                df_fore,
                x="hour",
                y="predicted_occupancy",
                markers=True,
                color_discrete_sequence=["#F97316"],
                labels={
                    "hour": "Hour",
                    "predicted_occupancy": "Predicted occupancy %",
                },
                template="plotly_dark",
            )
            fig_fore.update_traces(line=dict(width=3))
            st.plotly_chart(fig_fore, use_container_width=True)

# ---- MODEL INTELLIGENCE PAGE ----
elif page == "Model Intelligence":
    st.markdown("### 🧠 Model Intelligence")

    if latest_model_path is None:
        st.error("No trained model found. Run train_model.py first.")
    else:
        st.markdown("#### Current Model")
        c1, c2, c3 = st.columns(3)
        c1.metric("Model File", os.path.basename(latest_model_path))
        if metrics:
            c2.metric("RMSE", f"{metrics['rmse']:.2f}")
            c3.metric("Trained At", metrics["timestamp"])

        st.markdown("----")
        st.markdown("#### Model History")

        if metrics:
            rows = []
            with open("model_history.json", "r") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            hist_df = pd.DataFrame(rows)
            hist_df = hist_df.sort_values("timestamp", ascending=False)
            st.dataframe(hist_df[["timestamp", "model_name", "rmse"]])
        else:
            st.info("No model history found.")
