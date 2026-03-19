# dashboard/app.py
import ast
import os
import sqlite3
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="GymIQ · Footfall Analytics",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# STYLES + ANIMATIONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

* { font-family: 'DM Sans', sans-serif; }
h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 1rem 2rem; }
html, body, [data-testid="stAppViewContainer"] { background-color: #07080f; }

/* ─── SIDEBAR ─── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d0f1a 0%,#080a14 100%);
    border-right: 1px solid #1a1d2e;
}
[data-testid="stSidebar"] * { color: #c8cde8 !important; }

/* ─── SIDEBAR NAV HOVER ANIMATIONS ─── */
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.83rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.45rem 0.75rem;
    border-radius: 8px;
    transition: background 0.25s ease, color 0.25s ease, transform 0.2s ease,
                box-shadow 0.25s ease, letter-spacing 0.25s ease;
    cursor: pointer;
    display: block;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99,102,241,0.15) !important;
    color: #a5b4fc !important;
    transform: translateX(4px);
    box-shadow: -3px 0 0 #6366f1;
    letter-spacing: 0.08em;
}

/* ─── PAGE LOAD ENTRANCE ANIMATIONS ─── */
@keyframes fadeSlideUp {
    from { opacity:0; transform:translateY(22px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeSlideLeft {
    from { opacity:0; transform:translateX(-20px); }
    to   { opacity:1; transform:translateX(0); }
}
@keyframes fadeIn {
    from { opacity:0; }
    to   { opacity:1; }
}
@keyframes glowPulse {
    0%,100% { box-shadow: 0 0 0 rgba(99,102,241,0); }
    50%      { box-shadow: 0 0 18px rgba(99,102,241,0.35); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
@keyframes countUp {
    from { opacity:0; transform:scale(0.85); }
    to   { opacity:1; transform:scale(1); }
}

.animate-slide-up {
    animation: fadeSlideUp 0.55s cubic-bezier(0.22,1,0.36,1) both;
}
.animate-slide-up-2 { animation: fadeSlideUp 0.55s cubic-bezier(0.22,1,0.36,1) 0.1s both; }
.animate-slide-up-3 { animation: fadeSlideUp 0.55s cubic-bezier(0.22,1,0.36,1) 0.2s both; }
.animate-slide-up-4 { animation: fadeSlideUp 0.55s cubic-bezier(0.22,1,0.36,1) 0.3s both; }
.animate-slide-left { animation: fadeSlideLeft 0.5s cubic-bezier(0.22,1,0.36,1) 0.15s both; }
.animate-fade       { animation: fadeIn 0.7s ease 0.25s both; }

/* ─── STAT CARDS with GLOW PULSE ─── */
.stat-card {
    background: linear-gradient(135deg,#0f1120 0%,#131629 100%);
    border: 1px solid #1e2138;
    border-radius: 14px;
    padding: 1.15rem 1.35rem;
    position: relative;
    overflow: hidden;
    animation: fadeSlideUp 0.5s cubic-bezier(0.22,1,0.36,1) both, glowPulse 3s ease-in-out 0.6s 2;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(99,102,241,0.18);
}
.stat-card::before {
    content:'';
    position:absolute;
    top:0;left:0;right:0;
    height:2px;
    background: linear-gradient(90deg,#6366f1,#06b6d4,#6366f1);
    background-size: 200% auto;
    animation: shimmer 3s linear infinite;
}
.stat-label {
    font-size:0.69rem; font-weight:500; letter-spacing:0.12em;
    text-transform:uppercase; color:#3d4270; margin-bottom:0.35rem;
}
.stat-value {
    font-family:'Syne',sans-serif;
    font-size:1.65rem; font-weight:700; color:#e2e5f5; line-height:1;
    animation: countUp 0.4s cubic-bezier(0.22,1,0.36,1) 0.3s both;
}

/* ─── GAUGE ENTRANCE ─── */
@keyframes gaugeReveal {
    from { opacity:0; transform:scale(0.88) translateY(12px); }
    to   { opacity:1; transform:scale(1) translateY(0); }
}
.gauge-wrapper {
    animation: gaugeReveal 0.6s cubic-bezier(0.34,1.56,0.64,1) both;
}

/* ─── CARDS ─── */
.info-card {
    background: linear-gradient(135deg,#0e1628 0%,#0a1f1c 100%);
    border: 1px solid #1a3a35;
    border-radius: 14px;
    padding: 1.5rem 1.7rem;
    height:100%;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.info-card:hover {
    border-color:#10b981;
    box-shadow: 0 0 24px rgba(16,185,129,0.1);
}
.pipeline-card {
    background:#0c0e1a;
    border:1px solid #1a1d2e;
    border-radius:14px;
    padding:1.2rem 1.4rem;
    height:100%;
}
.pipeline-step {
    display:flex; align-items:flex-start; gap:0.6rem;
    padding:0.5rem 0; border-bottom:1px solid #13152a;
    font-size:0.83rem; color:#8b91b8;
    animation: fadeSlideLeft 0.4s ease both;
}
.pipeline-step:nth-child(1) { animation-delay: 0.1s; }
.pipeline-step:nth-child(2) { animation-delay: 0.2s; }
.pipeline-step:nth-child(3) { animation-delay: 0.3s; }
.pipeline-step:nth-child(4) { animation-delay: 0.4s; }
.pipeline-step:last-child { border-bottom:none; }
.step-num {
    background:linear-gradient(135deg,#6366f1,#4f46e5);
    color:white; font-family:'Syne',sans-serif;
    font-size:0.65rem; font-weight:700;
    width:18px; height:18px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    flex-shrink:0; margin-top:1px;
}

/* ─── RECOMMEND CARD ─── */
.recommend-card {
    background:linear-gradient(135deg,#091a14 0%,#0a1628 100%);
    border:1px solid #1a3a2a; border-left:3px solid #10b981;
    border-radius:10px; padding:0.85rem 1rem; margin-top:0.8rem;
    font-size:0.87rem; color:#a7f3d0; line-height:1.5;
    animation: fadeSlideUp 0.45s cubic-bezier(0.22,1,0.36,1) both;
}

/* ─── BADGES ─── */
.crowd-badge {
    display:inline-block; padding:0.3rem 0.9rem; border-radius:20px;
    font-family:'Syne',sans-serif; font-size:0.75rem; font-weight:700;
    letter-spacing:0.08em; text-transform:uppercase;
    animation: countUp 0.35s cubic-bezier(0.22,1,0.36,1) 0.2s both;
}
.badge-low    { background:rgba(52,211,153,0.12);  color:#34d399; border:1px solid #34d39940; }
.badge-medium { background:rgba(251,191,36,0.12);  color:#fbbf24; border:1px solid #fbbf2440; }
.badge-high   { background:rgba(248,113,113,0.12); color:#f87171; border:1px solid #f8717140; }

/* ─── MISC ─── */
.section-header {
    font-family:'Syne',sans-serif; font-size:0.68rem; font-weight:700;
    letter-spacing:0.14em; text-transform:uppercase; color:#3d4270;
    margin-bottom:0.75rem; padding-bottom:0.4rem; border-bottom:1px solid #13152a;
}
.custom-divider {
    height:1px;
    background:linear-gradient(90deg,transparent,#1e2138,transparent);
    margin:1.6rem 0;
}
.page-title {
    font-family:'Syne',sans-serif; font-size:1.65rem; font-weight:800;
    color:#e2e5f5; letter-spacing:-0.02em; margin-bottom:0.2rem;
    animation: fadeSlideUp 0.5s cubic-bezier(0.22,1,0.36,1) both;
}
.page-subtitle {
    font-size:0.82rem; color:#3d4270; margin-bottom:1.5rem;
    animation: fadeSlideUp 0.5s cubic-bezier(0.22,1,0.36,1) 0.08s both;
}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
DB_PATH    = "data/project.db"
TABLE_NAME = "gym_footfall"

FEATURE_COLS = [
    "day_of_week", "is_weekend", "time_block",
    "month", "week_of_year",
    "sin_month", "cos_month", "sin_week", "cos_week",
    "exam_period", "is_pre_exam_week",
    "is_new_year_jan", "is_summer",
    "special_event", "is_holiday", "sports_or_challenge", "is_new_term",
    "temperature_c",
    "previous_day_occupancy", "rolling_3day_avg",
]
TARGET_COL        = "people_count"
TIME_BLOCK_LABELS = {0:"6–8 AM", 1:"8–12 PM", 2:"12–4 PM", 3:"4–10 PM"}
DAY_LABELS        = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
BAR_COLOR_SCALE   = "Turbo"


def get_seasonal_features(date: datetime) -> dict:
    """
    Auto-derive all temporal/seasonal features from today's date.
    Called at prediction time so the user never has to enter these manually.
    """
    month        = date.month
    week_of_year = min(int(date.strftime("%W")), 52) + 1

    sin_month = round(np.sin(2 * np.pi * month / 12), 4)
    cos_month = round(np.cos(2 * np.pi * month / 12), 4)
    sin_week  = round(np.sin(2 * np.pi * week_of_year / 52), 4)
    cos_week  = round(np.cos(2 * np.pi * week_of_year / 52), 4)

    # New Year resolution window: Jan 1–21
    is_new_year_jan = 1 if (month == 1 and date.day <= 21) else 0

    # Summer quiet: June + July
    is_summer = 1 if month in [6, 7] else 0

    return {
        "month"          : month,
        "week_of_year"   : week_of_year,
        "sin_month"      : sin_month,
        "cos_month"      : cos_month,
        "sin_week"       : sin_week,
        "cos_week"       : cos_week,
        "is_new_year_jan": is_new_year_jan,
        "is_summer"      : is_summer,
    }

# ── LOADERS ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df

@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)

def get_latest_model_path() -> str | None:
    files = sorted(glob("models/model_*.pkl"))
    return files[-1] if files else None

@st.cache_data(ttl=300)
def load_model_history() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        hist = pd.read_sql("SELECT * FROM model_history ORDER BY timestamp DESC", conn)
    except Exception:
        hist = pd.DataFrame()
    conn.close()
    return hist

def predict_with_interval(model, input_df: pd.DataFrame):
    """Returns (point, lower, upper) all as integers."""
    point = int(round(max(0, model.predict(input_df)[0])))
    estimators = getattr(model, "estimators_", None)
    if estimators is not None:
        try:
            flat = [t[0] if hasattr(t, "__len__") else t for t in estimators]
            tree_preds = np.array([t.predict(input_df) for t in flat])
            lower = int(round(max(0, float(np.percentile(tree_preds, 10)))))
            upper = int(round(float(np.percentile(tree_preds, 90))))
            return point, lower, upper
        except Exception:
            pass
    return point, int(round(point * 0.85)), int(round(point * 1.15))

def crowd_info(n: int) -> tuple[str, str, str]:
    if n < 20:   return "Low",    "badge-low",    "#34d399"
    elif n < 50: return "Medium", "badge-medium", "#fbbf24"
    else:        return "High",   "badge-high",   "#f87171"

def gauge_chart(value: int, lower: int, upper: int, capacity: int = 80):
    label, _, colour = crowd_info(value)
    fig = go.Figure(go.Indicator(
        mode   = "gauge+number",
        value  = value,
        number = {"suffix":" people","font":{"size":34,"color":"#e2e5f5"},"valueformat":"d"},
        gauge  = {
            "axis"  : {"range":[0,capacity],"tickwidth":1,"tickcolor":"#2a2d45","tickfont":{"color":"#4b5280"}},
            "bar"   : {"color":colour,"thickness":0.22},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth":0,
            "steps" : [
                {"range":[0,  20],"color":"#061810"},
                {"range":[20, 50],"color":"#1a1200"},
                {"range":[50,capacity],"color":"#1a0808"},
            ],
            "threshold":{"line":{"color":colour,"width":3},"thickness":0.75,"value":value},
        },
        title={"text":f"<b>{label}</b>","font":{"size":13,"color":colour}},
    ))
    fig.update_layout(
        height=240, margin=dict(t=50,b=35,l=20,r=20),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e5f5",
    )
    fig.add_annotation(
        text=f"80% range: {lower} – {upper} people",
        xref="paper", yref="paper", x=0.5, y=-0.15,
        showarrow=False, font=dict(size=11,color="#3d4270"),
    )
    return fig

def plotly_style() -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#8b91b8"),
        margin=dict(t=10,b=20,l=10,r=10), template="plotly_dark",
    )

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1.2rem 0;'>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                    color:#e2e5f5;letter-spacing:-0.01em;'>🏋️ GymIQ</div>
        <div style='font-size:0.7rem;color:#3d4270;letter-spacing:0.09em;
                    text-transform:uppercase;margin-top:0.2rem;'>Smart Footfall Planner</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["Welcome","Plan Visit","Trends","Data & Model"],
                    label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

# ── LOAD ──────────────────────────────────────────────────────────────────────
df                = load_data()
latest_model_path = get_latest_model_path()
model             = load_model(latest_model_path) if latest_model_path else None
df_vis            = df.copy()

# ══════════════════════════════════════════════════════════════════════════════
# WELCOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Welcome":
    st.markdown('<div class="page-title animate-slide-up">Welcome to GymIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle animate-slide-up-2">ML-powered gym crowd predictions — find your perfect time to train</div>', unsafe_allow_html=True)

    left, right = st.columns([3,2], gap="large")
    with left:
        st.markdown("""
        <div class="info-card animate-slide-up-3">
            <div style='font-size:0.68rem;font-weight:700;letter-spacing:0.12em;
                        text-transform:uppercase;color:#10b981;margin-bottom:0.6rem;'>About GymIQ</div>
            <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;
                        color:#e2e5f5;line-height:1.35;margin-bottom:0.75rem;'>
                Know before you go.<br>Skip the crowd.
            </div>
            <div style='font-size:0.85rem;color:#6b7280;line-height:1.65;'>
                GymIQ predicts how busy your college gym will be across four daily time slots —
                using a machine learning model trained on footfall patterns, exam periods,
                weather, events, and more.
            </div>
        </div>""", unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="pipeline-card animate-slide-up-4">
            <div style='font-size:0.68rem;font-weight:700;letter-spacing:0.1em;
                        text-transform:uppercase;color:#3d4270;margin-bottom:0.8rem;'>How it works</div>
            <div class="pipeline-step"><div class="step-num">1</div>Footfall data stored in SQLite</div>
            <div class="pipeline-step"><div class="step-num">2</div>5 models trained, best one saved</div>
            <div class="pipeline-step"><div class="step-num">3</div>Dashboard loads the winning model</div>
            <div class="pipeline-step"><div class="step-num">4</div>Pick a slot → get crowd estimate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="custom-divider animate-fade"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header animate-fade">Today\'s average crowd by time slot</div>', unsafe_allow_html=True)

    today_dow = datetime.today().weekday()
    df_today  = df[df["day_of_week"] == today_dow]
    if not df_today.empty:
        db = df_today.groupby("time_block")["people_count"].mean().reset_index()
        db["people_count"] = db["people_count"].round(0).astype(int)
        db["block_label"]  = db["time_block"].map(TIME_BLOCK_LABELS)
        fig = px.bar(db, x="block_label", y="people_count",
                     color="people_count", color_continuous_scale=BAR_COLOR_SCALE,
                     labels={"block_label":"","people_count":"Avg people"}, text="people_count")
        fig.update_traces(textposition="outside", textfont=dict(family="Syne",size=13,color="#e2e5f5"))
        fig.update_layout(**plotly_style(), coloraxis_showscale=False, height=290,
                          xaxis=dict(tickfont=dict(size=12,family="Syne")))
        st.plotly_chart(fig, use_container_width=True)

        q = db.loc[db["people_count"].idxmin()]
        b = db.loc[db["people_count"].idxmax()]
        st.markdown(f"""
        <div class="recommend-card">
            💡 <strong>Today:</strong>
            Quietest at <strong>{q['block_label']}</strong> (~{int(q['people_count'])} people)
            &nbsp;·&nbsp;
            Busiest at <strong>{b['block_label']}</strong> (~{int(b['people_count'])} people)
        </div>""", unsafe_allow_html=True)

    st.markdown("<br><span style='font-size:0.8rem;color:#3d4270;'>→ Head to <strong style='color:#6366f1'>Plan Visit</strong> to get a personalised prediction</span>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PLAN VISIT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Plan Visit":
    st.markdown('<div class="page-title animate-slide-up">Plan Your Visit</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle animate-slide-up-2">Fill in today\'s conditions and get a crowd prediction</div>', unsafe_allow_html=True)

    if model is None:
        st.error("No trained model found. Run `scripts/train_model.py` first.")
        st.stop()

    col_form, col_right = st.columns([1,1], gap="large")

    with col_form:
        st.markdown('<div class="section-header animate-slide-up-3">Conditions</div>', unsafe_allow_html=True)
        day_of_week = st.selectbox("Day of week", options=list(range(7)), format_func=lambda d: DAY_LABELS[d])
        time_block  = st.selectbox("Time slot",   options=list(range(4)), format_func=lambda b: TIME_BLOCK_LABELS[b])

        c1, c2 = st.columns(2)
        with c1:
            exam_period_val   = 1 if st.selectbox("Exam period?",  ["No","Yes"]) == "Yes" else 0
            is_holiday_val    = 1 if st.selectbox("Holiday?",       ["No","Yes"]) == "Yes" else 0
            new_term_val      = 1 if st.selectbox("New semester?",  ["No","Yes"]) == "Yes" else 0
        with c2:
            special_event_val = 1 if st.selectbox("Special event?", ["No","Yes"]) == "Yes" else 0
            sports_val        = 1 if st.selectbox("Gym challenge?", ["No","Yes"]) == "Yes" else 0

        temperature_c  = st.slider("Approx temperature (°C)", 10.0, 42.0, 28.0, 0.5)
        is_weekend_val = 1 if day_of_week in [5,6] else 0
        hist_mean      = float(df[(df["day_of_week"]==day_of_week)&(df["time_block"]==time_block)]["people_count"].mean()) if not df.empty else 30.0

        # Auto-derive seasonal features from today's actual date
        seasonal = get_seasonal_features(datetime.today())

        predict_btn    = st.button("Predict crowd", type="primary", use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header animate-slide-up-3">Typical crowd for this day</div>', unsafe_allow_html=True)
        df_day = df[df["day_of_week"] == day_of_week]
        if not df_day.empty:
            dbd = df_day.groupby("time_block")["people_count"].mean().reset_index()
            dbd["people_count"] = dbd["people_count"].round(0).astype(int)
            dbd["block_label"]  = dbd["time_block"].map(TIME_BLOCK_LABELS)
            fig2 = px.bar(dbd, x="block_label", y="people_count",
                          color="people_count", color_continuous_scale=BAR_COLOR_SCALE,
                          labels={"block_label":"","people_count":"Avg people"}, text="people_count")
            fig2.update_traces(textposition="outside", textfont=dict(family="Syne",size=12,color="#e2e5f5"))
            fig2.update_layout(**plotly_style(), coloraxis_showscale=False, height=250,
                               xaxis=dict(tickfont=dict(size=11,family="Syne")))
            st.plotly_chart(fig2, use_container_width=True)

        if predict_btn:
            input_df = pd.DataFrame([{
                "day_of_week":day_of_week,"is_weekend":is_weekend_val,
                "time_block":time_block,
                **seasonal,                    # month, week, sin/cos, new_year, summer
                "exam_period":exam_period_val,
                "is_pre_exam_week": 0,         # user can't easily know this; default off
                "special_event":special_event_val,
                "is_holiday":is_holiday_val,"sports_or_challenge":sports_val,
                "is_new_term":new_term_val,
                "temperature_c":temperature_c,
                "previous_day_occupancy":round(hist_mean,1),
                "rolling_3day_avg":round(hist_mean,1),
            }])[FEATURE_COLS]

            point, lower, upper = predict_with_interval(model, input_df)
            label, badge_cls, _ = crowd_info(point)

            st.markdown('<div class="section-header" style="margin-top:1rem;">Prediction</div>', unsafe_allow_html=True)
            # Gauge with entrance animation wrapper
            st.markdown('<div class="gauge-wrapper">', unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(point, lower, upper), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="text-align:center;margin-top:-0.4rem;"><span class="crowd-badge {badge_cls}">{label}</span></div>',
                        unsafe_allow_html=True)

            # Better time suggestion
            all_blocks = [(tb, int(round(max(0, model.predict(input_df.copy().assign(time_block=tb))[0])))) for tb in range(4)]
            best_tb, best_p = min(all_blocks, key=lambda x: x[1])
            if best_tb != time_block:
                st.markdown(f"""
                <div class="recommend-card">
                    💡 <strong>Better slot today:</strong>
                    <strong>{TIME_BLOCK_LABELS[best_tb]}</strong> — ~{best_p} people
                    ({point - best_p} fewer)
                </div>""", unsafe_allow_html=True)

            h = load_model_history()
            if not h.empty and "mape" in h.columns:
                mape_val = float(h.iloc[0]["mape"])
                if mape_val < 200:
                    st.markdown(f"<div style='font-size:0.74rem;color:#3d4270;margin-top:0.5rem;text-align:center;'>Typical prediction error: ~{mape_val:.1f}%</div>",
                                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Trends":
    st.markdown('<div class="page-title animate-slide-up">Usage Trends</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle animate-slide-up-2">Historical footfall patterns across time slots, days, and conditions</div>', unsafe_allow_html=True)

    # Stat cards — each gets staggered animation via inline style
    cols = st.columns(4, gap="small")
    delays = ["0s","0.1s","0.2s","0.3s"]
    for col, (lbl, val), delay in zip(cols, [
        ("Total records",  f"{len(df_vis):,}"),
        ("Peak crowd",     f"{int(df_vis['people_count'].max())} people"),
        ("Avg crowd",      f"{int(round(df_vis['people_count'].mean()))} people"),
        ("Median crowd",   f"{int(round(df_vis['people_count'].median()))} people"),
    ], delays):
        with col:
            st.markdown(f"""
            <div class="stat-card" style="animation-delay:{delay};">
                <div class="stat-label">{lbl}</div>
                <div class="stat-value">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="custom-divider animate-fade"></div>', unsafe_allow_html=True)

    left, right = st.columns([3,2], gap="large")
    with left:
        st.markdown('<div class="section-header animate-slide-up-3">Avg crowd by time slot & day</div>', unsafe_allow_html=True)
        dh = df_vis.groupby(["day_of_week","time_block"])["people_count"].mean().reset_index()
        dh["people_count"] = dh["people_count"].round(0).astype(int)
        dh["day_label"]    = dh["day_of_week"].map(DAY_LABELS)
        dh["block_label"]  = dh["time_block"].map(TIME_BLOCK_LABELS)
        fig_h = px.density_heatmap(dh, x="block_label", y="day_label", z="people_count",
                                   color_continuous_scale="Viridis",
                                   labels={"block_label":"","day_label":"","people_count":"Avg people"})
        fig_h.update_layout(**plotly_style(), height=310)
        st.plotly_chart(fig_h, use_container_width=True)

    with right:
        st.markdown('<div class="section-header animate-slide-up-3">Exam vs Normal</div>', unsafe_allow_html=True)
        de = df_vis.groupby(["exam_period","time_block"])["people_count"].mean().reset_index()
        de["people_count"] = de["people_count"].round(0).astype(int)
        de["Condition"]    = de["exam_period"].map({0:"Normal",1:"Exam"})
        de["block_label"]  = de["time_block"].map(TIME_BLOCK_LABELS)
        fig_e = px.bar(de, x="block_label", y="people_count", color="Condition", barmode="group",
                       color_discrete_sequence=["#6366f1","#f97316"],
                       labels={"block_label":"","people_count":"Avg people"}, text="people_count")
        fig_e.update_traces(textposition="outside", textfont=dict(size=11))
        fig_e.update_layout(**plotly_style(), height=310)
        st.plotly_chart(fig_e, use_container_width=True)

    st.markdown('<div class="custom-divider animate-fade"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header animate-fade">Daily average crowd over time</div>', unsafe_allow_html=True)
    dd = df_vis.groupby("day_index")["people_count"].mean().reset_index()
    dd["people_count"] = dd["people_count"].round(0).astype(int)
    fig_l = px.line(dd, x="day_index", y="people_count",
                    labels={"day_index":"Day","people_count":"Avg people"})
    fig_l.update_traces(line_color="#6366f1", line_width=2)
    fig_l.update_layout(**plotly_style(), height=250)
    st.plotly_chart(fig_l, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data & Model":
    st.markdown('<div class="page-title animate-slide-up">Data & Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle animate-slide-up-2">Dataset snapshot, model metrics, and feature importance</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header animate-slide-up-3">Dataset snapshot</div>', unsafe_allow_html=True)
    st.caption(f"{len(df):,} total records")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown('<div class="custom-divider animate-fade"></div>', unsafe_allow_html=True)

    if latest_model_path is None:
        st.error("No trained model found. Run `scripts/train_model.py` first.")
    else:
        hist   = load_model_history()
        latest = hist.iloc[0] if not hist.empty else None

        st.markdown('<div class="section-header animate-slide-up-3">Current model</div>', unsafe_allow_html=True)
        delays = ["0s","0.08s","0.16s","0.24s"]
        for col, (lbl,val), delay in zip(st.columns(4,gap="small"), [
            ("Model file",    os.path.basename(latest_model_path)),
            ("Model type",    latest.get("model_name","—") if latest is not None else "—"),
            ("RMSE (people)", f"{float(latest['rmse']):.2f}" if latest is not None else "—"),
            ("MAPE",          f"{float(latest['mape']):.1f}%" if latest is not None and float(latest.get('mape',999))<200 else "—"),
        ], delays):
            with col:
                st.markdown(f'<div class="stat-card" style="animation-delay:{delay};"><div class="stat-label">{lbl}</div><div class="stat-value" style="font-size:0.95rem;word-break:break-all;">{val}</div></div>',
                            unsafe_allow_html=True)

        # Model comparison
        if latest is not None and "all_model_rmses" in hist.columns:
            raw = latest.get("all_model_rmses", None)
            if raw:
                try:
                    cd = ast.literal_eval(raw)
                    cf = pd.DataFrame(cd.items(), columns=["Model","RMSE"]).sort_values("RMSE")
                    cf["Label"] = cf.apply(lambda r: f"✅ {r['Model']}" if r["RMSE"]==cf["RMSE"].min() else r["Model"], axis=1)
                    st.markdown('<div class="custom-divider animate-fade"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-header animate-fade">Model comparison — 5 models, last training run</div>', unsafe_allow_html=True)
                    fig_c = px.bar(cf, x="Label", y="RMSE", color="RMSE",
                                   color_continuous_scale="RdYlGn_r",
                                   labels={"RMSE":"RMSE (people)","Label":""}, text="RMSE")
                    fig_c.update_traces(texttemplate="%{text:.2f}", textposition="outside",
                                        textfont=dict(family="Syne",size=13))
                    fig_c.update_layout(**plotly_style(), coloraxis_showscale=False, height=290)
                    st.plotly_chart(fig_c, use_container_width=True)
                except Exception:
                    pass

        # Feature importance
        inner = model.named_steps.get("ridge", model) if hasattr(model, "named_steps") else model
        if hasattr(inner, "feature_importances_"):
            st.markdown('<div class="custom-divider animate-fade"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header animate-fade">Feature importance</div>', unsafe_allow_html=True)
            fi = pd.DataFrame({"Feature":FEATURE_COLS,"Importance":inner.feature_importances_}).sort_values("Importance")
            fig_i = px.bar(fi, x="Importance", y="Feature", orientation="h",
                           color="Importance", color_continuous_scale="Teal",
                           labels={"Importance":"Relative importance"}, text="Importance")
            fig_i.update_traces(texttemplate="%{text:.3f}", textposition="outside", textfont=dict(size=10))
            fig_i.update_layout(**plotly_style(), coloraxis_showscale=False, height=380)
            st.plotly_chart(fig_i, use_container_width=True)

        # History table
        st.markdown('<div class="custom-divider animate-fade"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header animate-fade">Training history</div>', unsafe_allow_html=True)
        if not hist.empty:
            dc = [c for c in ["timestamp","model_name","rmse","mape","train_rows","test_rows"] if c in hist.columns]
            st.dataframe(hist[dc], use_container_width=True)
        else:
            st.info("No model history found.")