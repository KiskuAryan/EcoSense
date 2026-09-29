"""
EcoSense Dashboard — Maintenance Insights Page
Sensor health scores, maintenance scheduling, and cost analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import sys

st.set_page_config(page_title="Maintenance | EcoSense", page_icon="🔧", layout="wide")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.feature_engineering import compute_sensor_health_score

# ─── Load data ───
@st.cache_data
def load_data():
    base = Path(__file__).parent.parent.parent / "data"
    water = pd.read_csv(base / "raw" / "water_quality.csv", parse_dates=["timestamp"])
    air = pd.read_csv(base / "raw" / "air_quality.csv", parse_dates=["timestamp"])
    anomalies = pd.read_csv(base / "anomalies" / "injected_anomalies.csv",
                            parse_dates=["start_time", "end_time"])
    with open(base / "raw" / "sensor_config.json") as f:
        config = json.load(f)
    return water, air, anomalies, config

@st.cache_data
def compute_health(water_df, air_df, sensor_config):
    """Compute health scores for all sensors."""
    water_sensors = ["pH", "turbidity_ntu", "dissolved_oxygen_mg_l", "temperature_c", "conductivity_us_cm"]
    air_sensors = ["co2_ppm", "nox_ppm", "so2_ppm", "pm25_ug_m3"]
    
    water_ranges = {s: (c["normal_min"], c["normal_max"]) for s, c in sensor_config["water_quality"].items()}
    air_ranges = {s: (c["normal_min"], c["normal_max"]) for s, c in sensor_config["air_quality"].items()}
    
    water_health = compute_sensor_health_score(water_df, water_sensors, water_ranges)
    air_health = compute_sensor_health_score(air_df, air_sensors, air_ranges)
    
    return water_health, air_health

water_df, air_df, anomaly_log, sensor_config = load_data()
water_health, air_health = compute_health(water_df, air_df, sensor_config)

water_sensors = ["pH", "turbidity_ntu", "dissolved_oxygen_mg_l", "temperature_c", "conductivity_us_cm"]
air_sensors = ["co2_ppm", "nox_ppm", "so2_ppm", "pm25_ug_m3"]

# ─── Page Header ───
st.markdown("# 🔧 Sensor Health Scoring & Condition-Based Maintenance")
st.markdown("*Continuous condition monitoring, degradation trends, and scenario-based scheduling*")

st.info(
    "💡 **Methodology Context**: This system provides **Condition-Based Maintenance (CBM)** "
    "using multi-factor health scoring (normal envelope dwell time, boundary distance, rolling variance, and data availability) "
    "and empirical degradation slopes. It is not an algorithmic Remaining Useful Life (RUL) regression model; "
    "formal deep-learning RUL is proposed as a future expansion requiring auxiliary vibration and fluid telemetry."
)
st.markdown("---")

# ─── Current Health Scores ───
st.markdown("### 💚 Current Sensor Health Scores")
st.caption("Standardized Condition Framework: 🟢 Optimal (≥85) | 🟡 Normal (70–84) | 🟠 Degradation Warning (50–69) | 🔴 Critical (<50)")

def get_current_health(health_df, sensor):
    """Get the latest health score (average of last 24h)."""
    col = f"{sensor}_health_score"
    if col in health_df.columns:
        return health_df[col].tail(96).mean()
    return 50.0

def health_color(score):
    if score >= 85:
        return "#64ffda", "Optimal"
    elif score >= 70:
        return "#ffd93d", "Normal"
    elif score >= 50:
        return "#fd79a8", "Degradation Warning"
    else:
        return "#ff6b6b", "Critical Risk"

# Water quality health gauges
st.markdown("#### 💧 Water Quality Sensors")
wcols = st.columns(5)

for idx, sensor in enumerate(water_sensors):
    score = get_current_health(water_health, sensor)
    color, status = health_color(score)
    
    with wcols[idx]:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": sensor.replace("_", " ").title(), "font": {"size": 12, "color": "#8892b0"}},
            number={"font": {"size": 28, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8892b0"},
                "bar": {"color": color},
                "bgcolor": "rgba(0,0,0,0.2)",
                "bordercolor": "rgba(255,255,255,0.1)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(255,107,107,0.15)"},
                    {"range": [50, 70], "color": "rgba(253,121,168,0.12)"},
                    {"range": [70, 85], "color": "rgba(255,217,61,0.1)"},
                    {"range": [85, 100], "color": "rgba(100,255,218,0.1)"}
                ],
                "threshold": {
                    "line": {"color": "#ff6b6b", "width": 2},
                    "thickness": 0.8,
                    "value": 50
                }
            }
        ))
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8892b0")
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<p style='text-align:center; color:{color};'><b>{status}</b></p>",
                    unsafe_allow_html=True)

# Air quality health gauges
st.markdown("#### 🌫️ Air Quality Sensors")
acols = st.columns(4)

for idx, sensor in enumerate(air_sensors):
    score = get_current_health(air_health, sensor)
    color, status = health_color(score)
    
    with acols[idx]:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": sensor.replace("_", " ").upper(), "font": {"size": 12, "color": "#8892b0"}},
            number={"font": {"size": 28, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8892b0"},
                "bar": {"color": color},
                "bgcolor": "rgba(0,0,0,0.2)",
                "bordercolor": "rgba(255,255,255,0.1)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(255,107,107,0.15)"},
                    {"range": [50, 70], "color": "rgba(253,121,168,0.12)"},
                    {"range": [70, 85], "color": "rgba(255,217,61,0.1)"},
                    {"range": [85, 100], "color": "rgba(100,255,218,0.1)"}
                ],
                "threshold": {
                    "line": {"color": "#ff6b6b", "width": 2},
                    "thickness": 0.8,
                    "value": 50
                }
            }
        ))
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8892b0")
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<p style='text-align:center; color:{color};'><b>{status}</b></p>",
                    unsafe_allow_html=True)

st.markdown("---")

# ─── Health Score Trends ───
st.markdown("### 📈 Health Score Trends Over Time")

# Daily average health scores
water_daily = water_health.set_index("timestamp")[[f"{s}_health_score" for s in water_sensors]].resample("1D").mean()
air_daily = air_health.set_index("timestamp")[[f"{s}_health_score" for s in air_sensors]].resample("1D").mean()

tab1, tab2 = st.tabs(["Water Quality", "Air Quality"])

with tab1:
    fig = go.Figure()
    colors = ["#64ffda", "#48dbfb", "#a29bfe", "#fd79a8", "#ffeaa7"]
    for idx, sensor in enumerate(water_sensors):
        col = f"{sensor}_health_score"
        fig.add_trace(go.Scatter(
            x=water_daily.index, y=water_daily[col],
            mode="lines", name=sensor.replace("_", " ").title(),
            line=dict(width=2, color=colors[idx])
        ))
    
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,217,61,0.5)",
                  annotation_text="Warning Threshold")
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,107,107,0.5)",
                  annotation_text="Critical Threshold")
    
    fig.update_layout(
        height=400, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)",
        font=dict(color="#8892b0"),
        yaxis=dict(range=[0, 100], title="Health Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = go.Figure()
    colors_air = ["#ff6b6b", "#ffd93d", "#48dbfb", "#a29bfe"]
    for idx, sensor in enumerate(air_sensors):
        col = f"{sensor}_health_score"
        fig.add_trace(go.Scatter(
            x=air_daily.index, y=air_daily[col],
            mode="lines", name=sensor.replace("_", " ").upper(),
            line=dict(width=2, color=colors_air[idx])
        ))
    
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,217,61,0.5)",
                  annotation_text="Warning Threshold")
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,107,107,0.5)",
                  annotation_text="Critical Threshold")
    
    fig.update_layout(
        height=400, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)",
        font=dict(color="#8892b0"),
        yaxis=dict(range=[0, 100], title="Health Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ─── Cost Analysis ───
st.markdown("### 💰 Scenario-Based Maintenance Cost Model (Hypothetical Simulation)")
st.caption(
    "⚠️ **Simulation Disclaimer**: The figures below represent parameterized scenario assumptions "
    "for strategic comparison of maintenance policies. They do not represent actual proprietary accounting figures of Anatec Yanako."
)

n_sensors = len(water_sensors) + len(air_sensors)
n_anomalies = len(anomaly_log)

# Cost parameters (user adjustable)
with st.expander("Adjust Scenario Cost Parameters (JPY)"):
    c1, c2, c3 = st.columns(3)
    with c1:
        cost_calibration = st.number_input("Calibration Cost (JPY)", value=50000, step=5000)
        cost_inspection = st.number_input("Inspection Cost (JPY)", value=30000, step=5000)
    with c2:
        cost_repair = st.number_input("Reactive Repair (JPY)", value=200000, step=10000)
        cost_replacement = st.number_input("Emergency Replacement (JPY)", value=500000, step=50000)
    with c3:
        cost_downtime = st.number_input("Downtime Cost/hour (JPY)", value=100000, step=10000)
        cost_fine = st.number_input("Regulatory Fine Risk (JPY)", value=1000000, step=100000)

# Calculate costs
reactive = (n_anomalies * 0.3 * cost_repair + n_anomalies * 0.05 * cost_replacement +
            n_anomalies * 0.1 * cost_downtime * 4 + n_anomalies * 0.02 * cost_fine)
scheduled = (n_sensors * 2 * cost_calibration + n_sensors * 4 * cost_inspection +
             n_anomalies * 0.1 * cost_repair)
cbm_cost = (n_sensors * 1 * cost_calibration + n_sensors * 2 * cost_inspection +
            n_anomalies * 0.03 * cost_repair)

strategies = ["Reactive (Break-Fix)", "Calendar Scheduled", "Condition-Based (EcoSense)"]
costs = [reactive, scheduled, cbm_cost]
colors = ["#ff6b6b", "#ffd93d", "#64ffda"]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=strategies, y=costs,
    marker_color=colors,
    text=[f"JPY {c:,.0f}" for c in costs],
    textposition="outside",
    textfont=dict(color="#ccd6f6", size=14)
))

fig.update_layout(
    height=400, template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)",
    font=dict(color="#8892b0"),
    yaxis_title="Estimated 6-Month Cost (JPY)",
    title="Simulated 6-Month Maintenance Cost Comparison"
)
st.plotly_chart(fig, use_container_width=True)

# Savings summary
savings = reactive - cbm_cost
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Reactive Baseline", f"JPY {reactive:,.0f}")
with col2:
    st.metric("Condition-Based Policy", f"JPY {cbm_cost:,.0f}")
with col3:
    st.metric("Scenario Savings", f"JPY {savings:,.0f}",
              delta=f"{savings/reactive*100:.0f}% reduction",
              delta_color="normal")

st.markdown("---")

# ─── Maintenance Schedule ───
st.markdown("### 📅 Condition-Based Maintenance Action Schedule")

schedule_data = []
all_sensors = [(s, "Water Quality", sensor_config["water_quality"][s]) for s in water_sensors] + \
              [(s, "Air Quality", sensor_config["air_quality"][s]) for s in air_sensors]

for sensor, category, config in all_sensors:
    if category == "Water Quality":
        health = get_current_health(water_health, sensor)
    else:
        health = get_current_health(air_health, sensor)
    
    color, status = health_color(health)
    
    if health < 50:
        action = "Immediate Inspection & Dispatch (within 48 hours)"
        urgency = "CRITICAL"
    elif health < 70:
        action = "Schedule Recalibration & Cleaning (within 14 days)"
        urgency = "WARNING"
    elif health < 85:
        action = "Standard Telemetry Monitoring (60-day review)"
        urgency = "NORMAL"
    else:
        action = "Routine Preventative Check (90-day cycle)"
        urgency = "OPTIMAL"
    
    schedule_data.append({
        "Sensor": sensor,
        "Category": category,
        "Health Score": f"{health:.0f}/100",
        "Condition": status,
        "Urgency Level": urgency,
        "Recommended Action Window": action
    })

schedule_df = pd.DataFrame(schedule_data)

# Color-code the dataframe
def highlight_urgency(val):
    if val == "CRITICAL":
        return "background-color: rgba(255,107,107,0.3); color: #ff6b6b;"
    elif val == "WARNING":
        return "background-color: rgba(253,121,168,0.2); color: #fd79a8;"
    elif val == "NORMAL":
        return "background-color: rgba(255,217,61,0.2); color: #ffd93d;"
    else:
        return "background-color: rgba(100,255,218,0.1); color: #64ffda;"

st.dataframe(
    schedule_df.style.map(highlight_urgency, subset=["Urgency Level"]),
    use_container_width=True,
    hide_index=True
)
