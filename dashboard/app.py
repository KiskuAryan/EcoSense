"""
EcoSense — Interactive Dashboard
=================================

Multi-page Streamlit application for environmental sensor monitoring,
anomaly detection visualization, and predictive maintenance insights.

Launch with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoSense | Environmental Monitoring",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0c29 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Custom card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    .metric-card h3 {
        color: #a8b2d1;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-card .value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #64ffda 0%, #48dbfb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .status-normal { color: #64ffda; }
    .status-warning { color: #ffd93d; }
    .status-critical { color: #ff6b6b; }
    
    /* Header gradient text */
    .gradient-text {
        background: linear-gradient(135deg, #64ffda 0%, #48dbfb 50%, #a29bfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
    }
    
    .subtitle {
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(100,255,218,0.3), transparent);
        margin: 2rem 0;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Status badge */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-normal { background: rgba(100,255,218,0.15); color: #64ffda; }
    .badge-warning { background: rgba(255,217,61,0.15); color: #ffd93d; }
    .badge-critical { background: rgba(255,107,107,0.15); color: #ff6b6b; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Data Loading (cached)
# ─────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    """Load all data files distinguishing ground truth from model detections."""
    base = Path(__file__).parent.parent / "data"
    
    water = pd.read_csv(base / "raw" / "water_quality.csv", parse_dates=["timestamp"])
    air = pd.read_csv(base / "raw" / "air_quality.csv", parse_dates=["timestamp"])
    
    # Ground truth benchmark (injected defects)
    ground_truth = pd.read_csv(base / "anomalies" / "injected_anomalies.csv", 
                               parse_dates=["start_time", "end_time"])
    
    # Operational algorithm detections (model output)
    detected_path = base / "anomalies" / "detected_anomalies.csv"
    if detected_path.exists():
        detected = pd.read_csv(detected_path, parse_dates=["timestamp"])
    else:
        detected = pd.DataFrame()
    
    return water, air, ground_truth, detected


water_df, air_df, ground_truth_log, detected_log = load_data()


# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="gradient-text" style="font-size:1.5rem;">🏭 EcoSense</div>', 
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle" style="font-size:0.9rem;">Environmental Monitoring & Condition Analytics</p>',
                unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Date filter
    st.markdown("### 📅 Date Range")
    min_date = water_df["timestamp"].min().date()
    max_date = water_df["timestamp"].max().date()
    
    date_range = st.date_input(
        "Select range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Quick info
    st.markdown("### 📊 Dataset Info")
    st.markdown(f"**Records:** {len(water_df):,} per sensor")
    st.markdown(f"**Sensors:** 9 (5 Water + 4 Air)")
    st.markdown(f"**Operational Detections:** {len(detected_log):,} points")
    st.markdown(f"**Ground Truth Events:** {len(ground_truth_log)} episodes")
    st.markdown(f"**Period:** {min_date} to {max_date}")


# ─────────────────────────────────────────────────────────
# Main Page — Overview Dashboard
# ─────────────────────────────────────────────────────────

st.markdown('<div class="gradient-text">Environmental Monitoring Overview</div>', 
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Retrospective batch telemetry & sensor condition monitoring for Factory A</p>',
            unsafe_allow_html=True)

# Filter by date
if len(date_range) == 2:
    start, end = date_range
    mask_w = (water_df["timestamp"].dt.date >= start) & (water_df["timestamp"].dt.date <= end)
    mask_a = (air_df["timestamp"].dt.date >= start) & (air_df["timestamp"].dt.date <= end)
    water_filtered = water_df[mask_w]
    air_filtered = air_df[mask_a]
else:
    water_filtered = water_df
    air_filtered = air_df


# ─── KPI Metrics Row ───
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# Sensor status counts
water_sensors = ["pH", "turbidity_ntu", "dissolved_oxygen_mg_l", "temperature_c", "conductivity_us_cm"]
air_sensors = ["co2_ppm", "nox_ppm", "so2_ppm", "pm25_ug_m3"]

import json
config_path = Path(__file__).parent.parent / "data" / "raw" / "sensor_config.json"
with open(config_path) as f:
    sensor_config = json.load(f)

# Calculate latest status
def get_sensor_status(df, sensor, config):
    """Get status based on last 96 readings (24h)."""
    recent = df[sensor].tail(96).dropna()
    if len(recent) == 0:
        return "unknown"
    mean_val = recent.mean()
    nmin = config["normal_min"]
    nmax = config["normal_max"]
    margin = (nmax - nmin) * 0.15  # 15% warning margin
    
    if nmin + margin <= mean_val <= nmax - margin:
        return "normal"
    elif nmin <= mean_val <= nmax:
        return "warning"
    else:
        return "critical"

normal_count = 0
warning_count = 0
critical_count = 0

for sensor in water_sensors:
    status = get_sensor_status(water_filtered, sensor, sensor_config["water_quality"][sensor])
    if status == "normal": normal_count += 1
    elif status == "warning": warning_count += 1
    else: critical_count += 1

for sensor in air_sensors:
    status = get_sensor_status(air_filtered, sensor, sensor_config["air_quality"][sensor])
    if status == "normal": normal_count += 1
    elif status == "warning": warning_count += 1
    else: critical_count += 1

with kpi1:
    st.metric("Total Sensors", "9", help="5 Water Quality + 4 Air Quality sensors")

with kpi2:
    st.metric("Normal", f"{normal_count}", delta=f"{normal_count}/9 sensors", delta_color="normal")

with kpi3:
    st.metric("Warning", f"{warning_count}", 
              delta=f"{warning_count} need attention" if warning_count > 0 else "All clear",
              delta_color="off" if warning_count == 0 else "inverse")

with kpi4:
    st.metric(
        "Model Detections", 
        f"{len(detected_log):,}" if not detected_log.empty else "0",
        delta=f"Benchmark: {len(ground_truth_log)} injected events",
        help="Points flagged by operational 2-of-3 ensemble (Z-Score + IQR + Rolling Stats)"
    )


st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ─── Water Quality Current Readings ───
st.markdown("### 💧 Water Quality — Latest Readings")

wcols = st.columns(5)
for idx, sensor in enumerate(water_sensors):
    config = sensor_config["water_quality"][sensor]
    latest = water_filtered[sensor].dropna().iloc[-1] if not water_filtered[sensor].dropna().empty else 0
    avg = water_filtered[sensor].mean()
    delta = latest - avg
    
    status = get_sensor_status(water_filtered, sensor, config)
    status_emoji = "🟢" if status == "normal" else ("🟡" if status == "warning" else "🔴")
    
    with wcols[idx]:
        label = sensor.replace("_", " ").replace("ntu", "(NTU)").replace("mg l", "(mg/L)").replace("us cm", "(uS/cm)").replace("c", "(C)")
        st.metric(
            f"{status_emoji} {label}",
            f"{latest:.2f}",
            f"{delta:+.2f} vs avg",
            delta_color="normal" if abs(delta) < config["noise_std"] * 2 else "inverse"
        )
        st.progress(
            min(1.0, max(0.0, (latest - config["absolute_min"]) / (config["absolute_max"] - config["absolute_min"]))),
            text=f"Range: {config['normal_min']}–{config['normal_max']} {config['unit']}"
        )


# ─── Air Quality Current Readings ───
st.markdown("### 🌫️ Air Quality — Latest Readings")

acols = st.columns(4)
for idx, sensor in enumerate(air_sensors):
    config = sensor_config["air_quality"][sensor]
    latest = air_filtered[sensor].dropna().iloc[-1] if not air_filtered[sensor].dropna().empty else 0
    avg = air_filtered[sensor].mean()
    delta = latest - avg
    
    status = get_sensor_status(air_filtered, sensor, config)
    status_emoji = "🟢" if status == "normal" else ("🟡" if status == "warning" else "🔴")
    
    with acols[idx]:
        label = sensor.replace("_", " ").upper()
        st.metric(
            f"{status_emoji} {label}",
            f"{latest:.4f}" if latest < 1 else f"{latest:.1f}",
            f"{delta:+.4f} vs avg" if abs(delta) < 1 else f"{delta:+.1f} vs avg",
            delta_color="normal" if abs(delta) < config["noise_std"] * 2 else "inverse"
        )
        st.progress(
            min(1.0, max(0.0, (latest - config["absolute_min"]) / (config["absolute_max"] - config["absolute_min"]))),
            text=f"Limit: {config['normal_max']} {config['unit']}"
        )


st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ─── Recent Anomaly Events ───
st.markdown("### ⚠️ Recent Anomaly Activity")

tab_detected, tab_ground_truth = st.tabs([
    "🤖 Operational Model Detections", 
    "🎯 Ground Truth Benchmark (Injected Episodes)"
])

with tab_detected:
    if not detected_log.empty:
        st.markdown("*Latest individual timestamps flagged by operational 2-of-3 statistical ensemble:*")
        disp_det = detected_log.sort_values("timestamp", ascending=False).head(8)
        cols_show = ["timestamp", "sensor", "category", "value", "votes", "severity"]
        st.dataframe(disp_det[cols_show], use_container_width=True, hide_index=True)
    else:
        st.info("No operational anomaly points detected.")

with tab_ground_truth:
    if not ground_truth_log.empty:
        st.markdown("*Known simulated defect episodes injected into test data for benchmark evaluation:*")
        display_log = ground_truth_log.sort_values("start_time", ascending=False).head(6)
        
        for _, row in display_log.iterrows():
            type_emoji = {
                "sudden_spike": "⚡",
                "gradual_drift": "📈",
                "flatline": "📉",
                "cyclic_irregular": "🔄"
            }.get(row["anomaly_type"], "❓")
            
            severity_color = "🔴" if row["duration_hours"] > 24 else ("🟡" if row["duration_hours"] > 6 else "🟢")
            
            st.markdown(
                f"{type_emoji} **{row['anomaly_type'].replace('_', ' ').title()}** on "
                f"`{row['sensor']}` — {severity_color} {row['duration_hours']}h duration — "
                f"*{row['start_time'].strftime('%Y-%m-%d %H:%M')} to {row['end_time'].strftime('%Y-%m-%d %H:%M')}*"
            )
    else:
        st.info("No ground truth anomaly events recorded.")


st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ─── Quick Trend Sparklines ───
st.markdown("### 📈 24-Hour Trend Sparklines")

import plotly.graph_objects as go

# Get last 96 points (24 hours)
recent_water = water_filtered.tail(96)
recent_air = air_filtered.tail(96)

fig = go.Figure()

spark_cols = st.columns(3)

# Water sparklines
with spark_cols[0]:
    st.markdown("**💧 Water — pH**")
    chart_data = recent_water[["timestamp", "pH"]].set_index("timestamp")
    st.line_chart(chart_data, height=120, use_container_width=True)

with spark_cols[1]:
    st.markdown("**💧 Water — Temperature**")
    chart_data = recent_water[["timestamp", "temperature_c"]].set_index("timestamp")
    st.line_chart(chart_data, height=120, use_container_width=True)

with spark_cols[2]:
    st.markdown("**🌫️ Air — PM2.5**")
    chart_data = recent_air[["timestamp", "pm25_ug_m3"]].set_index("timestamp")
    st.line_chart(chart_data, height=120, use_container_width=True)


# Footer
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color:#4a5568; font-size:0.8rem;">'
    'EcoSense v1.0 | Environmental Sensor Anomaly Detection & Predictive Maintenance'
    '</p>',
    unsafe_allow_html=True
)
