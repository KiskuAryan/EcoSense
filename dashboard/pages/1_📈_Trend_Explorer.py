"""
EcoSense Dashboard — Trend Explorer Page
Interactive time-series exploration with zoom, pan, and anomaly highlighting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json

st.set_page_config(page_title="Trend Explorer | EcoSense", page_icon="📈", layout="wide")

# ─── Load data ───
@st.cache_data
def load_data():
    base = Path(__file__).parent.parent.parent / "data"
    water = pd.read_csv(base / "raw" / "water_quality.csv", parse_dates=["timestamp"])
    air = pd.read_csv(base / "raw" / "air_quality.csv", parse_dates=["timestamp"])
    det_path = base / "anomalies" / "detected_anomalies.csv"
    detected = pd.read_csv(det_path, parse_dates=["timestamp"]) if det_path.exists() else pd.DataFrame()
    with open(base / "raw" / "sensor_config.json") as f:
        config = json.load(f)
    return water, air, detected, config

water_df, air_df, detected_df, sensor_config = load_data()

water_sensors = ["pH", "turbidity_ntu", "dissolved_oxygen_mg_l", "temperature_c", "conductivity_us_cm"]
air_sensors = ["co2_ppm", "nox_ppm", "so2_ppm", "pm25_ug_m3"]

# ─── Sidebar Controls ───
with st.sidebar:
    st.markdown("## 📈 Trend Explorer")
    st.markdown("---")
    
    category = st.radio("Sensor Category", ["Water Quality", "Air Quality"])
    
    if category == "Water Quality":
        available = water_sensors
        df = water_df
        configs = sensor_config["water_quality"]
    else:
        available = air_sensors
        df = air_df
        configs = sensor_config["air_quality"]
    
    selected_sensors = st.multiselect(
        "Select Sensors", available, default=available[:2]
    )
    
    st.markdown("---")
    
    # Date range
    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()
    date_range = st.date_input("Date Range", value=(min_date, max_date),
                                min_value=min_date, max_value=max_date)
    
    st.markdown("---")
    
    anomaly_overlay = st.selectbox(
        "Anomaly Highlights",
        [
            "Model Detections (Operational Ensemble)",
            "Ground Truth (Injected Defects)",
            "Both (Compare Model vs Benchmark)",
            "None"
        ],
        index=0
    )
    show_normal_range = st.checkbox("Show Normal Operating Range", value=True)
    show_rolling_avg = st.checkbox("Show Rolling Average (Trend)", value=False)
    rolling_window = st.slider("Rolling Window (hours)", 1, 72, 24) if show_rolling_avg else 24
    
    resample_freq = st.selectbox("Resample Frequency", 
                                  ["Raw (15min)", "Hourly", "4-Hourly", "Daily"],
                                  index=0)


# ─── Filter data ───
if len(date_range) == 2:
    mask = (df["timestamp"].dt.date >= date_range[0]) & (df["timestamp"].dt.date <= date_range[1])
    df_filtered = df[mask].copy()
else:
    df_filtered = df.copy()

# Resample if needed
resample_map = {"Raw (15min)": None, "Hourly": "1h", "4-Hourly": "4h", "Daily": "1D"}
freq = resample_map[resample_freq]

if freq and selected_sensors:
    df_plot = df_filtered.set_index("timestamp")[selected_sensors].resample(freq).mean().reset_index()
else:
    df_plot = df_filtered

# ─── Main Chart ───
st.markdown("# 📈 Trend Explorer")
st.markdown(f"*{category} sensors — {len(df_filtered):,} data points*")

if not selected_sensors:
    st.warning("Please select at least one sensor from the sidebar.")
    st.stop()

colors = ["#64ffda", "#48dbfb", "#a29bfe", "#fd79a8", "#ffeaa7",
          "#55efc4", "#81ecec", "#74b9ff", "#dfe6e9"]

# Individual subplots for each sensor
fig = make_subplots(
    rows=len(selected_sensors), cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=[f"{s} ({configs[s]['unit']})" for s in selected_sensors]
)

for idx, sensor in enumerate(selected_sensors):
    row = idx + 1
    color = colors[idx % len(colors)]
    config = configs[sensor]
    
    # Main trace
    fig.add_trace(
        go.Scattergl(
            x=df_plot["timestamp"], y=df_plot[sensor],
            mode="lines", name=sensor,
            line=dict(width=1, color=color),
            opacity=0.9
        ), row=row, col=1
    )
    
    # Normal range
    if show_normal_range:
        fig.add_hline(y=config["normal_min"], line_dash="dash", line_color="rgba(100,255,218,0.3)",
                      row=row, col=1)
        fig.add_hline(y=config["normal_max"], line_dash="dash", line_color="rgba(100,255,218,0.3)",
                      row=row, col=1)
    
    # Anomaly highlights
    if anomaly_overlay in ["Ground Truth (Injected Defects)", "Both (Compare Model vs Benchmark)"]:
        anom_col = f"is_anomaly_{sensor}"
        if anom_col in df_filtered.columns:
            gt_mask = df_filtered[anom_col] == 1
            if gt_mask.any():
                fig.add_trace(
                    go.Scattergl(
                        x=df_filtered.loc[gt_mask, "timestamp"],
                        y=df_filtered.loc[gt_mask, sensor],
                        mode="markers", name="Ground Truth (Injected)",
                        marker=dict(size=7, color="#ff6b6b", opacity=0.75, symbol="circle-open", line=dict(width=2, color="#ff6b6b")),
                        showlegend=(idx == 0)
                    ), row=row, col=1
                )
    
    if anomaly_overlay in ["Model Detections (Operational Ensemble)", "Both (Compare Model vs Benchmark)"]:
        if not detected_df.empty:
            det_sub = detected_df[
                (detected_df["sensor"] == sensor) & 
                (detected_df["timestamp"] >= df_filtered["timestamp"].min()) & 
                (detected_df["timestamp"] <= df_filtered["timestamp"].max())
            ]
            if not det_sub.empty:
                fig.add_trace(
                    go.Scattergl(
                        x=det_sub["timestamp"],
                        y=det_sub["value"],
                        mode="markers", name="Model Detection (2-of-3)",
                        marker=dict(size=6, color="#ffd93d", opacity=0.85, symbol="diamond"),
                        showlegend=(idx == 0)
                    ), row=row, col=1
                )
    
    # Rolling average
    if show_rolling_avg:
        window_pts = rolling_window * 4  # 4 points per hour
        rolling_avg = df_plot[sensor].rolling(window=window_pts, min_periods=1).mean()
        fig.add_trace(
            go.Scattergl(
                x=df_plot["timestamp"], y=rolling_avg,
                mode="lines", name=f"{sensor} MA({rolling_window}h)",
                line=dict(width=2, color="#ffd93d", dash="dot"),
                opacity=0.7
            ), row=row, col=1
        )

fig.update_layout(
    height=300 * len(selected_sensors),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.1)",
    font=dict(color="#8892b0"),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=60, r=20, t=80, b=40)
)

fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")

st.plotly_chart(fig, use_container_width=True)


# ─── Statistics Table ───
st.markdown("### 📊 Statistics for Selected Period")

stats_data = []
for sensor in selected_sensors:
    config = configs[sensor]
    col = df_filtered[sensor].dropna()
    
    out_of_range = ((col < config["normal_min"]) | (col > config["normal_max"])).sum()
    
    stats_data.append({
        "Sensor": sensor,
        "Mean": f"{col.mean():.4f}",
        "Std Dev": f"{col.std():.4f}",
        "Min": f"{col.min():.4f}",
        "Max": f"{col.max():.4f}",
        "Missing": col.isna().sum() if hasattr(col, 'isna') else 0,
        "Out of Range": out_of_range,
        "Out of Range %": f"{out_of_range / len(col) * 100:.2f}%"
    })

st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)


# ─── Correlation between selected sensors ───
if len(selected_sensors) > 1:
    st.markdown("### 🔗 Correlation Matrix")
    
    import plotly.express as px
    
    corr_data = df_filtered[selected_sensors].corr()
    
    fig_corr = px.imshow(
        corr_data,
        text_auto=".3f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        labels=dict(color="Correlation")
    )
    fig_corr.update_layout(
        height=400,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8892b0")
    )
    st.plotly_chart(fig_corr, use_container_width=True)
