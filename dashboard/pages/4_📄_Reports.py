"""
EcoSense Dashboard — Reports Page
Generate and export PDF/Excel reports.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import json
import sys

st.set_page_config(page_title="Reports | EcoSense", page_icon="📄", layout="wide")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ─── Load data ───
@st.cache_data
def load_data():
    base = Path(__file__).parent.parent.parent / "data"
    water = pd.read_csv(base / "raw" / "water_quality.csv", parse_dates=["timestamp"])
    air = pd.read_csv(base / "raw" / "air_quality.csv", parse_dates=["timestamp"])
    gt = pd.read_csv(base / "anomalies" / "injected_anomalies.csv",
                     parse_dates=["start_time", "end_time"])
    det_path = base / "anomalies" / "detected_anomalies.csv"
    det = pd.read_csv(det_path, parse_dates=["timestamp"]) if det_path.exists() else pd.DataFrame()
    return water, air, gt, det

water_df, air_df, ground_truth_log, detected_log = load_data()

water_sensors = ["pH", "turbidity_ntu", "dissolved_oxygen_mg_l", "temperature_c", "conductivity_us_cm"]
air_sensors = ["co2_ppm", "nox_ppm", "so2_ppm", "pm25_ug_m3"]

# ─── Page Header ───
st.markdown("# 📄 Report Generation")
st.markdown("*Export professional reports for presentations and documentation*")
st.markdown("---")

# ─── Report Type Selection ───
st.markdown("### Select Report Type")

report_type = st.selectbox(
    "Choose report:",
    [
        "Data Visualization Report (PDF)",
        "Anomaly Analysis Report (PDF)",
        "Maintenance Improvement Proposal (PDF)",
        "Complete Dataset (Excel)",
        "Anomaly Log (Excel)",
        "Summary Statistics (Excel)"
    ]
)

st.markdown("---")

# ─── Author Info ───
author = st.text_input("Author Name", value="EcoSense Intern")

# ─── Generate Report ───
if st.button("Generate Report", type="primary", use_container_width=True):
    reports_dir = Path(__file__).parent.parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    with st.spinner("Generating report..."):
        try:
            if "PDF" in report_type:
                from src.report_generator import (
                    generate_data_visualization_report,
                    generate_anomaly_report,
                    generate_maintenance_proposal_report
                )
                
                if "Data Visualization" in report_type:
                    path = generate_data_visualization_report(
                        water_df, air_df,
                        str(reports_dir / "data_visualization_report.pdf"),
                        author=author
                    )
                    st.success(f"Report generated: {path}")
                    
                    with open(path, "rb") as f:
                        st.download_button(
                            "Download Data Visualization Report (PDF)",
                            f.read(),
                            file_name="data_visualization_report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                
                elif "Anomaly" in report_type:
                    path = generate_anomaly_report(
                        detected_log if not detected_log.empty else ground_truth_log,
                        ground_truth_log=ground_truth_log,
                        output_path=str(reports_dir / "anomaly_analysis_report.pdf"),
                        author=author
                    )
                    st.success(f"Report generated: {path}")
                    
                    with open(path, "rb") as f:
                        st.download_button(
                            "Download Anomaly Analysis Report (PDF)",
                            f.read(),
                            file_name="anomaly_analysis_report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

                elif "Maintenance Improvement" in report_type:
                    path = generate_maintenance_proposal_report(
                        output_path=str(reports_dir / "maintenance_proposal.pdf"),
                        author=author
                    )
                    st.success(f"Report generated: {path}")
                    
                    with open(path, "rb") as f:
                        st.download_button(
                            "Download Maintenance Improvement Proposal (PDF)",
                            f.read(),
                            file_name="maintenance_proposal.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
            
            elif "Complete Dataset" in report_type:
                from src.report_generator import generate_excel_report
                
                path = generate_excel_report(
                    water_df, air_df, detected_log,
                    str(reports_dir / "ecosense_complete_data.xlsx")
                )
                st.success(f"Report generated: {path}")
                
                with open(path, "rb") as f:
                    st.download_button(
                        "Download Complete Dataset (Excel)",
                        f.read(),
                        file_name="ecosense_complete_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            elif "Anomaly Log" in report_type:
                path = str(reports_dir / "anomaly_log.xlsx")
                with pd.ExcelWriter(path, engine="openpyxl") as writer:
                    if not detected_log.empty:
                        detected_log.to_excel(writer, sheet_name="Operational Detections", index=False)
                    if not ground_truth_log.empty:
                        ground_truth_log.to_excel(writer, sheet_name="Ground Truth Benchmark", index=False)
                st.success(f"Report generated: {path}")
                
                with open(path, "rb") as f:
                    st.download_button(
                        "Download Anomaly Log (Excel)",
                        f.read(),
                        file_name="anomaly_log.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            elif "Summary Statistics" in report_type:
                path = str(reports_dir / "summary_statistics.xlsx")
                
                with pd.ExcelWriter(path, engine="openpyxl") as writer:
                    water_df[water_sensors].describe().to_excel(writer, sheet_name="Water Quality Stats")
                    air_df[air_sensors].describe().to_excel(writer, sheet_name="Air Quality Stats")
                    
                    # Anomaly summary
                    if not ground_truth_log.empty:
                        anom_summary = ground_truth_log.groupby(["sensor", "anomaly_type"]).agg(
                            count=("duration_hours", "count"),
                            avg_duration_h=("duration_hours", "mean"),
                            total_duration_h=("duration_hours", "sum")
                        ).round(2)
                        anom_summary.to_excel(writer, sheet_name="Anomaly Summary")
                
                st.success(f"Report generated: {path}")
                
                with open(path, "rb") as f:
                    st.download_button(
                        "Download Summary Statistics (Excel)",
                        f.read(),
                        file_name="summary_statistics.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            st.exception(e)

st.markdown("---")

# ─── Report Descriptions ───
st.markdown("### 📑 Report Descriptions")

reports_info = {
    "Data Visualization Report (PDF)": {
        "description": "Comprehensive overview of sensor data including dataset statistics, "
                      "value distributions, data quality assessment, and key metrics.",
        "contents": ["Dataset overview", "Water quality summary table", 
                     "Air quality summary table", "Data completeness analysis"],
        "icon": "📊"
    },
    "Anomaly Analysis Report (PDF)": {
        "description": "Detailed analysis of detected anomalies including type distribution, "
                      "severity assessment, and recommended actions.",
        "contents": ["Anomaly summary", "Type distribution", 
                     "Method comparison", "Recommendations"],
        "icon": "🔴"
    },
    "Complete Dataset (Excel)": {
        "description": "Full sensor data export with all measurements, anomaly labels, "
                      "and metadata in a multi-sheet Excel workbook.",
        "contents": ["Water quality data", "Air quality data", 
                     "Statistical summaries", "Anomaly log"],
        "icon": "📊"
    },
    "Anomaly Log (Excel)": {
        "description": "Detailed anomaly workbook separating operational model detections from the injected ground-truth benchmark., "
                      "types, durations, and affected sensors.",
        "contents": ["Event timestamps", "Anomaly types", 
                     "Duration analysis", "Sensor mapping"],
        "icon": "⚠️"
    },
    "Summary Statistics (Excel)": {
        "description": "Statistical summary of all sensor data including descriptive statistics "
                      "and anomaly frequency breakdown.",
        "contents": ["Descriptive statistics", "Water quality stats", 
                     "Air quality stats", "Anomaly frequency"],
        "icon": "📈"
    }
}

for name, info in reports_info.items():
    with st.expander(f"{info['icon']} {name}"):
        st.write(info["description"])
        st.markdown("**Contents:**")
        for item in info["contents"]:
            st.markdown(f"- {item}")
