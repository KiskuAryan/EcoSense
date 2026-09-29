# EcoSense — Power BI Plant Operations Dashboard 📊

This directory contains the Power BI reporting assets for the **EcoSense** environmental measurement equipment monitoring system (designed for continuous wastewater and flue gas industrial telemetry).

---

## 📁 Files in This Directory

- `EcoSense.pbix` — The interactive Power BI Desktop report file.
- `README.md` — Documentation of the 5 report pages, data model, and DAX measures.

---

## 📑 Report Structure (5 Interactive Pages)

1. **Executive Overview** (`Page 1`):
   - **4 Question-Driven KPIs**: Actively monitored sensors (9), total operational anomaly points (676), fleet health index (91.5/100), and units in warning condition (3).
   - **Telemetry Visuals**: Wastewater pH telemetry curve, flue gas CO₂ emission concentration, monthly incident frequency, and process medium defect breakdown (Wastewater vs. Flue Gas).

2. **Operational Anomaly Audit** (`Page 2`):
   - **3 Diagnostic KPIs**: Total operational detections (676), affected equipment units (7 of 9), mean incident severity (0.33).
   - **Chronological Incident Spikes**: Daily time-series showing major defect clusters (February peak at 68 detections).
   - **Root Cause & Severity Breakdown**: Defect count per sensor (Conductivity and pH driving 64% of alerts), severity histogram, and chronological monthly volume.

3. **Sensor Deep-Dive & Temporal Trend Explorer** (`Page 3`):
   - **Single-Sensor Inspection**: Sensor selector and custom date window slicer.
   - **3 Statistical Baselines**: Minimum recorded reading, typical operating mean, and peak surge value.
   - **Diurnal Operating Cycle**: 24-hour diurnal profile (00:00–23:00) proving correlation with industrial plant production shifts.
   - **Longitudinal Telemetry**: 6-month continuous waveform tracking drift and dropouts.

4. **Fleet Condition & Sensor Health Index** (`Page 4`):
   - **Health Scoring**: Standardized 4-tier health framework:
     - 🟢 **Optimal**: 85–100
     - 🔵 **Normal**: 70–84
     - 🟡 **Warning**: 50–69 (requires proactive calibration)
     - 🔴 **Critical**: < 50 (immediate replacement)
   - **Fleet Distribution Donut**: 66.7% Healthy (6 units) vs. 33.3% Warning (3 units).
   - **Health Score Ranking Bar & Degradation Table**: Individual sensor ranking with weekly slope rates (identifying PM2.5 at -0.20/week drift).

5. **Condition-Based Maintenance & Action Register** (`Page 5`):
   - **Action & Dispatch Register**: Formal dispatch table listing sensors, health tiers, anomaly load, weekly degradation rates, and assigned actions (`Calibration`).
   - **4 Grounded KPIs**: Sensors requiring review (3), critical stoppages (0), total defect load to audit (676), units needing immediate calibration (1).
   - **Equipment Defect Load Chart**: Relative defect volume comparison.

---

## 🔄 Data Connection & Refresh

All visuals in this report connect directly to the CSV files in `../data/`:
- `../data/raw/water_quality.csv`
- `../data/raw/air_quality.csv`
- `../data/anomalies/detected_anomalies.csv`
- `../data/processed/sensor_trends.csv`
- `../data/processed/maintenance_schedule.csv`

To refresh the report with newly generated pipeline telemetry:
1. Open `EcoSense.pbix` in **Power BI Desktop**.
2. Click **Home ➔ Refresh** in the top ribbon.
