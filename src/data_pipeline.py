"""
EcoSense Data Pipeline Script
Generates processed features and detected anomalies dataset.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd
from src.feature_engineering import engineer_features
from src.anomaly_detector import run_all_detectors, create_ensemble_prediction

def main():
    base = Path("data")
    with open(base / "raw" / "sensor_config.json") as f:
        config = json.load(f)

    water_df = pd.read_csv(base / "raw" / "water_quality.csv", parse_dates=["timestamp"])
    air_df = pd.read_csv(base / "raw" / "air_quality.csv", parse_dates=["timestamp"])

    water_sensors = ["pH", "turbidity_ntu", "dissolved_oxygen_mg_l", "temperature_c", "conductivity_us_cm"]
    air_sensors = ["co2_ppm", "nox_ppm", "so2_ppm", "pm25_ug_m3"]

    normal_ranges = {}
    for cat in ["water_quality", "air_quality"]:
        if cat in config:
            for sensor_name, scfg in config[cat].items():
                normal_ranges[sensor_name] = (scfg["normal_min"], scfg["normal_max"])

    # 1. Feature engineering
    print("Engineering water features...")
    water_feat = engineer_features(water_df, water_sensors, normal_ranges)
    print("Engineering air features...")
    air_feat = engineer_features(air_df, air_sensors, normal_ranges)

    # Combine features
    shared_cols = ["timestamp", "hour", "day_of_week", "is_weekend", "shift"]
    combined_features = pd.merge(water_feat, air_feat, on=shared_cols, suffixes=("_water", "_air"))
    processed_dir = base / "processed"
    processed_dir.mkdir(exist_ok=True)
    out_features = processed_dir / "sensor_features.csv"
    combined_features.to_csv(out_features, index=False)
    print(f"Saved sensor features to {out_features} with shape {combined_features.shape}")

    # 2. Run anomaly detection and log detected anomalies
    print("Running anomaly detection pipeline...")
    all_anomalies = []
    
    for sensor in water_sensors:
        series = water_df[sensor]
        results = run_all_detectors(series)
        ensemble = create_ensemble_prediction(results, min_votes=2)
        anom_idx = ensemble[ensemble == 1].index
        for idx in anom_idx:
            z_flag = int(results["Z-Score"].loc[idx, "is_anomaly"])
            iqr_flag = int(results["IQR"].loc[idx, "is_anomaly"])
            roll_flag = int(results["Rolling Stats"].loc[idx, "is_anomaly"])
            votes = z_flag + iqr_flag + roll_flag
            
            severities = [
                float(results["Z-Score"].loc[idx, "severity"]) if z_flag else 0.0,
                float(results["IQR"].loc[idx, "severity"]) if iqr_flag else 0.0,
                float(results["Rolling Stats"].loc[idx, "severity"]) if roll_flag else 0.0
            ]
            sev = max(severities) if max(severities) > 0 else 0.5
            
            all_anomalies.append({
                "timestamp": water_df.loc[idx, "timestamp"],
                "sensor": sensor,
                "category": "Water Quality",
                "value": round(float(series.loc[idx]), 4),
                "detection_method": "Operational Ensemble (2-of-3)",
                "votes": votes,
                "zscore_flag": z_flag,
                "iqr_flag": iqr_flag,
                "rolling_flag": roll_flag,
                "severity": round(sev, 4)
            })

    for sensor in air_sensors:
        series = air_df[sensor]
        results = run_all_detectors(series)
        ensemble = create_ensemble_prediction(results, min_votes=2)
        anom_idx = ensemble[ensemble == 1].index
        for idx in anom_idx:
            z_flag = int(results["Z-Score"].loc[idx, "is_anomaly"])
            iqr_flag = int(results["IQR"].loc[idx, "is_anomaly"])
            roll_flag = int(results["Rolling Stats"].loc[idx, "is_anomaly"])
            votes = z_flag + iqr_flag + roll_flag
            
            severities = [
                float(results["Z-Score"].loc[idx, "severity"]) if z_flag else 0.0,
                float(results["IQR"].loc[idx, "severity"]) if iqr_flag else 0.0,
                float(results["Rolling Stats"].loc[idx, "severity"]) if roll_flag else 0.0
            ]
            sev = max(severities) if max(severities) > 0 else 0.5
            
            all_anomalies.append({
                "timestamp": air_df.loc[idx, "timestamp"],
                "sensor": sensor,
                "category": "Air Quality",
                "value": round(float(series.loc[idx]), 4),
                "detection_method": "Operational Ensemble (2-of-3)",
                "votes": votes,
                "zscore_flag": z_flag,
                "iqr_flag": iqr_flag,
                "rolling_flag": roll_flag,
                "severity": round(sev, 4)
            })

    detected_df = pd.DataFrame(all_anomalies)
    out_anomalies = base / "anomalies" / "detected_anomalies.csv"
    detected_df.to_csv(out_anomalies, index=False)
    print(f"Saved {len(detected_df)} detected anomaly points to {out_anomalies}")

if __name__ == "__main__":
    main()
