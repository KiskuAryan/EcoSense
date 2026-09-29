"""
EcoSense - Synthetic Environmental Sensor Data Generator
=========================================================

Generates realistic time-series data for environmental monitoring sensors,
simulating what continuous measurement equipment produces in industrial factory settings.

Sensor Categories:
    - Water Quality: pH, Turbidity, Dissolved Oxygen, Temperature, Conductivity
    - Air Quality: CO2, NOx, SO2, PM2.5

Features:
    - Realistic daily and seasonal patterns
    - Factory shift-based variations (3 shifts: morning, afternoon, night)
    - Correlated sensor behavior (e.g., temperature affects dissolved oxygen)
    - Injected anomalies: drift, spikes, flatlines, cyclic irregularities
    - Configurable time range and sampling interval

Author: EcoSense Project
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import os


# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────

SENSOR_CONFIG = {
    "water_quality": {
        "pH": {
            "base": 7.2,
            "noise_std": 0.15,
            "daily_amplitude": 0.3,
            "seasonal_amplitude": 0.2,
            "unit": "",
            "normal_min": 6.5,
            "normal_max": 8.5,
            "absolute_min": 0.0,
            "absolute_max": 14.0,
            "description": "Measure of acidity/alkalinity of water"
        },
        "turbidity_ntu": {
            "base": 2.5,
            "noise_std": 0.4,
            "daily_amplitude": 0.8,
            "seasonal_amplitude": 0.5,
            "unit": "NTU",
            "normal_min": 0.0,
            "normal_max": 5.0,
            "absolute_min": 0.0,
            "absolute_max": 1000.0,
            "description": "Clarity of water, measured in Nephelometric Turbidity Units"
        },
        "dissolved_oxygen_mg_l": {
            "base": 7.5,
            "noise_std": 0.3,
            "daily_amplitude": 0.5,
            "seasonal_amplitude": 1.0,
            "unit": "mg/L",
            "normal_min": 6.0,
            "normal_max": 9.0,
            "absolute_min": 0.0,
            "absolute_max": 20.0,
            "description": "Amount of oxygen dissolved in water"
        },
        "temperature_c": {
            "base": 22.0,
            "noise_std": 0.5,
            "daily_amplitude": 2.0,
            "seasonal_amplitude": 6.0,
            "unit": "°C",
            "normal_min": 15.0,
            "normal_max": 30.0,
            "absolute_min": -5.0,
            "absolute_max": 50.0,
            "description": "Water temperature"
        },
        "conductivity_us_cm": {
            "base": 450.0,
            "noise_std": 20.0,
            "daily_amplitude": 30.0,
            "seasonal_amplitude": 50.0,
            "unit": "µS/cm",
            "normal_min": 200.0,
            "normal_max": 800.0,
            "absolute_min": 0.0,
            "absolute_max": 2000.0,
            "description": "Ability of water to conduct electrical current"
        }
    },
    "air_quality": {
        "co2_ppm": {
            "base": 420.0,
            "noise_std": 15.0,
            "daily_amplitude": 40.0,
            "seasonal_amplitude": 20.0,
            "unit": "ppm",
            "normal_min": 350.0,
            "normal_max": 500.0,
            "absolute_min": 200.0,
            "absolute_max": 5000.0,
            "description": "Carbon dioxide concentration"
        },
        "nox_ppm": {
            "base": 0.045,
            "noise_std": 0.008,
            "daily_amplitude": 0.02,
            "seasonal_amplitude": 0.01,
            "unit": "ppm",
            "normal_min": 0.0,
            "normal_max": 0.1,
            "absolute_min": 0.0,
            "absolute_max": 1.0,
            "description": "Nitrogen oxides concentration"
        },
        "so2_ppm": {
            "base": 0.02,
            "noise_std": 0.005,
            "daily_amplitude": 0.01,
            "seasonal_amplitude": 0.008,
            "unit": "ppm",
            "normal_min": 0.0,
            "normal_max": 0.05,
            "absolute_min": 0.0,
            "absolute_max": 0.5,
            "description": "Sulfur dioxide concentration"
        },
        "pm25_ug_m3": {
            "base": 18.0,
            "noise_std": 4.0,
            "daily_amplitude": 8.0,
            "seasonal_amplitude": 6.0,
            "unit": "µg/m³",
            "normal_min": 0.0,
            "normal_max": 35.0,
            "absolute_min": 0.0,
            "absolute_max": 500.0,
            "description": "Fine particulate matter concentration"
        }
    }
}

# Anomaly definitions
ANOMALY_TYPES = {
    "gradual_drift": {
        "description": "Sensor calibration drift over time",
        "duration_hours": (48, 168),  # 2-7 days
        "probability": 0.08
    },
    "sudden_spike": {
        "description": "Equipment malfunction or process upset",
        "duration_hours": (1, 6),
        "probability": 0.12
    },
    "flatline": {
        "description": "Sensor failure or communication loss",
        "duration_hours": (2, 24),
        "probability": 0.05
    },
    "cyclic_irregular": {
        "description": "Process control oscillation",
        "duration_hours": (12, 72),
        "probability": 0.06
    }
}


# ─────────────────────────────────────────────────────────
# Data Generation Functions
# ─────────────────────────────────────────────────────────

def generate_timestamps(start_date: str, end_date: str, interval_minutes: int = 15) -> pd.DatetimeIndex:
    """
    Generate a DatetimeIndex from start to end at the given interval.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval_minutes: Sampling interval in minutes
    
    Returns:
        pd.DatetimeIndex of timestamps
    """
    return pd.date_range(start=start_date, end=end_date, freq=f"{interval_minutes}min")


def generate_base_signal(timestamps: pd.DatetimeIndex, config: dict, seed: int = None) -> np.ndarray:
    """
    Generate a realistic base signal with daily cycles, seasonal patterns,
    factory shift variations, and random noise.
    
    Args:
        timestamps: DatetimeIndex for the signal
        config: Sensor configuration dictionary
        seed: Random seed for reproducibility
    
    Returns:
        numpy array of sensor values
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    n = len(timestamps)
    hours = np.array([(t.hour + t.minute / 60) for t in timestamps])
    days_from_start = np.array([(t - timestamps[0]).total_seconds() / 86400 for t in timestamps])
    
    # --- Base value ---
    signal = np.full(n, config["base"], dtype=float)
    
    # --- Daily cycle (sinusoidal, peaks in early afternoon) ---
    daily_cycle = config["daily_amplitude"] * np.sin(2 * np.pi * (hours - 6) / 24)
    signal += daily_cycle
    
    # --- Seasonal cycle (sinusoidal, peaks in summer ~July) ---
    day_of_year = np.array([t.timetuple().tm_yday for t in timestamps])
    seasonal_cycle = config["seasonal_amplitude"] * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    signal += seasonal_cycle
    
    # --- Factory shift pattern ---
    # 3 shifts: Morning (6-14), Afternoon (14-22), Night (22-6)
    # Activity is higher during morning and afternoon shifts
    shift_factor = np.where(
        (hours >= 6) & (hours < 14), 1.0,      # Morning shift - full activity
        np.where(
            (hours >= 14) & (hours < 22), 0.85,  # Afternoon shift - slightly less
            0.4                                    # Night shift - reduced activity
        )
    )
    # Shift effect is a fraction of the daily amplitude
    shift_effect = config["daily_amplitude"] * 0.3 * (shift_factor - 0.75)
    signal += shift_effect
    
    # --- Weekly pattern (slightly lower on weekends) ---
    day_of_week = np.array([t.weekday() for t in timestamps])
    weekend_mask = (day_of_week >= 5)
    signal[weekend_mask] -= config["daily_amplitude"] * 0.15
    
    # --- Random noise ---
    noise = rng.normal(0, config["noise_std"], n)
    signal += noise
    
    # --- Slow random walk (long-term drift) ---
    random_walk = np.cumsum(rng.normal(0, config["noise_std"] * 0.01, n))
    # Normalize to prevent unbounded growth
    random_walk = random_walk - np.mean(random_walk)
    signal += random_walk
    
    # --- Clamp to absolute bounds ---
    signal = np.clip(signal, config["absolute_min"], config["absolute_max"])
    
    return signal


def inject_anomalies(signal: np.ndarray, timestamps: pd.DatetimeIndex, 
                     config: dict, sensor_name: str, seed: int = None) -> tuple:
    """
    Inject realistic anomalies into the signal.
    
    Args:
        signal: Base signal array
        timestamps: DatetimeIndex
        config: Sensor configuration
        sensor_name: Name of the sensor
        seed: Random seed
    
    Returns:
        Tuple of (modified_signal, anomaly_log DataFrame)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    n = len(signal)
    modified = signal.copy()
    anomaly_labels = np.zeros(n, dtype=int)  # 0 = normal, 1 = anomaly
    anomaly_log = []
    
    # Track used intervals to avoid overlap
    used_intervals = []
    
    for anomaly_type, anom_config in ANOMALY_TYPES.items():
        # Decide how many of this type to inject
        num_anomalies = rng.binomial(
            n=max(1, int(n / (4 * 24 * 7))),  # roughly per week
            p=anom_config["probability"]
        )
        
        for _ in range(num_anomalies):
            # Random duration within the configured range
            duration_hours = rng.uniform(*anom_config["duration_hours"])
            duration_points = int(duration_hours * 4)  # 4 points per hour (15-min intervals)
            
            if duration_points < 2:
                duration_points = 2
            
            # Random start point (avoid first and last 5%)
            margin = int(n * 0.05)
            start_idx = rng.randint(margin, n - margin - duration_points)
            end_idx = start_idx + duration_points
            
            # Check for overlap with existing anomalies
            overlaps = any(
                not (end_idx <= used_start or start_idx >= used_end)
                for used_start, used_end in used_intervals
            )
            if overlaps:
                continue
            
            used_intervals.append((start_idx, end_idx))
            anomaly_labels[start_idx:end_idx] = 1
            
            # Calculate anomaly magnitude based on sensor range
            sensor_range = config["normal_max"] - config["normal_min"]
            
            if anomaly_type == "gradual_drift":
                # Slowly increasing/decreasing offset
                direction = rng.choice([-1, 1])
                drift_magnitude = direction * sensor_range * rng.uniform(0.3, 0.7)
                drift = np.linspace(0, drift_magnitude, duration_points)
                modified[start_idx:end_idx] += drift
                
            elif anomaly_type == "sudden_spike":
                # Sharp spike up or down
                direction = rng.choice([-1, 1])
                spike_magnitude = direction * sensor_range * rng.uniform(0.5, 1.5)
                # Create spike shape (sharp rise, brief plateau, sharp fall)
                spike = np.zeros(duration_points)
                rise = min(4, duration_points // 3)
                fall = min(4, duration_points // 3)
                plateau = duration_points - rise - fall
                spike[:rise] = np.linspace(0, spike_magnitude, rise)
                spike[rise:rise + plateau] = spike_magnitude
                spike[rise + plateau:] = np.linspace(spike_magnitude, 0, fall)
                modified[start_idx:end_idx] += spike
                
            elif anomaly_type == "flatline":
                # Constant value (sensor stuck)
                stuck_value = modified[start_idx]
                modified[start_idx:end_idx] = stuck_value
                
            elif anomaly_type == "cyclic_irregular":
                # Oscillation with abnormal frequency
                t_anom = np.arange(duration_points)
                osc_amplitude = sensor_range * rng.uniform(0.2, 0.4)
                osc_frequency = rng.uniform(0.1, 0.5)  # cycles per point
                oscillation = osc_amplitude * np.sin(2 * np.pi * osc_frequency * t_anom)
                modified[start_idx:end_idx] += oscillation
            
            # Log the anomaly
            anomaly_log.append({
                "sensor": sensor_name,
                "anomaly_type": anomaly_type,
                "start_time": timestamps[start_idx],
                "end_time": timestamps[min(end_idx - 1, n - 1)],
                "duration_hours": round(duration_hours, 1),
                "start_index": start_idx,
                "end_index": end_idx,
                "description": anom_config["description"]
            })
    
    # Clamp modified signal to absolute bounds
    modified = np.clip(modified, config["absolute_min"], config["absolute_max"])
    
    anomaly_df = pd.DataFrame(anomaly_log) if anomaly_log else pd.DataFrame(
        columns=["sensor", "anomaly_type", "start_time", "end_time",
                 "duration_hours", "start_index", "end_index", "description"]
    )
    
    return modified, anomaly_labels, anomaly_df


def apply_correlations(water_data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply realistic inter-sensor correlations for water quality parameters.
    
    Physical relationships:
    - Higher temperature → lower dissolved oxygen (inverse correlation)
    - Higher turbidity → higher conductivity (positive correlation)
    - pH changes correlate with conductivity changes
    
    Args:
        water_data: DataFrame with water quality sensor columns
    
    Returns:
        Modified DataFrame with correlations applied
    """
    df = water_data.copy()
    
    if "temperature_c" in df.columns and "dissolved_oxygen_mg_l" in df.columns:
        # Dissolved oxygen decreases ~0.2 mg/L per 1°C increase
        temp_deviation = df["temperature_c"] - SENSOR_CONFIG["water_quality"]["temperature_c"]["base"]
        do_adjustment = -0.2 * temp_deviation
        df["dissolved_oxygen_mg_l"] += do_adjustment * 0.3  # Partial effect
        df["dissolved_oxygen_mg_l"] = df["dissolved_oxygen_mg_l"].clip(
            SENSOR_CONFIG["water_quality"]["dissolved_oxygen_mg_l"]["absolute_min"],
            SENSOR_CONFIG["water_quality"]["dissolved_oxygen_mg_l"]["absolute_max"]
        )
    
    if "turbidity_ntu" in df.columns and "conductivity_us_cm" in df.columns:
        # Higher turbidity → slightly higher conductivity
        turb_deviation = df["turbidity_ntu"] - SENSOR_CONFIG["water_quality"]["turbidity_ntu"]["base"]
        cond_adjustment = 5.0 * turb_deviation  # 5 µS/cm per NTU
        df["conductivity_us_cm"] += cond_adjustment * 0.2
        df["conductivity_us_cm"] = df["conductivity_us_cm"].clip(
            SENSOR_CONFIG["water_quality"]["conductivity_us_cm"]["absolute_min"],
            SENSOR_CONFIG["water_quality"]["conductivity_us_cm"]["absolute_max"]
        )
    
    return df


def add_missing_data(df: pd.DataFrame, missing_rate: float = 0.002, 
                     seed: int = None) -> pd.DataFrame:
    """
    Add realistic missing data patterns (random dropouts and brief outages).
    
    Args:
        df: Input DataFrame
        missing_rate: Proportion of individual values to make NaN
        seed: Random seed
    
    Returns:
        DataFrame with missing values introduced
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    result = df.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col.startswith("is_anomaly"):
            continue
        
        # Random single-point dropouts
        dropout_mask = rng.random(len(result)) < missing_rate
        result.loc[dropout_mask, col] = np.nan
        
        # Brief outages (1-4 consecutive points)
        n_outages = rng.randint(1, 4)
        for _ in range(n_outages):
            outage_start = rng.randint(0, len(result) - 20)
            outage_length = rng.randint(1, 5)
            result.loc[outage_start:outage_start + outage_length, col] = np.nan
    
    return result


# ─────────────────────────────────────────────────────────
# Main Generation Pipeline
# ─────────────────────────────────────────────────────────

def generate_dataset(
    start_date: str = "2026-01-01",
    end_date: str = "2026-06-30",
    interval_minutes: int = 15,
    master_seed: int = 42,
    output_dir: str = None
) -> dict:
    """
    Generate complete synthetic environmental sensor dataset.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval_minutes: Sampling interval in minutes
        master_seed: Master random seed for reproducibility
        output_dir: Output directory for CSV files
    
    Returns:
        Dictionary with 'water_quality', 'air_quality', and 'anomaly_log' DataFrames
    """
    print("=" * 60)
    print("EcoSense - Synthetic Data Generator")
    print("=" * 60)
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Interval: {interval_minutes} minutes")
    print(f"  Seed: {master_seed}")
    print()
    
    # Generate timestamps
    timestamps = generate_timestamps(start_date, end_date, interval_minutes)
    print(f"  Generated {len(timestamps):,} timestamps")
    
    all_anomalies = []
    seed_counter = master_seed
    
    # ─── Water Quality Sensors ───
    print("\n  Generating Water Quality data...")
    water_data = pd.DataFrame({"timestamp": timestamps})
    water_anomaly_labels = pd.DataFrame({"timestamp": timestamps})
    
    for sensor_name, config in SENSOR_CONFIG["water_quality"].items():
        seed_counter += 1
        
        # Generate base signal
        base_signal = generate_base_signal(timestamps, config, seed=seed_counter)
        
        # Inject anomalies
        seed_counter += 1
        anomalous_signal, labels, anomaly_df = inject_anomalies(
            base_signal, timestamps, config, sensor_name, seed=seed_counter
        )
        
        water_data[sensor_name] = anomalous_signal
        water_anomaly_labels[f"is_anomaly_{sensor_name}"] = labels
        
        n_anomalies = len(anomaly_df)
        print(f"    ✓ {sensor_name}: {n_anomalies} anomalies injected")
        
        if not anomaly_df.empty:
            all_anomalies.append(anomaly_df)
    
    # Apply inter-sensor correlations
    sensor_cols = [c for c in water_data.columns if c != "timestamp"]
    water_data[sensor_cols] = apply_correlations(water_data[sensor_cols])
    
    # Merge anomaly labels into water data
    for col in water_anomaly_labels.columns:
        if col != "timestamp":
            water_data[col] = water_anomaly_labels[col]
    
    # Add equipment metadata columns
    water_data["equipment_id"] = "WQ-SENSOR-001"
    water_data["location"] = "Factory_A_Wastewater_Outlet"
    
    # ─── Air Quality Sensors ───
    print("\n  Generating Air Quality data...")
    air_data = pd.DataFrame({"timestamp": timestamps})
    air_anomaly_labels = pd.DataFrame({"timestamp": timestamps})
    
    for sensor_name, config in SENSOR_CONFIG["air_quality"].items():
        seed_counter += 1
        
        base_signal = generate_base_signal(timestamps, config, seed=seed_counter)
        
        seed_counter += 1
        anomalous_signal, labels, anomaly_df = inject_anomalies(
            base_signal, timestamps, config, sensor_name, seed=seed_counter
        )
        
        air_data[sensor_name] = anomalous_signal
        air_anomaly_labels[f"is_anomaly_{sensor_name}"] = labels
        
        n_anomalies = len(anomaly_df)
        print(f"    ✓ {sensor_name}: {n_anomalies} anomalies injected")
        
        if not anomaly_df.empty:
            all_anomalies.append(anomaly_df)
    
    # Merge anomaly labels into air data
    for col in air_anomaly_labels.columns:
        if col != "timestamp":
            air_data[col] = air_anomaly_labels[col]
    
    # Add equipment metadata
    air_data["equipment_id"] = "AQ-SENSOR-001"
    air_data["location"] = "Factory_A_Exhaust_Stack"
    
    # ─── Add Missing Data ───
    print("\n  Adding realistic missing data patterns...")
    water_data = add_missing_data(water_data, missing_rate=0.002, seed=master_seed + 100)
    air_data = add_missing_data(air_data, missing_rate=0.002, seed=master_seed + 200)
    
    # ─── Combine Anomaly Logs ───
    if all_anomalies:
        anomaly_log = pd.concat(all_anomalies, ignore_index=True)
        anomaly_log = anomaly_log.sort_values("start_time").reset_index(drop=True)
    else:
        anomaly_log = pd.DataFrame()
    
    print(f"\n  Total anomalies injected: {len(anomaly_log)}")
    
    # ─── Save to CSV ───
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "raw"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    anomaly_dir = output_dir.parent / "anomalies"
    anomaly_dir.mkdir(parents=True, exist_ok=True)
    
    water_path = output_dir / "water_quality.csv"
    air_path = output_dir / "air_quality.csv"
    anomaly_path = anomaly_dir / "injected_anomalies.csv"
    config_path = output_dir / "sensor_config.json"
    
    water_data.to_csv(water_path, index=False)
    air_data.to_csv(air_path, index=False)
    anomaly_log.to_csv(anomaly_path, index=False)
    
    # Save sensor config as JSON for reference
    with open(config_path, "w") as f:
        json.dump(SENSOR_CONFIG, f, indent=2)
    
    print(f"\n  Files saved:")
    print(f"    → {water_path}")
    print(f"    → {air_path}")
    print(f"    → {anomaly_path}")
    print(f"    → {config_path}")
    
    # ─── Summary Statistics ───
    print("\n" + "=" * 60)
    print("  Dataset Summary")
    print("=" * 60)
    print(f"  Water Quality: {water_data.shape[0]:,} rows × {water_data.shape[1]} columns")
    print(f"  Air Quality:   {air_data.shape[0]:,} rows × {air_data.shape[1]} columns")
    print(f"  Anomalies:     {len(anomaly_log)} events across {anomaly_log['sensor'].nunique() if not anomaly_log.empty else 0} sensors")
    
    if not anomaly_log.empty:
        print(f"\n  Anomaly Breakdown:")
        for atype, count in anomaly_log["anomaly_type"].value_counts().items():
            print(f"    • {atype}: {count}")
    
    missing_water = water_data.select_dtypes(include=[np.number]).isnull().sum().sum()
    missing_air = air_data.select_dtypes(include=[np.number]).isnull().sum().sum()
    print(f"\n  Missing values: {missing_water + missing_air} total "
          f"(Water: {missing_water}, Air: {missing_air})")
    
    print("\n" + "=" * 60)
    print("  ✅ Data generation complete!")
    print("=" * 60)
    
    return {
        "water_quality": water_data,
        "air_quality": air_data,
        "anomaly_log": anomaly_log
    }


# ─────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="EcoSense Synthetic Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_generator.py
  python data_generator.py --start 2025-01-01 --end 2025-12-31
  python data_generator.py --interval 30 --seed 123
        """
    )
    parser.add_argument("--start", default="2026-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-06-30", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", type=int, default=15, help="Sampling interval (minutes)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=None, help="Output directory")
    
    args = parser.parse_args()
    
    generate_dataset(
        start_date=args.start,
        end_date=args.end,
        interval_minutes=args.interval,
        master_seed=args.seed,
        output_dir=args.output
    )
