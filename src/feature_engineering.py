"""
EcoSense - Feature Engineering Module
======================================

Transforms raw sensor time-series data into meaningful features
for anomaly detection and predictive maintenance analysis.

Feature Categories:
    - Statistical: rolling mean, std, min, max, range (1h, 4h, 24h windows)
    - Rate of change: first derivatives (deltas)
    - Temporal: hour of day, day of week, weekend indicator, shift classification
    - Boundary deviation: normal envelope distance, consecutive out-of-range dwell time, 24h stress
    - Health indicators: multi-component condition score (0-100)
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def add_rolling_features(df: pd.DataFrame, columns: list, 
                         windows: list = [4, 16, 96]) -> pd.DataFrame:
    """
    Add rolling statistical features for specified columns.
    
    Windows correspond to:
        - 4 points = 1 hour
        - 16 points = 4 hours
        - 96 points = 24 hours
    
    Args:
        df: Input DataFrame
        columns: List of column names to compute features for
        windows: List of rolling window sizes (in number of points)
    
    Returns:
        DataFrame with added rolling features
    """
    result = df.copy()
    
    for col in columns:
        if col not in result.columns:
            continue
            
        for w in windows:
            prefix = f"{col}_roll{w}"
            
            # Rolling mean
            result[f"{prefix}_mean"] = result[col].rolling(window=w, min_periods=1).mean()
            
            # Rolling standard deviation
            result[f"{prefix}_std"] = result[col].rolling(window=w, min_periods=1).std()
            
            # Rolling min and max
            result[f"{prefix}_min"] = result[col].rolling(window=w, min_periods=1).min()
            result[f"{prefix}_max"] = result[col].rolling(window=w, min_periods=1).max()
            
            # Rolling range (max - min)
            result[f"{prefix}_range"] = result[f"{prefix}_max"] - result[f"{prefix}_min"]
    
    return result


def add_rate_of_change(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Add rate of change (first derivative) and acceleration (second derivative).
    
    Args:
        df: Input DataFrame
        columns: List of column names
    
    Returns:
        DataFrame with rate of change features
    """
    result = df.copy()
    
    for col in columns:
        if col not in result.columns:
            continue
        
        # First derivative (rate of change between consecutive points)
        result[f"{col}_rate_of_change"] = result[col].diff()
        
        # Second derivative (acceleration)
        result[f"{col}_acceleration"] = result[f"{col}_rate_of_change"].diff()
        
        # Absolute rate of change (magnitude regardless of direction)
        result[f"{col}_abs_rate"] = result[f"{col}_rate_of_change"].abs()
    
    return result


def add_temporal_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Add time-based features useful for pattern recognition.
    
    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column
    
    Returns:
        DataFrame with temporal features
    """
    result = df.copy()
    ts = pd.to_datetime(result[timestamp_col])
    
    # Basic time components
    result["hour"] = ts.dt.hour
    result["day_of_week"] = ts.dt.dayofweek
    result["day_of_month"] = ts.dt.day
    result["month"] = ts.dt.month
    result["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    
    # Factory shift indicator (0=Night, 1=Morning, 2=Afternoon)
    result["shift"] = pd.cut(
        ts.dt.hour,
        bins=[-1, 6, 14, 22, 24],
        labels=[0, 1, 2, 0],  # Night, Morning, Afternoon, Night
        ordered=False
    ).astype(int)
    
    # Cyclical encoding of hour (for ML models)
    result["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    
    # Cyclical encoding of day of week
    result["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    result["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    
    return result


def add_deviation_features(df: pd.DataFrame, columns: list, 
                           baseline_window: int = 672) -> pd.DataFrame:
    """
    Calculate deviation from a rolling baseline (how far from 'normal').
    
    Default baseline_window of 672 = 7 days of 15-min data.
    
    Args:
        df: Input DataFrame
        columns: Sensor columns
        baseline_window: Window for computing baseline (in points)
    
    Returns:
        DataFrame with deviation features
    """
    result = df.copy()
    
    for col in columns:
        if col not in result.columns:
            continue
        
        # Long-term rolling mean as baseline
        baseline_mean = result[col].rolling(
            window=baseline_window, min_periods=baseline_window // 2
        ).mean()
        baseline_std = result[col].rolling(
            window=baseline_window, min_periods=baseline_window // 2
        ).std()
        
        # Z-score relative to baseline
        result[f"{col}_zscore"] = (result[col] - baseline_mean) / baseline_std.replace(0, np.nan)
        
        # Absolute deviation from baseline
        result[f"{col}_abs_deviation"] = (result[col] - baseline_mean).abs()
        
        # Percentage deviation from baseline
        result[f"{col}_pct_deviation"] = (
            (result[col] - baseline_mean) / baseline_mean.replace(0, np.nan) * 100
        )
    
    return result


def add_cumulative_stress(df: pd.DataFrame, columns: list,
                          normal_ranges: dict) -> pd.DataFrame:
    """
    Calculate cumulative stress index — how long and how far
    the sensor has been outside its normal range.
    
    This is a key predictive maintenance feature: sensors that have been
    under stress for extended periods are more likely to need maintenance.
    
    Args:
        df: Input DataFrame
        columns: Sensor columns
        normal_ranges: Dict of {column: (min, max)} normal ranges
    
    Returns:
        DataFrame with cumulative stress features
    """
    result = df.copy()
    
    for col in columns:
        if col not in result.columns or col not in normal_ranges:
            continue
        
        nmin, nmax = normal_ranges[col]
        
        # Is the value outside normal range?
        outside_range = ((result[col] < nmin) | (result[col] > nmax)).astype(float)
        result[f"{col}_outside_range"] = outside_range
        
        # Distance from nearest normal boundary
        distance = np.where(
            result[col] < nmin, nmin - result[col],
            np.where(result[col] > nmax, result[col] - nmax, 0)
        )
        result[f"{col}_boundary_distance"] = distance
        
        # Cumulative time outside range (resets when back to normal)
        groups = (outside_range != outside_range.shift()).cumsum()
        result[f"{col}_consecutive_outside"] = (
            outside_range.groupby(groups).cumsum()
        )
        
        # Rolling stress score (sum of boundary distances over 24h)
        result[f"{col}_stress_24h"] = (
            pd.Series(distance).rolling(window=96, min_periods=1).sum()
        )
    
    return result


def compute_sensor_health_score(df: pd.DataFrame, columns: list,
                                normal_ranges: dict,
                                window: int = 96) -> pd.DataFrame:
    """
    Compute a 0-100 health score for each sensor based on:
    - Percentage of time within normal range
    - Magnitude of deviations
    - Rate of change stability
    - Data availability (non-NaN)
    
    Args:
        df: Input DataFrame with sensor data
        columns: Sensor columns to score
        normal_ranges: Dict of {column: (min, max)}
        window: Rolling window for health calculation
    
    Returns:
        DataFrame with health score columns
    """
    result = df.copy()
    
    for col in columns:
        if col not in result.columns or col not in normal_ranges:
            continue
        
        nmin, nmax = normal_ranges[col]
        nrange = nmax - nmin
        
        # Component 1: Within range score (0-40 points)
        in_range = ((result[col] >= nmin) & (result[col] <= nmax)).astype(float)
        range_score = in_range.rolling(window=window, min_periods=1).mean() * 40
        
        # Component 2: Deviation magnitude score (0-30 points)
        center = (nmin + nmax) / 2
        deviation = ((result[col] - center).abs() / (nrange / 2)).clip(0, 2)
        dev_score = (1 - deviation.rolling(window=window, min_periods=1).mean() / 2) * 30
        
        # Component 3: Stability score (0-20 points)
        rate = result[col].diff().abs()
        norm_rate = (rate / nrange).clip(0, 1)
        stability_score = (1 - norm_rate.rolling(window=window, min_periods=1).mean()) * 20
        
        # Component 4: Data availability (0-10 points)
        available = result[col].notna().astype(float)
        availability_score = available.rolling(window=window, min_periods=1).mean() * 10
        
        # Total health score
        result[f"{col}_health_score"] = (
            range_score + dev_score + stability_score + availability_score
        ).clip(0, 100)
    
    return result


def engineer_features(df: pd.DataFrame, sensor_columns: list,
                      normal_ranges: dict,
                      timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Full feature engineering pipeline — applies all transformations.
    
    Args:
        df: Raw sensor DataFrame
        sensor_columns: List of sensor measurement columns
        normal_ranges: Dict of {column: (min, max)} normal operating ranges
        timestamp_col: Timestamp column name
    
    Returns:
        Fully featured DataFrame
    """
    print(f"  Engineering features for {len(sensor_columns)} sensors...")
    
    result = df.copy()
    
    # 1. Temporal features
    result = add_temporal_features(result, timestamp_col)
    print("    ✓ Temporal features added")
    
    # 2. Rolling statistics
    result = add_rolling_features(result, sensor_columns, windows=[4, 16, 96])
    print("    ✓ Rolling features added")
    
    # 3. Rate of change
    result = add_rate_of_change(result, sensor_columns)
    print("    ✓ Rate of change features added")
    
    # 4. Deviation from baseline
    result = add_deviation_features(result, sensor_columns)
    print("    ✓ Deviation features added")
    
    # 5. Cumulative stress
    result = add_cumulative_stress(result, sensor_columns, normal_ranges)
    print("    ✓ Cumulative stress features added")
    
    # 6. Health scores
    result = compute_sensor_health_score(result, sensor_columns, normal_ranges)
    print("    ✓ Health scores computed")
    
    total_features = len(result.columns) - len(df.columns)
    print(f"  ✅ Total new features: {total_features}")
    
    return result
