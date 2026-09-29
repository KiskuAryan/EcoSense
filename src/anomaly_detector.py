"""
EcoSense - Anomaly Detection Module
=====================================

Implements multiple anomaly detection algorithms for environmental
sensor time-series data. Each method has different strengths:

    - Z-Score:            Fast, interpretable, good for Gaussian data
    - IQR:                Robust to outliers, non-parametric
    - Rolling Statistics: Captures local context, good for trending data
    - Isolation Forest:   ML-based, captures complex multivariate anomalies

The module provides a unified interface to run all methods and compare results.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats as sp_stats


class ZScoreDetector:
    """
    Z-Score based anomaly detection.
    
    Flags data points that deviate more than `threshold` standard deviations
    from the mean. Simple, fast, and highly interpretable.
    
    Best for: Stationary data with approximately normal distribution.
    Weakness: Sensitive to non-stationarity and seasonal patterns.
    """
    
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.name = "Z-Score"
        self.mean_ = None
        self.std_ = None
    
    def fit(self, data: pd.Series) -> "ZScoreDetector":
        """Compute mean and std from training data."""
        self.mean_ = data.mean()
        self.std_ = data.std()
        return self
    
    def detect(self, data: pd.Series) -> pd.DataFrame:
        """
        Detect anomalies using Z-score method.
        
        Args:
            data: Series of sensor values
        
        Returns:
            DataFrame with columns: value, z_score, is_anomaly, severity
        """
        if self.mean_ is None:
            self.fit(data)
        
        z_scores = (data - self.mean_) / self.std_ if self.std_ != 0 else pd.Series(0, index=data.index)
        is_anomaly = z_scores.abs() > self.threshold
        
        # Severity: how many thresholds beyond the limit
        severity = (z_scores.abs() - self.threshold).clip(lower=0)
        severity = severity / severity.max() if severity.max() > 0 else severity
        
        return pd.DataFrame({
            "value": data,
            "z_score": z_scores,
            "is_anomaly": is_anomaly.astype(int),
            "severity": severity,
            "method": self.name
        })


class IQRDetector:
    """
    Interquartile Range (IQR) based anomaly detection.
    
    Uses Q1, Q3, and IQR to define normal bounds. Values outside
    [Q1 - factor*IQR, Q3 + factor*IQR] are flagged as anomalies.
    
    Best for: Non-Gaussian data, robust to extreme outliers in training data.
    Weakness: Doesn't account for temporal patterns.
    """
    
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.name = "IQR"
        self.q1_ = None
        self.q3_ = None
        self.iqr_ = None
        self.lower_bound_ = None
        self.upper_bound_ = None
    
    def fit(self, data: pd.Series) -> "IQRDetector":
        """Compute IQR bounds from training data."""
        self.q1_ = data.quantile(0.25)
        self.q3_ = data.quantile(0.75)
        self.iqr_ = self.q3_ - self.q1_
        self.lower_bound_ = self.q1_ - self.factor * self.iqr_
        self.upper_bound_ = self.q3_ + self.factor * self.iqr_
        return self
    
    def detect(self, data: pd.Series) -> pd.DataFrame:
        """
        Detect anomalies using IQR method.
        
        Args:
            data: Series of sensor values
        
        Returns:
            DataFrame with columns: value, is_anomaly, severity, lower_bound, upper_bound
        """
        if self.q1_ is None:
            self.fit(data)
        
        is_anomaly = (data < self.lower_bound_) | (data > self.upper_bound_)
        
        # Severity based on distance from nearest bound
        distance = np.where(
            data < self.lower_bound_, self.lower_bound_ - data,
            np.where(data > self.upper_bound_, data - self.upper_bound_, 0)
        )
        severity = pd.Series(distance, index=data.index)
        severity = severity / severity.max() if severity.max() > 0 else severity
        
        return pd.DataFrame({
            "value": data,
            "is_anomaly": is_anomaly.astype(int),
            "severity": severity,
            "lower_bound": self.lower_bound_,
            "upper_bound": self.upper_bound_,
            "method": self.name
        })


class RollingStatsDetector:
    """
    Rolling Statistics anomaly detection.
    
    Uses a rolling window to compute local mean and standard deviation,
    then flags points that deviate significantly from local statistics.
    
    Best for: Non-stationary data with trends and seasonal patterns.
    Note on Window Alignment:
        - center=True: Used for historical/retrospective offline anomaly auditing. Avoids phase lag
          by using centered statistics (sees both preceding and succeeding context).
        - center=False: Required for live/online streaming deployment where future observations
          are not yet available (causal trailing window).
    """
    
    def __init__(self, window: int = 96, threshold: float = 3.0, center: bool = True):
        """
        Args:
            window: Rolling window size (default 96 = 24 hours at 15-min intervals)
            threshold: Number of rolling std deviations for anomaly threshold
            center: If True (default), computes centered window for retrospective batch audit.
                    If False, computes trailing window for causal streaming detection.
        """
        self.window = window
        self.threshold = threshold
        self.center = center
        self.name = "Rolling Stats"
    
    def detect(self, data: pd.Series) -> pd.DataFrame:
        """
        Detect anomalies using rolling statistics.
        
        Args:
            data: Series of sensor values
        
        Returns:
            DataFrame with columns: value, rolling_mean, rolling_std,
                                    upper_band, lower_band, is_anomaly, severity
        """
        rolling_mean = data.rolling(window=self.window, min_periods=1, center=self.center).mean()
        rolling_std = data.rolling(window=self.window, min_periods=1, center=self.center).std()
        
        upper_band = rolling_mean + self.threshold * rolling_std
        lower_band = rolling_mean - self.threshold * rolling_std
        
        is_anomaly = (data > upper_band) | (data < lower_band)
        
        # Severity
        distance = np.where(
            data > upper_band, data - upper_band,
            np.where(data < lower_band, lower_band - data, 0)
        )
        severity = pd.Series(distance, index=data.index)
        severity = severity / severity.max() if severity.max() > 0 else severity
        
        return pd.DataFrame({
            "value": data,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "is_anomaly": is_anomaly.astype(int),
            "severity": severity,
            "method": self.name
        })


class IsolationForestDetector:
    """
    Isolation Forest anomaly detection (scikit-learn).
    
    An unsupervised ML model that isolates anomalies by randomly
    partitioning data. Anomalies require fewer partitions to isolate.
    
    Best for: Multivariate data, complex non-linear patterns.
    Weakness: Less interpretable, requires feature engineering.
    """
    
    def __init__(self, contamination: float = 0.05, n_estimators: int = 100,
                 random_state: int = 42):
        """
        Args:
            contamination: Expected proportion of anomalies in the data
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.name = "Isolation Forest"
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, features: pd.DataFrame) -> "IsolationForestDetector":
        """
        Fit the Isolation Forest model.
        
        Args:
            features: DataFrame of features (multiple columns for multivariate detection)
        """
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Handle NaN values
        clean_features = features.fillna(features.mean())
        scaled = self.scaler.fit_transform(clean_features)
        self.model.fit(scaled)
        
        return self
    
    def detect(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies using the fitted Isolation Forest.
        
        Args:
            features: DataFrame of features
        
        Returns:
            DataFrame with columns: is_anomaly, anomaly_score, severity
        """
        if self.model is None:
            self.fit(features)
        
        clean_features = features.fillna(features.mean())
        scaled = self.scaler.transform(clean_features)
        
        # Predict: -1 = anomaly, 1 = normal
        predictions = self.model.predict(scaled)
        
        # Anomaly scores (lower = more anomalous)
        scores = self.model.decision_function(scaled)
        
        is_anomaly = (predictions == -1).astype(int)
        
        # Convert scores to 0-1 severity (invert so higher = more anomalous)
        severity = pd.Series(-scores, index=features.index)
        severity = (severity - severity.min()) / (severity.max() - severity.min())
        
        return pd.DataFrame({
            "is_anomaly": is_anomaly,
            "anomaly_score": scores,
            "severity": severity,
            "method": self.name
        }, index=features.index)


# ─────────────────────────────────────────────────────────
# Unified Detection Pipeline
# ─────────────────────────────────────────────────────────

def run_all_detectors(data: pd.Series, 
                      features_df: pd.DataFrame = None,
                      zscore_threshold: float = 3.0,
                      iqr_factor: float = 1.5,
                      rolling_window: int = 96,
                      rolling_threshold: float = 3.0,
                      rolling_center: bool = True,
                      if_contamination: float = 0.05) -> dict:
    """
    Run all anomaly detection methods on the same data.
    
    Args:
        data: Series of sensor values for univariate methods
        features_df: DataFrame of features for Isolation Forest (optional)
        zscore_threshold: Z-score threshold
        iqr_factor: IQR multiplier
        rolling_window: Rolling window size
        rolling_threshold: Rolling stats threshold
        rolling_center: True for retrospective offline audit, False for causal streaming
        if_contamination: Isolation Forest contamination parameter
    
    Returns:
        Dictionary of {method_name: result_DataFrame}
    """
    results = {}
    
    # Drop NaN for fitting
    clean_data = data.dropna()
    
    # 1. Z-Score
    zscore = ZScoreDetector(threshold=zscore_threshold)
    zscore.fit(clean_data)
    results["Z-Score"] = zscore.detect(data.fillna(data.mean()))
    
    # 2. IQR
    iqr = IQRDetector(factor=iqr_factor)
    iqr.fit(clean_data)
    results["IQR"] = iqr.detect(data.fillna(data.mean()))
    
    # 3. Rolling Statistics
    rolling = RollingStatsDetector(window=rolling_window, threshold=rolling_threshold, center=rolling_center)
    results["Rolling Stats"] = rolling.detect(data.fillna(data.mean()))
    
    # 4. Isolation Forest (if features provided)
    if features_df is not None:
        iso_forest = IsolationForestDetector(contamination=if_contamination)
        iso_forest.fit(features_df)
        results["Isolation Forest"] = iso_forest.detect(features_df)
    
    return results


def compare_detectors(results: dict, ground_truth: pd.Series = None) -> pd.DataFrame:
    """
    Compare detection results across all methods.
    
    Args:
        results: Dictionary from run_all_detectors
        ground_truth: Optional Series of true anomaly labels (0/1)
    
    Returns:
        Comparison DataFrame with metrics per method
    """
    comparison = []
    
    for method_name, result_df in results.items():
        detected = result_df["is_anomaly"].sum()
        total = len(result_df)
        detection_rate = detected / total * 100
        
        row = {
            "method": method_name,
            "total_points": total,
            "anomalies_detected": detected,
            "detection_rate_pct": round(detection_rate, 2),
        }
        
        # If ground truth is available, compute precision/recall/F1
        if ground_truth is not None:
            # Align indices
            gt = ground_truth.reindex(result_df.index).fillna(0).astype(int)
            pred = result_df["is_anomaly"].astype(int)
            
            tp = ((pred == 1) & (gt == 1)).sum()
            fp = ((pred == 1) & (gt == 0)).sum()
            fn = ((pred == 0) & (gt == 1)).sum()
            tn = ((pred == 0) & (gt == 0)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            row.update({
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            })
        
        comparison.append(row)
    
    return pd.DataFrame(comparison)


def create_ensemble_prediction(results: dict, 
                                min_votes: int = 2) -> pd.Series:
    """
    Create an ensemble anomaly prediction using majority voting.
    
    A point is flagged as anomalous if at least `min_votes` methods agree.
    
    Args:
        results: Dictionary from run_all_detectors
        min_votes: Minimum number of methods that must agree
    
    Returns:
        Series of ensemble anomaly labels (0/1)
    """
    # Collect all predictions
    vote_df = pd.DataFrame()
    for method_name, result_df in results.items():
        vote_df[method_name] = result_df["is_anomaly"]
    
    # Count votes
    total_votes = vote_df.sum(axis=1)
    
    # Ensemble decision
    ensemble = (total_votes >= min_votes).astype(int)
    
    return ensemble
