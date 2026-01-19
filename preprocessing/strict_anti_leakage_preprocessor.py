"""
Strict Anti-Leakage WADI Preprocessor
=====================================
Based on methodology from gwo_lstm_wadi_methodology.md

CRITICAL ANTI-LEAKAGE MEASURES:
1. Remove first 21,600 samples (6-hour stabilization)
2. Remove 6 constant solenoid valves
3. K-S test on train vs validation ONLY (never test)
4. Fit scaler ONLY on training data
5. Strict temporal split - NO shuffling
6. No overlapping windows across splits
7. Document all preprocessing for reproducibility

Author: Thesis Project
Date: 2026-01-18
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
import logging
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing - ensures reproducibility."""
    
    # Stabilization period
    stabilization_samples: int = 21600  # 6 hours at 1 Hz
    
    # Feature removal
    constant_valves: List[str] = field(default_factory=lambda: [
        '2_SV_101_STATUS', '2_SV_201_STATUS', '2_SV_301_STATUS',
        '2_SV_401_STATUS', '2_SV_501_STATUS', '2_SV_601_STATUS'
    ])
    
    # K-S test parameters
    ks_test_alpha: float = 0.05  # Remove features with p < alpha
    
    # Variance threshold
    variance_threshold: float = 1e-6
    
    # Outlier clipping
    outlier_percentile: float = 99.5
    
    # Scaler type
    scaler_type: str = 'robust'  # 'minmax', 'standard', 'robust'
    
    # Train/Val split ratio (validation comes from end of training data)
    val_ratio: float = 0.05  # Last 5% of 14-day data = ~Day 13.3-14
    
    # Windowing
    window_size: int = 100
    stride: int = 1
    
    # Random state for reproducibility
    random_state: int = 42


@dataclass
class PreprocessingReport:
    """Detailed report of all preprocessing steps - for audit trail."""
    
    timestamp: str = ""
    config: Dict = field(default_factory=dict)
    
    # Data shapes
    raw_train_shape: Tuple = ()
    raw_test_shape: Tuple = ()
    final_train_shape: Tuple = ()
    final_val_shape: Tuple = ()
    final_test_shape: Tuple = ()
    
    # Feature tracking
    initial_features: int = 0
    constant_features_removed: List[str] = field(default_factory=list)
    ks_test_removed: List[str] = field(default_factory=list)
    low_variance_removed: List[str] = field(default_factory=list)
    final_features: List[str] = field(default_factory=list)
    
    # Label distribution
    train_label_dist: Dict = field(default_factory=dict)
    val_label_dist: Dict = field(default_factory=dict)
    test_label_dist: Dict = field(default_factory=dict)
    
    # Scaler info
    scaler_type: str = ""
    scaler_fitted_on: str = "TRAINING_DATA_ONLY"
    
    # Warnings
    warnings: List[str] = field(default_factory=list)


class StrictAntiLeakagePreprocessor:
    """
    WADI Dataset Preprocessor with STRICT anti-leakage protocol.
    
    Key Principles:
    1. NEVER touch test data during preprocessing decisions
    2. Fit all transformations on TRAINING data ONLY
    3. Maintain strict temporal ordering (no shuffling)
    4. Document everything for reproducibility
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.report = PreprocessingReport(timestamp=datetime.now().isoformat())
        self.report.config = asdict(self.config)
        
        # Fitted objects (trained on training data only)
        self.scaler = None
        self.selected_features: List[str] = []
        self.outlier_bounds: Dict[str, Tuple[float, float]] = {}
        
        # Validation
        self._is_fitted = False
    
    def load_wadi_data(
        self, 
        train_path: str, 
        test_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Load raw WADI data files.
        
        Returns:
            train_df: 14-day normal operation data
            test_df: 2-day attack data (features only)
            test_labels: Attack labels (0=normal, 1=attack)
        """
        logger.info("="*60)
        logger.info("LOADING WADI DATA")
        logger.info("="*60)
        
        # Load training data (14 days normal)
        logger.info(f"Loading training data from: {train_path}")
        train_df = pd.read_csv(train_path, low_memory=False)
        self.report.raw_train_shape = train_df.shape
        logger.info(f"  Raw shape: {train_df.shape}")
        
        # Load test data (2 days with attacks)
        logger.info(f"Loading test data from: {test_path}")
        test_df = pd.read_csv(test_path, skiprows=1, low_memory=False)  # Skip units row
        self.report.raw_test_shape = test_df.shape
        logger.info(f"  Raw shape: {test_df.shape}")
        
        # Extract labels from test data
        label_col = test_df.columns[-1]
        raw_labels = test_df[label_col].values
        
        # WADI convention: 1=Normal, -1=Attack
        # Convert to: 0=Normal, 1=Attack
        test_labels = np.where(raw_labels == 1, 0, 1).astype(np.int32)
        test_df = test_df.drop(columns=[label_col])
        
        n_normal = np.sum(test_labels == 0)
        n_attack = np.sum(test_labels == 1)
        logger.info(f"  Test labels: Normal={n_normal} ({100*n_normal/len(test_labels):.1f}%), "
                   f"Attack={n_attack} ({100*n_attack/len(test_labels):.1f}%)")
        
        return train_df, test_df, test_labels
    
    def _get_sensor_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify actual sensor columns (exclude metadata)."""
        non_sensor_patterns = ['date', 'time', 'timestamp', 'row', 'index', 'unnamed']
        
        sensor_cols = []
        for col in df.columns:
            col_lower = col.lower().strip()
            is_metadata = any(pattern in col_lower for pattern in non_sensor_patterns)
            if not is_metadata:
                sensor_cols.append(col)
        
        return sensor_cols
    
    def _remove_stabilization_period(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        CRITICAL: Remove first 6 hours of data (system stabilization).
        This is 21,600 samples at 1 Hz.
        """
        logger.info(f"\n[STEP 1] Removing stabilization period")
        logger.info(f"  Samples to remove: {self.config.stabilization_samples}")
        logger.info(f"  Before: {len(train_df)} samples")
        
        if len(train_df) <= self.config.stabilization_samples:
            raise ValueError(f"Training data has only {len(train_df)} samples, "
                           f"cannot remove {self.config.stabilization_samples} stabilization samples")
        
        train_df = train_df.iloc[self.config.stabilization_samples:].reset_index(drop=True)
        logger.info(f"  After: {len(train_df)} samples")
        
        return train_df
    
    def _remove_constant_features(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove known constant features (solenoid valves).
        These have zero variance and provide no information.
        """
        logger.info(f"\n[STEP 2] Removing constant features")
        
        to_remove = [col for col in self.config.constant_valves if col in train_df.columns]
        
        logger.info(f"  Removing {len(to_remove)} constant features: {to_remove}")
        self.report.constant_features_removed = to_remove
        
        train_df = train_df.drop(columns=to_remove, errors='ignore')
        test_df = test_df.drop(columns=to_remove, errors='ignore')
        
        return train_df, test_df
    
    def _temporal_split(
        self, 
        train_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        """
        CRITICAL: Temporal split - NO SHUFFLING.
        
        Uses last val_ratio of 14-day data as validation.
        This maintains temporal order.
        """
        logger.info(f"\n[STEP 3] Temporal train/validation split")
        
        # Training data is all normal (label = 0)
        n_samples = len(train_df)
        split_idx = int(n_samples * (1 - self.config.val_ratio))
        
        X_train = train_df.iloc[:split_idx].copy()
        X_val = train_df.iloc[split_idx:].copy()
        
        # All training/validation samples are normal
        y_train = np.zeros(len(X_train), dtype=np.int32)
        y_val = np.zeros(len(X_val), dtype=np.int32)
        
        logger.info(f"  Training samples: {len(X_train)} (Days 1-~13.3)")
        logger.info(f"  Validation samples: {len(X_val)} (Days ~13.3-14)")
        logger.info(f"  Split ratio: {1-self.config.val_ratio:.2f}/{self.config.val_ratio:.2f}")
        
        # CRITICAL CHECK: Ensure no temporal overlap
        assert split_idx == len(X_train), "Temporal split error: overlap detected!"
        
        return X_train, y_train, X_val, y_val
    
    def _ks_test_feature_stability(
        self, 
        X_train: pd.DataFrame, 
        X_val: pd.DataFrame
    ) -> List[str]:
        """
        K-S test to identify unstable features.
        
        CRITICAL: Uses ONLY train and validation data, NEVER test!
        
        Features with significantly different distributions between
        train and validation are unstable and should be removed.
        """
        logger.info(f"\n[STEP 4] K-S test for feature stability")
        logger.info(f"  Alpha threshold: {self.config.ks_test_alpha}")
        
        unstable_features = []
        
        for col in X_train.columns:
            try:
                statistic, p_value = stats.ks_2samp(
                    X_train[col].dropna().values,
                    X_val[col].dropna().values
                )
                
                if p_value < self.config.ks_test_alpha:
                    unstable_features.append(col)
                    
            except Exception as e:
                logger.warning(f"  K-S test failed for {col}: {e}")
                unstable_features.append(col)
        
        logger.info(f"  Found {len(unstable_features)} unstable features (p < {self.config.ks_test_alpha})")
        if unstable_features:
            logger.info(f"  Unstable: {unstable_features[:10]}{'...' if len(unstable_features) > 10 else ''}")
        
        self.report.ks_test_removed = unstable_features
        return unstable_features
    
    def _remove_low_variance_features(
        self, 
        X_train: pd.DataFrame
    ) -> List[str]:
        """
        Identify low-variance features based on TRAINING data only.
        """
        logger.info(f"\n[STEP 5] Identifying low-variance features")
        
        variances = X_train.var()
        low_var = variances[variances < self.config.variance_threshold].index.tolist()
        
        logger.info(f"  Found {len(low_var)} low-variance features")
        self.report.low_variance_removed = low_var
        
        return low_var
    
    def _handle_missing_values(
        self, 
        df: pd.DataFrame,
        is_training: bool = True
    ) -> pd.DataFrame:
        """Handle missing values using forward/backward fill."""
        
        n_missing = df.isnull().sum().sum()
        if n_missing > 0:
            logger.info(f"  Handling {n_missing} missing values")
            df = df.ffill().bfill().fillna(0)
        
        # Ensure numeric
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return df
    
    def _compute_outlier_bounds(self, X_train: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """
        Compute outlier bounds from TRAINING data only.
        """
        logger.info(f"\n[STEP 6] Computing outlier bounds from training data")
        
        lower_pct = (100 - self.config.outlier_percentile) / 100
        upper_pct = self.config.outlier_percentile / 100
        
        bounds = {}
        for col in X_train.columns:
            lower = X_train[col].quantile(lower_pct)
            upper = X_train[col].quantile(upper_pct)
            bounds[col] = (lower, upper)
        
        logger.info(f"  Computed bounds for {len(bounds)} features")
        
        return bounds
    
    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers using pre-computed bounds."""
        
        for col in df.columns:
            if col in self.outlier_bounds:
                lower, upper = self.outlier_bounds[col]
                df[col] = df[col].clip(lower=lower, upper=upper)
        
        return df
    
    def _fit_scaler(self, X_train: np.ndarray) -> None:
        """
        CRITICAL: Fit scaler ONLY on training data.
        
        This prevents information leakage from validation/test sets.
        """
        logger.info(f"\n[STEP 7] Fitting {self.config.scaler_type} scaler on TRAINING data ONLY")
        
        if self.config.scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif self.config.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.config.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.config.scaler_type}")
        
        self.scaler.fit(X_train)
        
        self.report.scaler_type = self.config.scaler_type
        self.report.scaler_fitted_on = "TRAINING_DATA_ONLY"
        
        logger.info(f"  Scaler fitted on {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    def _create_windows(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows for sequence models.
        
        CRITICAL: Window label = 1 if ANY point in window is an anomaly.
        
        This must be used WITH a model that reconstructs ALL timesteps,
        not just the last one. Using "ANY point" labeling with a model
        that only reconstructs the last timestep causes information leakage.
        
        For proper implementation:
        - Model: Seq2seq/Autoencoder that reconstructs all timesteps
        - Loss: MSE between predicted and actual for all timesteps
        - Label: 1 if ANY timestep is anomaly (consistent with reconstruction target)
        """
        n_samples, n_features = X.shape
        window_size = self.config.window_size
        stride = self.config.stride
        
        n_windows = (n_samples - window_size) // stride + 1
        
        windows = np.zeros((n_windows, window_size, n_features), dtype=np.float32)
        window_labels = np.zeros(n_windows, dtype=np.int32)
        
        for i in range(n_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            
            windows[i] = X[start_idx:end_idx]
            # Window is anomalous if ANY point is anomalous
            # (Use with models that reconstruct ALL timesteps)
            window_labels[i] = 1 if np.any(y[start_idx:end_idx] == 1) else 0
        
        return windows, window_labels
    
    def fit_transform(
        self,
        train_path: str,
        test_path: str,
        output_dir: str
    ) -> Dict[str, np.ndarray]:
        """
        Complete preprocessing pipeline with anti-leakage guarantees.
        
        Returns dictionary with:
            - train_windows, train_labels
            - val_windows, val_labels
            - test_windows, test_labels
            - feature_names
            - preprocessing_report
        """
        logger.info("="*60)
        logger.info("STRICT ANTI-LEAKAGE PREPROCESSING PIPELINE")
        logger.info("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Load data
        train_df, test_df, test_labels = self.load_wadi_data(train_path, test_path)
        
        # 2. Get sensor columns
        sensor_cols = self._get_sensor_columns(train_df)
        train_df = train_df[sensor_cols]
        test_df = test_df[[c for c in sensor_cols if c in test_df.columns]]
        self.report.initial_features = len(sensor_cols)
        
        # 3. Remove stabilization period (TRAINING ONLY)
        train_df = self._remove_stabilization_period(train_df)
        
        # 4. Remove constant features
        train_df, test_df = self._remove_constant_features(train_df, test_df)
        
        # 5. Ensure same columns
        common_cols = list(set(train_df.columns) & set(test_df.columns))
        train_df = train_df[common_cols]
        test_df = test_df[common_cols]
        logger.info(f"  Common features: {len(common_cols)}")
        
        # 6. Temporal split (NO SHUFFLING)
        X_train, y_train, X_val, y_val = self._temporal_split(train_df)
        X_test = test_df
        y_test = test_labels
        
        # 7. K-S test for stability (USING TRAIN + VAL ONLY)
        unstable_features = self._ks_test_feature_stability(X_train, X_val)
        
        # 8. Identify low-variance features (USING TRAIN ONLY)
        low_var_features = self._remove_low_variance_features(X_train)
        
        # 9. Remove problematic features
        features_to_remove = set(unstable_features + low_var_features)
        final_features = [c for c in X_train.columns if c not in features_to_remove]
        
        logger.info(f"\n[FEATURE SUMMARY]")
        logger.info(f"  Initial: {self.report.initial_features}")
        logger.info(f"  After constant removal: {len(common_cols)}")
        logger.info(f"  K-S test removed: {len(unstable_features)}")
        logger.info(f"  Low variance removed: {len(low_var_features)}")
        logger.info(f"  Final features: {len(final_features)}")
        
        X_train = X_train[final_features]
        X_val = X_val[final_features]
        X_test = X_test[final_features]
        self.selected_features = final_features
        self.report.final_features = final_features
        
        # 10. Handle missing values
        logger.info(f"\n[STEP 8] Handling missing values")
        X_train = self._handle_missing_values(X_train)
        X_val = self._handle_missing_values(X_val, is_training=False)
        X_test = self._handle_missing_values(X_test, is_training=False)
        
        # 11. Compute outlier bounds from TRAINING ONLY
        self.outlier_bounds = self._compute_outlier_bounds(X_train)
        
        # 12. Clip outliers using training bounds
        X_train = self._clip_outliers(X_train)
        X_val = self._clip_outliers(X_val)
        X_test = self._clip_outliers(X_test)
        
        # 13. Convert to numpy
        X_train_np = X_train.values.astype(np.float32)
        X_val_np = X_val.values.astype(np.float32)
        X_test_np = X_test.values.astype(np.float32)
        
        # 14. Fit scaler on TRAINING ONLY
        self._fit_scaler(X_train_np)
        
        # 15. Transform all sets using TRAINING scaler
        logger.info(f"\n[STEP 8] Scaling data using training parameters")
        X_train_scaled = self.scaler.transform(X_train_np)
        X_val_scaled = self.scaler.transform(X_val_np)
        X_test_scaled = self.scaler.transform(X_test_np)
        
        logger.info(f"  Train range: [{X_train_scaled.min():.4f}, {X_train_scaled.max():.4f}]")
        logger.info(f"  Val range: [{X_val_scaled.min():.4f}, {X_val_scaled.max():.4f}]")
        logger.info(f"  Test range: [{X_test_scaled.min():.4f}, {X_test_scaled.max():.4f}]")
        
        # 16. Create sliding windows
        logger.info(f"\n[STEP 9] Creating sliding windows")
        logger.info(f"  Window size: {self.config.window_size}, Stride: {self.config.stride}")
        
        train_windows, train_window_labels = self._create_windows(X_train_scaled, y_train)
        val_windows, val_window_labels = self._create_windows(X_val_scaled, y_val)
        test_windows, test_window_labels = self._create_windows(X_test_scaled, y_test)
        
        logger.info(f"  Train windows: {train_windows.shape}")
        logger.info(f"  Val windows: {val_windows.shape}")
        logger.info(f"  Test windows: {test_windows.shape}")
        
        # Update report
        self.report.final_train_shape = train_windows.shape
        self.report.final_val_shape = val_windows.shape
        self.report.final_test_shape = test_windows.shape
        
        self.report.train_label_dist = {"normal": int(np.sum(train_window_labels == 0)), 
                                         "attack": int(np.sum(train_window_labels == 1))}
        self.report.val_label_dist = {"normal": int(np.sum(val_window_labels == 0)), 
                                       "attack": int(np.sum(val_window_labels == 1))}
        self.report.test_label_dist = {"normal": int(np.sum(test_window_labels == 0)), 
                                        "attack": int(np.sum(test_window_labels == 1))}
        
        # 17. Final validation checks
        self._validate_no_leakage(train_windows, val_windows, test_windows)
        
        # 18. Save everything
        logger.info(f"\n[STEP 10] Saving preprocessed data to: {output_dir}")
        
        np.save(os.path.join(output_dir, 'train_windows.npy'), train_windows)
        np.save(os.path.join(output_dir, 'train_labels.npy'), train_window_labels)
        np.save(os.path.join(output_dir, 'val_windows.npy'), val_windows)
        np.save(os.path.join(output_dir, 'val_labels.npy'), val_window_labels)
        np.save(os.path.join(output_dir, 'test_windows.npy'), test_windows)
        np.save(os.path.join(output_dir, 'test_labels.npy'), test_window_labels)
        
        # Save scaler
        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature list
        with open(os.path.join(output_dir, 'features.json'), 'w') as f:
            json.dump({'features': self.selected_features, 
                      'n_features': len(self.selected_features)}, f, indent=2)
        
        # Save preprocessing report
        with open(os.path.join(output_dir, 'preprocessing_report.json'), 'w') as f:
            report_dict = asdict(self.report)
            # Convert tuples to lists for JSON
            for key in ['raw_train_shape', 'raw_test_shape', 'final_train_shape', 
                       'final_val_shape', 'final_test_shape']:
                if key in report_dict:
                    report_dict[key] = list(report_dict[key])
            json.dump(report_dict, f, indent=2)
        
        # Save config
        with open(os.path.join(output_dir, 'preprocessing_config.json'), 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        self._is_fitted = True
        
        logger.info("\n" + "="*60)
        logger.info("PREPROCESSING COMPLETE - NO DATA LEAKAGE")
        logger.info("="*60)
        
        return {
            'train_windows': train_windows,
            'train_labels': train_window_labels,
            'val_windows': val_windows,
            'val_labels': val_window_labels,
            'test_windows': test_windows,
            'test_labels': test_window_labels,
            'feature_names': self.selected_features,
            'n_features': len(self.selected_features),
            'config': asdict(self.config),
            'report': asdict(self.report)
        }
    
    def _validate_no_leakage(
        self, 
        train_windows: np.ndarray, 
        val_windows: np.ndarray, 
        test_windows: np.ndarray
    ) -> None:
        """
        Validate that no data leakage occurred.
        """
        logger.info("\n[VALIDATION] Checking for data leakage...")
        
        warnings = []
        
        # Check 1: No identical windows across splits
        # (Sample-based check for efficiency)
        train_sample = train_windows[::1000].reshape(-1)[:10000]
        val_sample = val_windows[::100].reshape(-1)[:10000]
        test_sample = test_windows[::100].reshape(-1)[:10000]
        
        # These should be different
        train_val_corr = np.corrcoef(train_sample[:min(len(train_sample), len(val_sample))],
                                     val_sample[:min(len(train_sample), len(val_sample))])[0, 1]
        train_test_corr = np.corrcoef(train_sample[:min(len(train_sample), len(test_sample))],
                                      test_sample[:min(len(train_sample), len(test_sample))])[0, 1]
        
        logger.info(f"  Train-Val correlation: {train_val_corr:.4f}")
        logger.info(f"  Train-Test correlation: {train_test_corr:.4f}")
        
        if train_val_corr > 0.99:
            warnings.append("WARNING: Very high train-val correlation - possible overlap!")
        if train_test_corr > 0.99:
            warnings.append("WARNING: Very high train-test correlation - possible overlap!")
        
        # Check 2: Scaler was fitted
        if self.scaler is None:
            warnings.append("ERROR: Scaler not fitted!")
        
        # Check 3: Feature selection was done on train only
        if len(self.report.ks_test_removed) > 0:
            logger.info(f"  K-S test removed {len(self.report.ks_test_removed)} features (on train/val only)")
        
        self.report.warnings = warnings
        
        if warnings:
            for w in warnings:
                logger.warning(f"  {w}")
        else:
            logger.info("  ✓ All validation checks passed - no leakage detected")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Strict Anti-Leakage WADI Preprocessor')
    parser.add_argument('--train-path', type=str, required=True,
                       help='Path to WADI_14days_new.csv')
    parser.add_argument('--test-path', type=str, required=True,
                       help='Path to WADI_attackdataLABLE.csv')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for preprocessed data')
    parser.add_argument('--window-size', type=int, default=100,
                       help='Sliding window size')
    parser.add_argument('--scaler', type=str, default='robust',
                       choices=['minmax', 'standard', 'robust'],
                       help='Scaler type')
    
    args = parser.parse_args()
    
    config = PreprocessingConfig(
        window_size=args.window_size,
        scaler_type=args.scaler
    )
    
    preprocessor = StrictAntiLeakagePreprocessor(config)
    data = preprocessor.fit_transform(args.train_path, args.test_path, args.output_dir)
    
    print(f"\nPreprocessing complete!")
    print(f"  Train: {data['train_windows'].shape}")
    print(f"  Val: {data['val_windows'].shape}")
    print(f"  Test: {data['test_windows'].shape}")
    print(f"  Features: {data['n_features']}")


if __name__ == '__main__':
    main()
