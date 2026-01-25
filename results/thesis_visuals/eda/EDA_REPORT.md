# Exploratory Data Analysis Report

## Overview

This report presents comprehensive exploratory data analysis of the WADI dataset, documenting data characteristics before and after preprocessing.

**Report Generated:** January 2026

---

## 1. Dataset Overview

### 1.1 Raw Dataset Statistics

| Metric | Training Data | Test Data |
|--------|---------------|-----------|
| Total Samples | 784,571 | 172,804 |
| Total Features | 125 | 131 |
| Duration | 14 days | 2 days |
| Sampling Rate | 1 second | 1 second |

### 1.2 Preprocessed Dataset Statistics

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| Windows | 141,206 | 15,689 | 34,541 |
| Window Size | 100 timesteps | 100 timesteps | 100 timesteps |
| Features | 30 | 30 | 30 |
| Tensor Shape | (141,206, 100, 30) | (15,689, 100, 30) | (34,541, 100, 30) |

### 1.3 Test Set Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Normal | 32,268 | 93.42% |
| Anomaly | 2,273 | 6.58% |

---

## 2. Missing Value Analysis

### 2.1 Overall Missing Values

| Metric | Before Preprocessing | After Preprocessing |
|--------|---------------------|---------------------|
| Total Missing Values | 3,138,318 | 0 |
| Total Cells | 100,425,088 | 423,618,000 |
| Missing Percentage | 3.1250% | 0.0000% |
| Features with Missing | 8 | 0 |

### 2.2 Missing Value Treatment

Missing values were handled using forward-fill followed by backward-fill imputation, ensuring temporal continuity in sensor readings. This approach is appropriate for SCADA data where sensor readings are expected to be continuous.

### 2.3 Features with Missing Values (Top 10)

| Feature | Missing Count | Missing % |
|---------|---------------|----------|
| 2_LS_001_AL | 784,571 | 100.0000% |
| 2_LS_002_AL | 784,571 | 100.0000% |
| 2_P_001_STATUS | 784,571 | 100.0000% |
| 2_P_002_STATUS | 784,571 | 100.0000% |
| 1_AIT_002_PV | 12 | 0.0015% |
| 2B_AIT_004_PV | 10 | 0.0013% |
| 1_AIT_004_PV | 6 | 0.0008% |
| 3_AIT_004_PV | 6 | 0.0008% |

---

## 3. Correlation Analysis

### 3.1 Correlation Statistics

| Metric | Before Preprocessing | After Preprocessing |
|--------|---------------------|---------------------|
| Mean Absolute Correlation | 0.2845 | 0.2831 |
| Maximum Correlation | 1.0000 | 1.0000 |
| Minimum Correlation | 0.0025 | 0.0209 |
| Highly Correlated Pairs (r > 0.8) | 6 | 8 |

### 3.2 Correlation Matrix Interpretation

The correlation analysis reveals:

1. **Feature Selection Effectiveness**: The preprocessing pipeline successfully identified and retained features with meaningful inter-relationships while removing redundant sensors.

2. **Preserved Correlations**: The number of highly correlated pairs increased slightly from 6 to 8, indicating that important sensor relationships are preserved while redundant features were removed.

3. **Preserved Sensor Relationships**: Critical cross-sensor correlations that indicate physical system relationships are preserved in the preprocessed data.

---

## 4. Feature Statistics

### 4.1 Data Shape Transformation

```
Raw Data:         (784,571, 125)
                       |
                       v
Preprocessed:     (141,206, 100, 30)
                  (windows, timesteps, features)
```

### 4.2 Feature Reduction Summary

| Stage | Features | Reduction |
|-------|----------|-----------|
| Original | 125 | - |
| After K-S Test | ~50 | ~60% |
| After Variance Filter | 30 | 76% total |

### 4.3 Normalization

All features are normalized to [0, 1] range using min-max scaling computed on the training set. The same scaling parameters are applied to validation and test sets to prevent data leakage.

**Note:** Test data values may slightly exceed [0, 1] bounds when test distributions differ from training, which is expected behavior for anomalous data.

---

## 5. Visualizations

The following visualizations are generated in `results/thesis_visuals/eda/`:

1. **correlation_matrix_comparison.png** - Side-by-side correlation matrices
2. **missing_values_analysis.png** - Missing value distribution
3. **statistics_summary.png** - Comprehensive statistics comparison

---

## 6. Key Findings

### 6.1 Data Quality

- **Missing Values**: Minimal (3.1250%) in raw data, completely handled in preprocessing
- **Feature Quality**: 30 informative features retained from 125 original sensors
- **Temporal Integrity**: Sliding window approach preserves temporal dependencies

### 6.2 Preprocessing Impact

1. **Dimensionality Reduction**: 76.9% reduction in feature space (125 to 30)
2. **Data Completeness**: 100% complete data after preprocessing
3. **Normalization**: All features scaled to [0, 1] range

### 6.3 Dataset Suitability

The preprocessed WADI dataset is well-suited for:

- LSTM-based anomaly detection (temporal sequences)
- Reconstruction-based methods (normalized features)
- Edge deployment (reduced dimensionality)

---

## References

- WADI Dataset: iTrust, Singapore University of Technology and Design
- Preprocessing Pipeline: `preprocessing/strict_anti_leakage_preprocessor.py`

---

*Report generated as part of thesis: Edge-Deployable LSTM Autoencoder for Real-Time SCADA Anomaly Detection*
