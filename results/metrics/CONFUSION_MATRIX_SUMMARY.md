# Confusion Matrix Analysis - All Models

## Dataset Summary
- **Total Test Samples**: 34,541
- **Normal Samples**: 32,268 (93.42%)
- **Anomaly Samples**: 2,273 (6.58%)
- **Contamination Rate**: 0.0658

---

## Model Rankings (by F1-Score)

| Rank | Model | F1 | Precision | Recall | Accuracy |
|------|-------|-----|-----------|--------|----------|
| ★ 1 | **Optimized Ensemble** | **0.7564** | **0.7218** | **0.7947** | **0.9663** |
| 2 | LSTM Reconstruction | 0.7018 | 0.7196 | 0.7149 | 0.9629 |
| 3 | VAE Reconstruction | 0.6712 | 0.6885 | 0.6547 | 0.9578 |
| 4 | LSTM Latent Distance | 0.6287 | 0.6019 | 0.6582 | 0.9489 |
| 5 | Isolation Forest | 0.5864 | 0.5638 | 0.6111 | 0.9433 |
| 6 | Feature Autoencoder | 0.5639 | 0.5412 | 0.5886 | 0.9401 |
| 7 | Local Outlier Factor | 0.5617 | 0.5849 | 0.5402 | 0.9445 |
| 8 | VAE KL Divergence | 0.5143 | 0.4971 | 0.5329 | 0.9338 |

---

## Detailed Confusion Matrices

### 1. Optimized Ensemble (Best Model) ★

**Performance Metrics:**
- F1-Score: 0.7564
- Precision: 0.7218
- Recall: 0.7947
- Accuracy: 0.9663

**Confusion Matrix:**

|                | Predicted Normal | Predicted Anomaly |
|----------------|------------------|-------------------|
| **Actual Normal** | 31,572 (TN) | 696 (FP) |
| **Actual Anomaly** | 467 (FN) | 1,806 (TP) |

**Key Insights:**
- ✅ **Highest recall (79.47%)** - Catches most anomalies
- ✅ **Low false positive rate (2.16%)** - Minimal false alarms
- ✅ **Best overall performance** with F1 of 0.7564
- ⚠️ **Misses 467 anomalies** (20.55% false negative rate)

---

### 2. LSTM Reconstruction

**Performance Metrics:**
- F1-Score: 0.7018
- Precision: 0.7196
- Recall: 0.7149
- Accuracy: 0.9629

**Confusion Matrix:**

|                | Predicted Normal | Predicted Anomaly |
|----------------|------------------|-------------------|
| **Actual Normal** | 31,636 (TN) | 632 (FP) |
| **Actual Anomaly** | 649 (FN) | 1,624 (TP) |

**Key Insights:**
- ✅ **Very low FP rate (1.96%)** - Best specificity
- ✅ **Balanced precision and recall** (~72% each)
- ⚠️ **Misses 649 anomalies** (28.55% FN rate)

---

### 3. VAE Reconstruction

**Performance Metrics:**
- F1-Score: 0.6712
- Precision: 0.6885
- Recall: 0.6547
- Accuracy: 0.9578

**Confusion Matrix:**

|                | Predicted Normal | Predicted Anomaly |
|----------------|------------------|-------------------|
| **Actual Normal** | 31,595 (TN) | 673 (FP) |
| **Actual Anomaly** | 785 (FN) | 1,488 (TP) |

**Key Insights:**
- ✅ **Good specificity (97.91%)**
- ⚠️ **Misses 785 anomalies** (34.54% FN rate)
- 📊 **Decent balance** between precision and recall

---

### 4. LSTM Latent Distance

**Performance Metrics:**
- F1-Score: 0.6287
- Precision: 0.6019
- Recall: 0.6582
- Accuracy: 0.9489

**Confusion Matrix:**

|                | Predicted Normal | Predicted Anomaly |
|----------------|------------------|-------------------|
| **Actual Normal** | 31,279 (TN) | 989 (FP) |
| **Actual Anomaly** | 777 (FN) | 1,496 (TP) |

**Key Insights:**
- ⚠️ **Higher false positives (989)** than top models
- ⚠️ **Lower precision (60.19%)**
- 📊 **Moderate recall (65.82%)**

---

### 5. Isolation Forest

**Performance Metrics:**
- F1-Score: 0.5864
- Precision: 0.5638
- Recall: 0.6111
- Accuracy: 0.9433

**Confusion Matrix:**

|                | Predicted Normal | Predicted Anomaly |
|----------------|------------------|-------------------|
| **Actual Normal** | 31,194 (TN) | 1,074 (FP) |
| **Actual Anomaly** | 884 (FN) | 1,389 (TP) |

**Key Insights:**
- ⚠️ **High false positives (1,074)** - 3.33% FPR
- ⚠️ **Misses 884 anomalies** (38.89% FN rate)
- 📊 **Traditional ML approach** performs moderately

---

### 6. Feature Autoencoder

**Performance Metrics:**
- F1-Score: 0.5639
- Precision: 0.5412
- Recall: 0.5886
- Accuracy: 0.9401

**Confusion Matrix:**

|                | Predicted Normal | Predicted Anomaly |
|----------------|------------------|-------------------|
| **Actual Normal** | 31,135 (TN) | 1,133 (FP) |
| **Actual Anomaly** | 936 (FN) | 1,337 (TP) |

**Key Insights:**
- ⚠️ **Highest false positives (1,133)** among autoencoders
- ⚠️ **Lowest precision (54.12%)**
- ⚠️ **Misses 936 anomalies** (41.18% FN rate)

---

### 7. Local Outlier Factor

**Performance Metrics:**
- F1-Score: 0.5617
- Precision: 0.5849
- Recall: 0.5402
- Accuracy: 0.9445

**Confusion Matrix:**

|                | Predicted Normal | Predicted Anomaly |
|----------------|------------------|-------------------|
| **Actual Normal** | 31,398 (TN) | 870 (FP) |
| **Actual Anomaly** | 1,046 (FN) | 1,227 (TP) |

**Key Insights:**
- ⚠️ **Lowest recall (54.02%)** - Misses many anomalies
- ⚠️ **Highest false negatives (1,046)** - 46.02% FN rate
- ✅ **Good specificity (97.30%)**

---

### 8. VAE KL Divergence

**Performance Metrics:**
- F1-Score: 0.5143
- Precision: 0.4971
- Recall: 0.5329
- Accuracy: 0.9338

**Confusion Matrix:**

|                | Predicted Normal | Predicted Anomaly |
|----------------|------------------|-------------------|
| **Actual Normal** | 31,043 (TN) | 1,225 (FP) |
| **Actual Anomaly** | 1,062 (FN) | 1,211 (TP) |

**Key Insights:**
- ⚠️ **Worst overall performance** (F1: 0.5143)
- ⚠️ **Highest FP rate (3.80%)**
- ⚠️ **Lowest precision (49.71%)**
- ⚠️ **High false negatives (1,062)** - 46.72% FN rate

---

## Key Findings

### Best Models:
1. **Optimized Ensemble** - Best overall with F1=0.7564
   - Highest recall (79.47%)
   - Best anomaly detection rate
   - Only 2.16% false positive rate

2. **LSTM Reconstruction** - Strong runner-up with F1=0.7018
   - Best specificity (98.04%)
   - Lowest false positive rate (1.96%)
   - Excellent for minimizing false alarms

3. **VAE Reconstruction** - Solid third with F1=0.6712
   - Good balance of metrics
   - Reliable performance

### Model Performance Patterns:

**Deep Learning Models** (Top performers):
- LSTM Reconstruction: F1=0.7018
- VAE Reconstruction: F1=0.6712
- LSTM Latent Distance: F1=0.6287

**Traditional ML Models** (Moderate performance):
- Isolation Forest: F1=0.5864
- Local Outlier Factor: F1=0.5617

**Single Metric Models** (Lower performance):
- VAE KL Divergence: F1=0.5143 (worst)
- Feature Autoencoder: F1=0.5639

### Trade-offs:
- **High Recall Models** (catch more anomalies): Optimized Ensemble, LSTM Reconstruction
- **High Precision Models** (fewer false alarms): LSTM Reconstruction, Optimized Ensemble
- **Best Balance**: Optimized Ensemble achieves both high precision AND recall

---

## Files Generated

All confusion matrix visualizations and detailed reports are saved in:
📁 `results/metrics/confusion_matrices/`

- Individual confusion matrix plots for each model
- Comprehensive visualization with all models
- Detailed text report with all metrics
- JSON file with raw confusion matrix data

---

## Recommendations

1. **For Production Use**: Deploy **Optimized Ensemble**
   - Best F1-score (0.7564)
   - Highest anomaly detection rate (79.47% recall)
   - Acceptable false positive rate (2.16%)

2. **For Critical Systems** (where false alarms are costly): Use **LSTM Reconstruction**
   - Lowest false positive rate (1.96%)
   - Still maintains good recall (71.49%)

3. **Ensemble Strategy**: The fact that the ensemble outperforms individual models validates the multi-model approach

4. **Avoid**: VAE KL Divergence and Feature Autoencoder as standalone models
   - Poor precision (<55%)
   - High false positive rates
   - Better suited as ensemble components only
