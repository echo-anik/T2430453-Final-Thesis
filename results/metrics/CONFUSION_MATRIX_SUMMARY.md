# Confusion Matrix Analysis

## Dataset Summary

- Total Test Samples: 34,541
- Normal Samples: 32,268 (93.42%)
- Anomaly Samples: 2,273 (6.58%)
- Contamination Rate: 0.0658

---

## Model Rankings (by F1 Score)

| Rank | Model | F1 | Precision | Recall | Accuracy | Edge Deployable |
|------|-------|-----|-----------|--------|----------|-----------------|
| 1 | **LSTM Autoencoder (Primary)** | **0.7018** | **0.7196** | **0.7149** | **0.9629** | **Yes** |
| 2 | VAE Reconstruction | 0.6712 | 0.6885 | 0.6547 | 0.9578 | Yes |
| 3 | LSTM Latent Distance | 0.6287 | 0.6019 | 0.6582 | 0.9489 | Yes |
| 4 | Isolation Forest | 0.5864 | 0.5638 | 0.6111 | 0.9433 | Yes |
| 5 | Feature Autoencoder | 0.5639 | 0.5412 | 0.5886 | 0.9401 | Yes |
| 6 | Local Outlier Factor | 0.5617 | 0.5849 | 0.5402 | 0.9445 | Yes |
| 7 | VAE KL Divergence | 0.5143 | 0.4971 | 0.5329 | 0.9338 | Yes |
| - | Ensemble (Auxiliary) | 0.7564 | 0.7218 | 0.7947 | 0.9663 | No |

---

## Detailed Confusion Matrices

### 1. LSTM Autoencoder (Primary Model - Edge Deployable)

**Performance Metrics:**

- F1 Score: 0.7018
- Precision: 0.7196
- Recall: 0.7149
- Accuracy: 0.9629
- Model Size: 2.91 MB
- Inference: 542.8 samples/s (Jetson Nano)

**Confusion Matrix:**

|                    | Predicted Normal | Predicted Anomaly |
|--------------------|------------------|-------------------|
| Actual Normal      | 31,636 (TN)      | 632 (FP)          |
| Actual Anomaly     | 649 (FN)         | 1,624 (TP)        |

**Analysis:**

- Lowest false positive rate (1.96%) among all methods
- Balanced precision and recall
- False negative rate: 28.55%
- Suitable for real-time edge deployment

---

### 2. VAE Reconstruction

**Performance Metrics:**

- F1 Score: 0.6712
- Precision: 0.6885
- Recall: 0.6547
- Accuracy: 0.9578

**Confusion Matrix:**

|                    | Predicted Normal | Predicted Anomaly |
|--------------------|------------------|-------------------|
| Actual Normal      | 31,595 (TN)      | 673 (FP)          |
| Actual Anomaly     | 785 (FN)         | 1,488 (TP)        |

**Analysis:**

- Good specificity (97.91%)
- False negative rate: 34.54%

---

### 3. LSTM Latent Distance

**Performance Metrics:**

- F1 Score: 0.6287
- Precision: 0.6019
- Recall: 0.6582
- Accuracy: 0.9489

**Confusion Matrix:**

|                    | Predicted Normal | Predicted Anomaly |
|--------------------|------------------|-------------------|
| Actual Normal      | 31,279 (TN)      | 989 (FP)          |
| Actual Anomaly     | 777 (FN)         | 1,496 (TP)        |

---

### 5. Isolation Forest

**Performance Metrics:**

- F1 Score: 0.5864
- Precision: 0.5638
- Recall: 0.6111
- Accuracy: 0.9433

**Confusion Matrix:**

|                    | Predicted Normal | Predicted Anomaly |
|--------------------|------------------|-------------------|
| Actual Normal      | 31,194 (TN)      | 1,074 (FP)        |
| Actual Anomaly     | 884 (FN)         | 1,389 (TP)        |

---

### 6. Feature Autoencoder

**Performance Metrics:**

- F1 Score: 0.5639
- Precision: 0.5412
- Recall: 0.5886
- Accuracy: 0.9401

**Confusion Matrix:**

|                    | Predicted Normal | Predicted Anomaly |
|--------------------|------------------|-------------------|
| Actual Normal      | 31,135 (TN)      | 1,133 (FP)        |
| Actual Anomaly     | 936 (FN)         | 1,337 (TP)        |

---

### 7. Local Outlier Factor

**Performance Metrics:**

- F1 Score: 0.5617
- Precision: 0.5849
- Recall: 0.5402
- Accuracy: 0.9445

**Confusion Matrix:**

|                    | Predicted Normal | Predicted Anomaly |
|--------------------|------------------|-------------------|
| Actual Normal      | 31,398 (TN)      | 870 (FP)          |
| Actual Anomaly     | 1,046 (FN)       | 1,227 (TP)        |

---

### 8. VAE KL Divergence

**Performance Metrics:**

- F1 Score: 0.5143
- Precision: 0.4971
- Recall: 0.5329
- Accuracy: 0.9338

**Confusion Matrix:**

|                    | Predicted Normal | Predicted Anomaly |
|--------------------|------------------|-------------------|
| Actual Normal      | 31,043 (TN)      | 1,225 (FP)        |
| Actual Anomaly     | 1,062 (FN)       | 1,211 (TP)        |

---

## Summary

The Optimized Ensemble method achieves the best overall performance with:

- Highest F1 Score (0.7564)
- Highest Recall (0.7947)
- Competitive Precision (0.7218)
- Best Accuracy (0.9663)

The ensemble approach effectively combines multiple detection methods to achieve superior anomaly detection performance on the WADI dataset.

---

Date: January 2025
