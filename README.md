# Unsupervised Anomaly Detection for Industrial Control Systems

**Unsupervised Ensemble Learning with Edge Deployment Simulation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)  
[![TensorRT](https://img.shields.io/badge/TensorRT-8.0+-green.svg)](https://developer.nvidia.com/tensorrt)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Abstract

This thesis presents a **truly unsupervised, ensemble-based framework** for anomaly detection in Industrial Control Systems (ICS), specifically targeting water distribution infrastructure (WADI dataset). The system achieves **F1 = 0.7564**, significantly outperforming published state-of-the-art methods, without requiring any labeled anomaly data during training. The framework includes edge deployment simulation for Jetson Nano, demonstrating real-time inference capabilities for critical infrastructure protection.

---

## 1. Key Results

### 1.1 Detection Performance (Unsupervised)

| Method | F1 Score | Precision | Recall |
|--------|----------|-----------|--------|
| **Equal-Weight Ensemble (Ours)** | **0.7564** | **0.7218** | **0.7947** |
| LSTM Reconstruction | 0.7018 | 0.7196 | 0.7149 |
| VAE Reconstruction | 0.6712 | 0.6885 | 0.6547 |
| LSTM Latent Distance | 0.6287 | 0.6019 | 0.6582 |
| Isolation Forest | 0.5864 | 0.5638 | 0.6111 |
| Feature Autoencoder | 0.5639 | 0.5412 | 0.5886 |
| Local Outlier Factor | 0.5617 | 0.5849 | 0.5402 |
| VAE KL Divergence | 0.5143 | 0.4971 | 0.5329 |

### 1.2 Comparison with Published State-of-the-Art

Our truly unsupervised ensemble significantly outperforms published baselines:

| Method | F1 Score | Year | Source |
|--------|----------|------|--------|
| **Our Ensemble** | **0.7564** | 2026 | This work |
| STADN | 0.62 | 2023 | Tang et al. |
| GDN | 0.58 | 2021 | Deng et al. |
| TranAD | 0.55 | 2022 | Tuli et al. |
| LSTM-VAE | 0.53 | 2022 | Faber et al. |
| USAD | 0.52 | 2020 | Audibert et al. |
| OmniAnomaly | 0.47 | 2019 | Su et al. |
| MAD-GAN | 0.45 | 2019 | Li et al. |
| DAGMM | 0.39 | 2018 | Zong et al. |

**Key Achievement**: +22.0% F1 score improvement over previous best (STADN: 0.62)

### 1.3 Edge Deployment Performance (Jetson Nano Simulation)

Performance metrics from edge deployment simulation:

| Model | Inference Time (ms) | Throughput (samples/s) | Power Draw (W) | Model Size (MB) |
|-------|---------------------|------------------------|----------------|-----------------|
| LSTM-AE (FP32) | 1.84 | 542.8 | 5.38 | 2.91 |
| LSTM-AE (FP16) | 1.69 | 593.2 | 4.73 | 1.46 |
| Feature AE (FP32) | 0.82 | 1212.6 | 5.09 | 0.72 |
| Feature AE (FP16) | 0.85 | 1178.4 | 4.56 | 0.36 |

**Real-time Capability**: All models achieve >100 samples/s, meeting industrial requirements

---

## 2. Introduction

Industrial Control Systems (ICS) protecting critical infrastructure are increasingly vulnerable to cyber-physical attacks. Detecting anomalies in real-time is challenging due to:

- **High-dimensional sensor data** with complex temporal dependencies
- **Scarcity of labeled attack data** in operational environments
- **Need for edge deployment** with limited computational resources
- **Requirement for interpretability** in safety-critical systems

This work addresses these challenges through a truly unsupervised ensemble approach that:
- Requires **no labeled anomaly data** for training  
- Achieves **F1 = 0.7564**, outperforming all published baselines  
- Enables **real-time edge inference** on Jetson Nano  
- Provides **interpretable anomaly scores** from multiple detectors  

---
## 3. Methodology

This chapter presents the proposed framework for unsupervised anomaly detection in ICS. The methodology consists of three main phases: temporal-statistical feature engineering, latent representation learning, and ensemble-based anomaly detection.

**Related Documentation:** For detailed explanation of the complete data transformation pipeline from raw samples to temporal windows, see [Data Transformation Pipeline Documentation](results/thesis_visuals/DATA_TRANSFORMATION_EXPLAINED.md).

---

### 3.1 Data Preprocessing and Windowing

**Preprocessing Pipeline:** The WADI dataset undergoes systematic transformation from raw sensor measurements to temporal windows. For comprehensive details on each preprocessing stage, refer to the [Data Transformation Pipeline](results/thesis_visuals/DATA_TRANSFORMATION_EXPLAINED.md).

**Pipeline Overview:**

- Raw sensor data from WADI dataset (784,571 training samples × 130 features)
- Feature selection: 130 → 30 features (K-S test, variance filtering, constant removal)
- Temporal windowing: samples segmented into **windows of 100 timesteps**
- Each window contains **30 sensor readings**, forming a 100×30 matrix → **3,000 raw features per window**
- Normal operation windows used for training; test windows include attack scenarios
- Final dataset: 141,206 training windows, 15,689 validation windows, 34,541 test windows

---

### 3.2 Temporal-Statistical Feature Engineering

To improve interpretability and reduce dimensionality, the following features are computed from each window:

1. **Rolling Statistics**  
   - Mean: \(\mu_i = \frac{1}{T} \sum_{t=1}^{T} x_{i,t}\)  
   - Standard Deviation: \(\sigma_i = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (x_{i,t}-\mu_i)^2}\)  
   - Range: \(R_i = \max(x_{i,:}) - \min(x_{i,:})\)

2. **Rate-of-Change and Slopes**  
   - Slope: \(Slope_i = \frac{\mu_{i,t > T/2} - \mu_{i,t < T/2}}{0.75 T}\)  
   - Average Rate-of-Change: \(\Delta Rate_i = \frac{1}{T-1} \sum_{t=2}^{T} |x_{i,t} - x_{i,t-1}|\)

3. **Cross-Sensor Correlation**  
   - Captures physical dependencies between sensors:  
     \(\rho_{i,j} = \frac{\sum (x_i - \mu_i)(x_j - \mu_j)}{\sqrt{\sum (x_i - \mu_i)^2}\sqrt{\sum (x_j - \mu_j)^2}}\)

4. **Zero-Crossing Rate**  
   - Frequency content analysis:  
     \(ZCR_i = \frac{1}{T} \sum_{t=2}^{T} \mathbf{1}[sign(x_{i,t}-\mu_i) \neq sign(x_{i,t-1}-\mu_i)]\)

- **Output**: 276 engineered features per window (~11× dimensionality reduction).

---

### 3.3 Latent Representation Learning

- Train an **LSTM Autoencoder** on normal windows to learn a latent representation of normal system behavior.  
- Extract anomaly signals from:
  - **Reconstruction Error (MSE)**  
  - **Latent Distance** from normal manifold center

- These signals capture deviations from learned normal patterns.

---

### 3.4 Ensemble-Based Anomaly Detection

- Deploy multiple **unsupervised anomaly detectors** on engineered features and learned representations:
  - **LSTM Autoencoder** (reconstruction error)  
  - **Variational Autoencoder (VAE)** (reconstruction + KL divergence)  
  - **Isolation Forest** (tree-based outlier detection)  
  - **Local Outlier Factor (LOF)** (density-based detection)  
  - **Feature Autoencoder** (dimensionality-reduced reconstruction)  
  - **LSTM Latent Distance** (manifold deviation)

- Each detector produces anomaly scores on test data  
- Combine predictions via **equal-weight averaging** to produce final anomaly score per window  
- Apply threshold optimization on validation set to convert scores to binary predictions

---

### 3.5 Implementation Notes

- Training is fully unsupervised: only normal operation windows are used.  
- Dimensionality reduction and feature engineering enable **real-time deployment** on edge devices.  
- Hyperparameters (LSTM size, ensemble tree depth) are optimized using **validation normal windows** only.

## 4. Latent Representation Learning

- LSTM autoencoder trained on normal operation windows  
- Two anomaly signals extracted:
  - **Reconstruction error** (MSE)
  - **Latent distance** from normal manifold center  

Signals are later fused with engineered features.

---

## 5. Ensemble-Based Detection

- Deploy multiple unsupervised detectors on engineered features (276 dims) and latent signals (2 dims)  
- Individual detectors:
  - **LSTM Autoencoder** reconstruction error  
  - **VAE** reconstruction + KL divergence  
  - **Isolation Forest** outlier scoring  
  - **Local Outlier Factor (LOF)** density-based scoring  
  - **Feature Autoencoder** compressed reconstruction  
  - **LSTM Latent Distance** from normal manifold  
- Ensemble via **equal-weight averaging** of anomaly scores  
- **No labeled anomaly data required** during training

---

## 6. Dataset

- **WADI (Water Distribution) Dataset**
  - 14 days normal operation for training
  - 2 days with 15 attack scenarios for testing
  - 30 selected sensors (after K-S test filtering)
- Training samples: 141,206  
- Testing samples: 34,541  
- Contamination rate: 6.58%

---

## 7. Experimental Results

### 7.1 Unsupervised Performance (Main Contribution)

| Method | F1 | Precision | Recall | Notes |
|--------|----|-----------|--------|------|
| Optimized Ensemble (Ours) | **0.7564** | 0.7218 | 0.7947 | Primary contribution |
| LSTM Reconstruction (Ours) | 0.7018 | 0.7196 | 0.7149 | Single model |
| STADN (Tang 2023) | 0.62 | 0.58 | 0.67 | Literature |
| GDN (Deng 2021) | 0.58 | 0.54 | 0.63 | Literature |
| TranAD (Tuli 2022) | 0.55 | 0.51 | 0.57 | Literature |
| LSTM-VAE (Faber 2022) | 0.53 | 0.48 | 0.59 | Literature |

**Key Takeaways:**

- Ensemble methods outperform individual unsupervised detectors  
- Feature engineering improves unsupervised F1 from 0.41 → 0.76  
- Fully unsupervised approach is **deployable and realistic**  

---

### 7.2 Semi-Supervised Upper-Bound (Auxiliary)

> Included for completeness; semi-supervised requires labeled attack data and is not considered realistic for ICS deployment.

| Method | F1 | Precision | Recall | Notes |
|--------|----|-----------|--------|------|
| Semi-Supervised Ensemble (5% labeled) | 0.8765 | 0.9456 | 0.8167 | Upper bound |

*Figure: ROC / PR curves can be included here if desired.*

---

## 8. Discussion

- **Unsupervised ensemble**: Robust, interpretable, high performance  
- **Single LSTM-AE**: Strong baseline, slightly lower F1  
- **Semi-supervised**: Upper bound; unrealistic without labeled attacks  
- **Feature engineering**: Crucial for dimensionality reduction and interpretability  

---

## 9. Limitations

- Still assumes normal training data is fully attack-free  
- Performance may degrade under unseen attack types with unusual patterns  
- Future work could incorporate online adaptation for concept drift  

---

## 10. Conclusion

- Feature-centric, unsupervised detection provides **high F1, low computation, and deployable models**  
- Semi-supervised results demonstrate upper-bound performance but are **not practical in real-world ICS**  
- Method achieves **significant improvement over SOTA unsupervised baselines**

---

## 11. Installation & Usage

```bash
# Clone repository
git clone https://github.com/echo-anik/THESIS-P3.git
cd THESIS

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run evaluation
python scripts/final_evaluation.py
python scripts/generate_visualizations.py
