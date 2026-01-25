# Unsupervised Anomaly Detection for Industrial Control Systems

## Edge-Deployable LSTM Autoencoder for Real-Time SCADA Anomaly Detection

---

## Abstract

This thesis presents an unsupervised LSTM Autoencoder framework for anomaly detection in Industrial Control Systems (ICS), specifically targeting water distribution infrastructure using the WADI dataset. The proposed single-model approach achieves an F1 score of 0.7018, outperforming published state-of-the-art methods while maintaining edge deployability on resource-constrained devices. The model is validated for deployment on Jetson Nano, achieving real-time inference at 542.8 samples/second with only 2.91 MB model size, demonstrating practical viability for critical infrastructure protection.

---

## 1. Key Results

### 1.1 Primary Contribution: Edge-Deployable LSTM Autoencoder

| Metric | Value |
|--------|-------|
| **F1 Score** | **0.7018** |
| Precision | 0.7196 |
| Recall | 0.7149 |
| Model Size | 2.91 MB (FP32) / 1.46 MB (FP16) |
| Inference Time | 1.84 ms (Jetson Nano) |
| Throughput | 542.8 samples/s |

### 1.2 Detection Performance Comparison

| Method | F1 Score | Precision | Recall | Edge Deployable |
|--------|----------|-----------|--------|-----------------|
| **LSTM Autoencoder (Ours)** | **0.7018** | **0.7196** | **0.7149** | **Yes** |
| VAE Reconstruction | 0.6712 | 0.6885 | 0.6547 | Yes |
| LSTM Latent Distance | 0.6287 | 0.6019 | 0.6582 | Yes |
| Isolation Forest | 0.5864 | 0.5638 | 0.6111 | Yes |
| Feature Autoencoder | 0.5639 | 0.5412 | 0.5886 | Yes |
| Local Outlier Factor | 0.5617 | 0.5849 | 0.5402 | Yes |
| VAE KL Divergence | 0.5143 | 0.4971 | 0.5329 | Yes |
| Ensemble (auxiliary) | 0.7564 | 0.7218 | 0.7947 | No (multi-model) |

### 1.3 Comparison with Published Methods

| Method | F1 Score | Year | Edge Deployable | Reference |
|--------|----------|------|-----------------|-----------|
| **LSTM Autoencoder (Ours)** | **0.7018** | 2025 | **Yes** | This work |
| STADN | 0.62 | 2023 | Limited | Tang et al. |
| GDN | 0.58 | 2021 | No | Deng et al. |
| TranAD | 0.55 | 2022 | No | Tuli et al. |
| LSTM-VAE | 0.53 | 2022 | Limited | Faber et al. |
| USAD | 0.52 | 2020 | No | Audibert et al. |
| OmniAnomaly | 0.47 | 2019 | No | Su et al. |
| MAD-GAN | 0.45 | 2019 | No | Li et al. |
| DAGMM | 0.39 | 2018 | No | Zong et al. |

**Performance Improvement**: +13.2% F1 score over previous best (STADN: 0.62), while maintaining edge deployment capability

### 1.4 Edge Deployment Performance (Jetson Nano)

| Model | Inference Time (ms) | Throughput (samples/s) | Power Draw (W) | Model Size (MB) |
|-------|---------------------|------------------------|----------------|-----------------|
| **LSTM-AE (FP32)** | **1.84** | **542.8** | **5.38** | **2.91** |
| LSTM-AE (FP16) | 1.69 | 593.2 | 4.73 | 1.46 |
| Feature AE (FP32) | 0.82 | 1212.6 | 5.09 | 0.72 |
| Feature AE (FP16) | 0.85 | 1178.4 | 4.56 | 0.36 |

The LSTM Autoencoder achieves 542.8 samples/s throughput on Jetson Nano, exceeding the 100 samples/s threshold for industrial real-time requirements by 5.4x.

---

## 2. Introduction

Industrial Control Systems protecting critical infrastructure are increasingly vulnerable to cyber-physical attacks. Detecting anomalies in real-time presents several challenges:

- High-dimensional sensor data with complex temporal dependencies
- Scarcity of labeled attack data in operational environments
- Edge deployment constraints with limited computational resources
- Requirements for interpretability in safety-critical systems

This work addresses these challenges through an edge-deployable LSTM Autoencoder that:

- Requires no labeled anomaly data for training
- Achieves F1 = 0.7018 on WADI dataset, outperforming published baselines
- Enables real-time edge inference on Jetson Nano (542.8 samples/s)
- Maintains a compact 2.91 MB model footprint suitable for embedded systems

---

## 3. Methodology

### 3.1 Data Preprocessing and Windowing

The WADI dataset undergoes systematic transformation from raw sensor measurements to temporal windows. For detailed documentation of the preprocessing pipeline, refer to:

- [Data Transformation Pipeline](results/thesis_visuals/DATA_TRANSFORMATION_EXPLAINED.md)

**Pipeline Summary:**

- Raw sensor data: 784,571 training samples, 130 features
- Feature selection: 130 to 30 features (K-S test, variance filtering)
- Temporal windowing: 100 timesteps per window
- Final dataset: 141,206 training windows, 34,541 test windows

### 3.2 Temporal-Statistical Feature Engineering

Statistical features computed from each temporal window:

1. **Rolling Statistics**: Mean, standard deviation, range
2. **Rate-of-Change**: Slope, average rate-of-change
3. **Cross-Sensor Correlation**: Inter-sensor dependencies
4. **Zero-Crossing Rate**: Frequency content analysis

Output: 276 engineered features per window

### 3.3 LSTM Autoencoder Architecture

The LSTM Autoencoder is trained on normal operation windows to learn compact representations of typical system behavior:

**Encoder:**
- Input: 100 timesteps x 30 features
- LSTM layers with dropout for regularization
- Bottleneck latent representation

**Decoder:**
- Symmetric LSTM architecture
- Reconstruction of input sequence

**Anomaly Detection:**
- Reconstruction error (MSE) as primary anomaly score
- Threshold determined via validation set optimization

### 3.4 Auxiliary Methods (For Comparison)

Additional unsupervised detectors evaluated for comparison:

- Variational Autoencoder (reconstruction + KL divergence)
- Isolation Forest (tree-based outlier detection)
- Local Outlier Factor (density-based detection)
- Feature Autoencoder (dimensionality-reduced reconstruction)
- LSTM Latent Distance (manifold deviation)

Ensemble combination provided as auxiliary result (F1: 0.7564) but is not the primary contribution due to edge deployment constraints.

---

## 4. Dataset

**WADI (Water Distribution) Dataset:**

- Training: 14 days normal operation (141,206 windows)
- Testing: 2 days with 15 attack scenarios (34,541 windows)
- Selected features: 30 sensors (after statistical filtering)
- Contamination rate: 6.58%

---

## 5. Experimental Results

### 5.1 Performance Summary

| Method | F1 | Precision | Recall | Edge Deployable |
|--------|-----|-----------|--------|-----------------|
| **LSTM Autoencoder (Primary)** | **0.7018** | **0.7196** | **0.7149** | **Yes** |
| STADN (Literature) | 0.62 | 0.58 | 0.67 | Limited |
| GDN (Literature) | 0.58 | 0.54 | 0.63 | No |
| Ensemble (Auxiliary) | 0.7564 | 0.7218 | 0.7947 | No |

### 5.2 Statistical Validation

Results validated across 10 experimental runs for the LSTM Autoencoder:

- Mean F1: 0.6988 (SD: 0.0124)
- 95% Confidence Interval: [0.6899, 0.7077]
- Coefficient of Variation: 1.77%

Statistical significance confirmed via paired t-tests (p < 0.01 for all comparisons with baseline methods).

For detailed statistical analysis, refer to:

- [Statistical Analysis Report](results/thesis_visuals/statistical_analysis/STATISTICAL_ANALYSIS_REPORT.md)

### 5.3 Hyperparameter Selection

Systematic ablation study conducted for hyperparameter optimization:

| Parameter | Selected Value | Justification |
|-----------|----------------|---------------|
| Epochs | 100 | Full convergence without overfitting |
| Window Size | 100 | Optimal temporal context |
| Learning Rate | 0.0001 | Best final performance |
| Batch Size | 128 | Optimal gradient quality |

For complete ablation study results, refer to:

- [Ablation Study Report](results/thesis_visuals/ablation_study/ABLATION_STUDY_REPORT.md)

---

## 6. Documentation

### Reports and Analysis

| Document | Description |
|----------|-------------|
| [Exploratory Data Analysis](results/thesis_visuals/eda/EDA_REPORT.md) | Dataset statistics, missing values, correlations |
| [Data Transformation Pipeline](results/thesis_visuals/DATA_TRANSFORMATION_EXPLAINED.md) | Preprocessing methodology |
| [Ablation Study Report](results/thesis_visuals/ablation_study/ABLATION_STUDY_REPORT.md) | Hyperparameter optimization |
| [Statistical Analysis Report](results/thesis_visuals/statistical_analysis/STATISTICAL_ANALYSIS_REPORT.md) | Multi-run validation |
| [Confusion Matrix Analysis](results/metrics/CONFUSION_MATRIX_SUMMARY.md) | Detailed model evaluation |
| [Literature Baselines](docs/literature_baselines.md) | Published method comparison |

### Generated Visualizations

All thesis-quality figures are available in `results/thesis_visuals/`:

- EDA visualizations (correlation matrices, missing values, statistics)
- Confusion matrices for all methods
- Performance comparison charts
- Ablation study visualizations
- Statistical analysis plots

---

## 7. Repository Structure

```
THESIS/
├── data/
│   └── preprocessed/          # Preprocessed windows and labels
├── datasets/                  # Raw WADI dataset files
├── docs/
│   └── literature_baselines.md
├── preprocessing/
│   └── strict_anti_leakage_preprocessor.py
├── results/
│   ├── metrics/               # Evaluation results
│   └── thesis_visuals/        # Generated figures
├── scripts/                   # Analysis and visualization scripts
├── README.md
└── requirements.txt
```

---

## 8. Installation

```bash
# Clone repository
git clone https://github.com/echo-anik/T2430453-Final-Thesis.git
cd THESIS

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## 9. Conclusion

This work demonstrates that a single LSTM Autoencoder achieves competitive anomaly detection performance (F1: 0.7018) while maintaining edge deployment capability. The proposed approach:

- **Outperforms published baselines** by 13.2% F1 improvement over STADN (0.62)
- **Enables real-time edge deployment** with 542.8 samples/s throughput on Jetson Nano
- **Maintains compact footprint** with 2.91 MB model size (1.46 MB with FP16 quantization)
- **Requires no labeled anomaly data** through unsupervised reconstruction-based detection

The practical contribution lies in achieving state-of-the-art detection accuracy while preserving deployability on resource-constrained industrial edge devices.

---

## References

See [Literature Baselines](docs/literature_baselines.md) for complete reference list.

---

*Project: T2430453 Final Thesis*  
*Date: January 2025*
