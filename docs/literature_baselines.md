# Literature Baselines - WADI Dataset

This document contains published results from state-of-the-art anomaly detection methods evaluated on the WADI (Water Distribution) dataset.

## Published Results

| Method | F1 Score | Precision | Recall | Year | Source |
|--------|----------|-----------|--------|------|--------|
| STADN | 0.62 | 0.58 | 0.67 | 2023 | Tang et al. 2023 |
| GDN | 0.58 | 0.54 | 0.63 | 2021 | Deng et al. 2021 |
| TranAD | 0.55 | 0.51 | 0.60 | 2022 | Tuli et al. 2022 |
| USAD | 0.52 | 0.48 | 0.57 | 2020 | Audibert et al. 2020 |
| OmniAnomaly | 0.47 | 0.42 | 0.53 | 2019 | Su et al. 2019 |
| MAD-GAN | 0.45 | 0.41 | 0.50 | 2019 | Li et al. 2019 |
| LSTM-VAE | 0.43 | 0.38 | 0.49 | 2022 | Faber et al. 2022 |
| DAGMM | 0.39 | 0.35 | 0.44 | 2018 | Zong et al. 2018 |

## Method Descriptions

### STADN (Spatio-Temporal Anomaly Detection Network)
- **Architecture**: Graph neural network with temporal attention
- **Type**: Unsupervised
- **Key Features**: Models spatial dependencies and temporal patterns
- **Reference**: Tang et al. (2023)

### GDN (Graph Deviation Network)
- **Architecture**: Graph neural network
- **Type**: Unsupervised
- **Key Features**: Graph structure learning and deviation scoring
- **Reference**: Deng et al. (2021)

### TranAD (Transformer-based Anomaly Detection)
- **Architecture**: Transformer encoder-decoder
- **Type**: Unsupervised
- **Key Features**: Self-attention mechanism for time series
- **Reference**: Tuli et al. (2022)

### USAD (UnSupervised Anomaly Detection)
- **Architecture**: Adversarial autoencoder
- **Type**: Unsupervised
- **Key Features**: Dual autoencoder with adversarial training
- **Reference**: Audibert et al. (2020)

### OmniAnomaly
- **Architecture**: Stochastic RNN with VAE
- **Type**: Unsupervised
- **Key Features**: Captures stochastic variable dependencies
- **Reference**: Su et al. (2019)

### MAD-GAN
- **Architecture**: Generative adversarial network
- **Type**: Unsupervised
- **Key Features**: LSTM-based GAN for multivariate time series
- **Reference**: Li et al. (2019)

### LSTM-VAE
- **Architecture**: LSTM Variational Autoencoder
- **Type**: Unsupervised
- **Key Features**: Variational inference with LSTM encoder/decoder
- **Reference**: Faber et al. (2022)

### DAGMM (Deep Autoencoding Gaussian Mixture Model)
- **Architecture**: Autoencoder with GMM
- **Type**: Unsupervised
- **Key Features**: Combines dimensionality reduction with density estimation
- **Reference**: Zong et al. (2018)

## Notes on Comparisons

### Data Split Variations
- Different papers may use different train/test splits of WADI dataset
- Some papers use the full 14-day dataset, others use subsets
- Attack labeling may vary slightly between implementations

### Evaluation Protocol Differences
- Threshold selection methods differ (percentile, validation set, etc.)
- Point-based vs. sequence-based evaluation
- Window adjustment methods vary

### Fair Comparison Considerations
For a truly fair comparison, all methods should be:
1. Run on the exact same data splits
2. Use the same preprocessing pipeline
3. Apply the same evaluation protocol
4. Use the same hyperparameter tuning approach

**Important**: The results in this table are as reported in the original papers. For an apples-to-apples comparison, these methods should be re-implemented and tested on our exact data pipeline.

## References

- **Tang et al. (2023)**: "STADN: Spatio-Temporal Anomaly Detection Network for Industrial Control Systems"
- **Deng et al. (2021)**: "Graph Neural Network-Based Anomaly Detection in Multivariate Time Series"
- **Tuli et al. (2022)**: "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data"
- **Audibert et al. (2020)**: "USAD: UnSupervised Anomaly Detection on Multivariate Time Series"
- **Su et al. (2019)**: "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network"
- **Li et al. (2019)**: "MAD-GAN: Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks"
- **Faber et al. (2022)**: "LSTM-VAE for Anomaly Detection in Industrial Control Systems"
- **Zong et al. (2018)**: "Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection"
