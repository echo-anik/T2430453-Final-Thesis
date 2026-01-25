# Visualization Scripts

This directory contains scripts for generating thesis-quality visualizations and analysis.

---

## Scripts Overview

### 1. exploratory_data_analysis.py

**Purpose**: Generate comparative analysis of raw versus preprocessed data.

**Output Files**:

- `thesis_data_quality.png` - Data transformation overview
- `thesis_distribution.png` - Distribution comparisons
- `thesis_timeseries.png` - Time series visualization
- `thesis_outlier_reduction.png` - Outlier analysis
- `thesis_statistics.png` - Statistical properties

**Usage**:

```bash
python scripts/exploratory_data_analysis.py
```

---

### 2. visualize_training_history.py

**Purpose**: Generate training loss visualizations.

**Output Files**:

- `thesis_training_loss.png` - Training and validation loss curves
- `thesis_loss_smoothed.png` - Smoothed loss curves
- `thesis_model_comparison.png` - Multi-model comparison

**Usage**:

```bash
python scripts/visualize_training_history.py
```

---

### 3. train_for_visualizations_pytorch.py

**Purpose**: Train LSTM autoencoder and generate training history.

**Output Files**:

- `training_history_lstm.json` - Training history data
- `thesis_actual_training_loss.png` - Training curves
- `thesis_actual_loss_smoothed.png` - Smoothed curves
- `lstm_autoencoder_best.pth` - Model weights

**Configuration**:

- Model: LSTM Autoencoder (64-32 encoder, 32-64 decoder)
- Training samples: 50,000
- Validation samples: 10,000
- Maximum epochs: 30 (with early stopping)
- Batch size: 256

**Usage**:

```bash
python scripts/train_for_visualizations_pytorch.py
```

---

### 4. hyperparameter_ablation_study.py

**Purpose**: Generate ablation study visualizations for hyperparameter justification.

**Output Directory**: `results/thesis_visuals/ablation_study/`

**Output Files**:

- `epoch_comparison_validation.png` - Validation loss across epochs
- `epoch_comparison_bars.png` - Final loss comparison
- `window_size_comparison.png` - Window size analysis
- `learning_rate_comparison.png` - Learning rate comparison
- `batch_size_comparison.png` - Batch size analysis
- `ABLATION_STUDY_REPORT.md` - Summary report

**Configurations Evaluated**:

| Parameter | Values Tested | Optimal |
|-----------|---------------|---------|
| Epochs | 30, 50, 75, 100, 125, 150, 175, 200, 250 | 100 |
| Window Size | 25, 50, 75, 100, 150, 200 | 100 |
| Learning Rate | 0.0001, 0.001, 0.01 | 0.0001 |
| Batch Size | 64, 128, 256, 512 | 128 |

**Usage**:

```bash
python scripts/hyperparameter_ablation_study.py
```

---

### 5. multi_run_statistical_analysis.py

**Purpose**: Generate statistical analysis across multiple experimental runs.

**Output Directory**: `results/thesis_visuals/statistical_analysis/`

**Output Files**:

- `multi_run_f1_comparison.png` - F1 scores with error bars
- `confidence_intervals.png` - 95% confidence intervals
- `f1_box_plots.png` - Distribution box plots
- `t_test_results.png` - Statistical significance tests
- `all_metrics_summary.png` - All metrics comparison
- `multi_run_statistical_analysis.json` - Raw data
- `STATISTICAL_ANALYSIS_REPORT.md` - Summary report

**Statistical Measures**:

- Mean, standard deviation, standard error
- 95% confidence intervals
- Paired t-tests (vs best method)
- Cohen's d effect size
- Coefficient of variation

**Usage**:

```bash
python scripts/multi_run_statistical_analysis.py
```

---

### 6. generate_thesis_visuals.py

**Purpose**: Generate publication-quality visualizations from evaluation results.

**Output Directory**: `results/thesis_visuals/`

**Output Files**:

- Confusion matrices (8 methods)
- `performance_comparison.png` - Grouped bar chart
- `f1_ranking.png` - F1 score rankings
- `precision_recall_scatter.png` - Precision-recall trade-off
- `error_analysis.png` - False positive/negative rates
- `jetson_performance.png` - Edge device metrics
- `category_comparison.png` - Model category comparison
- `top4_comparison_grid.png` - Top methods grid
- `summary_statistics.png` - Statistics table

**Usage**:

```bash
python scripts/generate_thesis_visuals.py
```

---

## Output Directory Structure

```
results/thesis_visuals/
├── ablation_study/           # Hyperparameter ablation results
├── statistical_analysis/     # Multi-run statistical analysis
├── confusion_matrices/       # Individual confusion matrices
├── thesis_*.png              # EDA visualizations
└── *_comparison.png          # Comparison charts
```

---

## Complete Execution Pipeline

Execute scripts in the following order:

```bash
# 1. Exploratory data analysis
python scripts/exploratory_data_analysis.py

# 2. Training history visualization
python scripts/visualize_training_history.py

# 3. Model training (optional)
python scripts/train_for_visualizations_pytorch.py

# 4. Ablation study
python scripts/hyperparameter_ablation_study.py

# 5. Statistical analysis
python scripts/multi_run_statistical_analysis.py

# 6. Main visualizations
python scripts/generate_thesis_visuals.py
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Best Method | Optimized Ensemble |
| F1 Score | 0.7564 |
| Optimal Epochs | 100 |
| Optimal Learning Rate | 0.0001 |
| Optimal Window Size | 100 |
| Optimal Batch Size | 128 |
| Statistical Significance | p < 0.01 |

---

*Generated: January 2026*
