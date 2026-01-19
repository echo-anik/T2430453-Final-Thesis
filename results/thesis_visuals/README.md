# Thesis Visualizations Guide

##  Location
All publication-quality figures are located in: `results/thesis_visuals/`

---

##  Available Figures

### 1. Confusion Matrices (Individual Models)

**Location:** `results/thesis_visuals/confusion_matrices/`

Each model has its own confusion matrix with a professionally chosen color scheme:

| File | Model | Color Scheme | Best For |
|------|-------|--------------|----------|
| `optimized_ensemble_cm.png` |  Optimized Ensemble | Red-Purple | **Main results figure** |
| `lstm_reconstruction_cm.png` | LSTM Reconstruction | Yellow-Green-Blue | Deep learning comparison |
| `vae_reconstruction_cm.png` | VAE Reconstruction | Purple-Blue-Green | VAE analysis |
| `lstm_latent_distance_cm.png` | LSTM Latent Distance | Yellow-Green-Blue | LSTM variants |
| `vae_kl_divergence_cm.png` | VAE KL Divergence | Purple-Blue-Green | VAE variants |
| `feature_autoencoder_cm.png` | Feature Autoencoder | Yellow-Orange-Red | Autoencoder comparison |
| `isolation_forest_cm.png` | Isolation Forest | Orange | Traditional ML |
| `local_outlier_factor_cm.png` | Local Outlier Factor | Orange | Traditional ML |

**Recommended Usage:**
- **Chapter 4 (Results)**: Use `optimized_ensemble_cm.png` as the main figure
- **Chapter 5 (Discussion)**: Include top 3 models for comparison
- **Appendix**: Include all 8 confusion matrices

---

### 2. Performance Comparison Chart

**File:** `performance_comparison.png`

**Description:** Grouped bar chart showing F1-Score, Precision, and Recall for all models side-by-side

**Recommended Usage:**
- **Chapter 4.1**: Overall model performance comparison
- Great for showing the superiority of the Optimized Ensemble at a glance

**Key Insights:**
- Clear visual ranking of all models
- Shows the trade-off between precision and recall
- Highlights the balanced performance of top models

---

### 3. F1-Score Ranking Chart

**File:** `f1_ranking.png`

**Description:** Horizontal bar chart ranking all models by F1-Score with color-coded performance levels

**Color Coding:**
- 🟢 Green: Excellent (F1 ≥ 0.70)
-  Blue: Good (F1 ≥ 0.60)
- 🟠 Orange: Fair (F1 ≥ 0.55)
-  Red: Poor (F1 < 0.55)

**Recommended Usage:**
- **Chapter 4.2**: Model ranking section
- **Abstract/Executive Summary**: Quick performance overview
- **Introduction**: Motivation for ensemble approach

---

### 4. Precision-Recall Scatter Plot

**File:** `precision_recall_scatter.png`

**Description:** Bubble chart showing the precision-recall trade-off for all models. Bubble size represents F1-Score.

**Recommended Usage:**
- **Chapter 5.1**: Trade-off analysis discussion
- Shows which models prioritize precision vs recall
- Demonstrates the balance achieved by ensemble methods

**Key Insights:**
- Models in the upper-right corner are best (high precision AND recall)
- Larger bubbles indicate better overall performance (F1)
- Helps explain why ensemble performs best

---

### 5. Error Analysis Chart

**File:** `error_analysis.png`

**Description:** Comparison of False Positive Rate (FPR) and False Negative Rate (FNR) across all models

**Recommended Usage:**
- **Chapter 5.2**: Error analysis and failure modes
- **Discussion**: Cost-benefit analysis of different error types
- Critical for SCADA systems where both error types matter

**Key Insights:**
- Shows which models minimize false alarms (low FPR)
- Shows which models catch the most anomalies (low FNR)
- Essential for discussing real-world deployment considerations

---

### 6. Edge Device Performance Chart

**File:** `jetson_performance.png`

**Description:** 2×2 grid comparing FP32 vs FP16 quantization on edge devices:
- Inference Latency
- Throughput
- Power Consumption
- Speedup Factor

**Recommended Usage:**
- **Chapter 4.3**: Edge deployment feasibility
- **Chapter 6**: Real-world implementation discussion
- Shows practical deployment considerations

**Key Insights:**
- FP16 quantization benefits
- Real-time capability analysis
- Energy efficiency for edge deployment

---

### 7. Model Category Comparison

**File:** `category_comparison.png`

**Description:** Bar chart grouping models by architecture type (LSTM, VAE, Traditional ML, Ensemble)

**Recommended Usage:**
- **Chapter 2**: Literature review - comparing approaches
- **Chapter 5**: Discussion of architectural choices
- Shows why deep learning outperforms traditional ML

**Key Insights:**
- Deep learning models consistently outperform traditional ML
- Ensemble approach achieves best results
- Validates architectural design choices

---

### 8. Top 4 Models Comparison Grid

**File:** `top4_comparison_grid.png`

**Description:** 2×2 grid showing confusion matrices for the top 4 performing models side-by-side

**Recommended Usage:**
- **Chapter 4.4**: Detailed comparison of best models
- Perfect for showing why Optimized Ensemble is superior
- Compact way to compare multiple models

**Models Included:**
1.  Optimized Ensemble (F1: 0.7564)
2. LSTM Reconstruction (F1: 0.7018)
3. VAE Reconstruction (F1: 0.6712)
4. LSTM Latent Distance (F1: 0.6287)

---

### 9. Summary Statistics Table

**File:** `summary_statistics.png`

**Description:** Visual table summarizing dataset statistics and best model performance

**Recommended Usage:**
- **Chapter 3**: Dataset description
- **Chapter 4**: Results introduction
- **Tables and Figures**: As Table 1 or Table 2

**Information Included:**
- Dataset size (train/test split)
- Contamination rate
- Best model name and metrics
- Clean, professional table format

---

##  Suggested Figure Captions

### For Optimized Ensemble Confusion Matrix:
```
Figure X: Confusion matrix for the Optimized Ensemble model showing 
classification performance on the WADI test set (n=34,541). The model 
achieved an F1-score of 0.7564 with 1,806 true positives (correctly 
detected anomalies) and only 696 false positives (2.16% false positive rate).
```

### For Performance Comparison Chart:
```
Figure X: Performance comparison across all evaluated models. The Optimized 
Ensemble achieves the highest F1-score (0.7564) with balanced precision 
(0.7218) and recall (0.7947), outperforming both deep learning and 
traditional machine learning baselines.
```

### For Precision-Recall Scatter:
```
Figure X: Precision-recall trade-off analysis for all models. Bubble size 
represents F1-score. The Optimized Ensemble (upper right) achieves the best 
balance between precision and recall, critical for minimizing both false 
alarms and missed anomalies in SCADA systems.
```

### For Error Analysis:
```
Figure X: Comparison of false positive rates (FPR) and false negative rates 
(FNR) across all models. Lower values are better. The Optimized Ensemble 
maintains low FPR (2.16%) while achieving the lowest FNR (20.55%), making 
it suitable for production deployment.
```

### For Edge Device Performance:
```
Figure X: Edge device performance analysis comparing FP32 and FP16 
quantization on Jetson Nano. FP16 quantization reduces inference latency 
and power consumption while maintaining model accuracy, enabling real-time 
anomaly detection on resource-constrained edge devices.
```

---

##  Recommended Thesis Structure

### Chapter 4: Results

**4.1 Model Performance Overview**
- Figure: `performance_comparison.png`
- Figure: `f1_ranking.png`

**4.2 Best Model Analysis**
- Figure: `optimized_ensemble_cm.png` (main result)
- Table: `summary_statistics.png`

**4.3 Comparative Analysis**
- Figure: `top4_comparison_grid.png`
- Figure: `precision_recall_scatter.png`

**4.4 Edge Deployment Feasibility**
- Figure: `jetson_performance.png`

### Chapter 5: Discussion

**5.1 Model Performance Analysis**
- Reference: `category_comparison.png`
- Discuss why ensemble outperforms individual models

**5.2 Error Analysis**
- Figure: `error_analysis.png`
- Discuss trade-offs between FP and FN

**5.3 Practical Implications**
- Reference edge performance results
- Discuss real-time capability

### Appendix A: Detailed Results

- All individual confusion matrices
- Full performance metrics tables

---

##  Color Scheme Guide

All visualizations use a consistent, professional color scheme:

| Color | Hex Code | Usage |
|-------|----------|-------|
| Professional Blue | `#2E86AB` | Primary metrics (F1-Score) |
| Teal Green | `#06A77D` | Positive outcomes (Precision) |
| Orange | `#F18F01` | Warning/Alternative (Recall) |
| Deep Red | `#8B1E3F` | Ensemble model (best) |
| Purple | `#3C096C` | LSTM models |
| Violet | `#5A189A` | VAE models |

All figures use:
- 300 DPI resolution (publication quality)
- White background
- Black borders on bars/elements
- Professional serif font (Times New Roman)
- Consistent sizing and layout

---

##  Quality Checklist

Before including in thesis:
- [ ] All figures are 300 DPI or higher 
- [ ] Text is readable when printed 
- [ ] Color schemes are colorblind-friendly 
- [ ] Consistent styling across all figures 
- [ ] Proper axis labels and titles 
- [ ] Legend included where needed 
- [ ] Professional font (Times New Roman) 
- [ ] White background for printing 

---

##  Tips for Thesis Writing

1. **Always reference figures in text**: "As shown in Figure X, the Optimized Ensemble..."

2. **Explain key findings**: Don't just show the figure, interpret what it means

3. **Use consistent terminology**: Stick to either "Optimized Ensemble" or "best model"

4. **Highlight significance**: "The 5.5% improvement over the baseline..."

5. **Address limitations**: "While the model achieves 96.6% accuracy, the 20.5% FNR..."

6. **Compare to baselines**: Reference the literature baselines from your results

---

##  Figure Formats

All figures are saved as PNG with the following specs:
- Resolution: 300 DPI
- Color space: RGB
- Background: White
- Size: Optimized for full-page or half-page inclusion

If you need different formats (PDF, EPS, SVG), let me know!

---

##  Regenerating Figures

To regenerate all figures with modifications:
```bash
python scripts/generate_thesis_visuals.py
```

All figures will be saved to `results/thesis_visuals/`

---

**Created:** January 20, 2026  
**Total Figures:** 16 publication-quality visualizations  
**Status:** Ready for thesis inclusion 
