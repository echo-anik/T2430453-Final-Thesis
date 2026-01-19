"""
Create visualizations from existing JSON results files
Generates charts for both final_results.json and unsupervised_v2_results.json
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Load both result files
with open("results/metrics/final_results.json", "r") as f:
    final_results = json.load(f)

with open("results/metrics/unsupervised_v2_results.json", "r") as f:
    unsupervised_results = json.load(f)

print("📊 Generating visualizations from JSON results...\n")

# ============================================================================
# CHART 1: F1 Scores - All Methods from final_results.json
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

methods = []
f1_scores = []
colors_list = []

# Unsupervised methods
for method, metrics in final_results['unsupervised'].items():
    methods.append(method)
    f1_scores.append(metrics['f1'])
    colors_list.append('#FF6B6B')  # Red for unsupervised

# Semi-supervised methods
for method, metrics in final_results['semi_supervised'].items():
    methods.append(method)
    f1_scores.append(metrics['f1'])
    colors_list.append('#4ECDC4')  # Teal for semi-supervised

# Baseline methods
for method, metrics in final_results['baseline'].items():
    methods.append(method)
    f1_scores.append(metrics['f1'])
    colors_list.append('#95E1D3')  # Light teal for baseline

# Literature baselines
for method, metrics in final_results['literature_baselines'].items():
    methods.append(method)
    f1_scores.append(metrics['f1'])
    colors_list.append('#CCCCCC')  # Gray for literature

# Sort by F1 score
sorted_data = sorted(zip(methods, f1_scores, colors_list), key=lambda x: x[1], reverse=True)
methods, f1_scores, colors_list = zip(*sorted_data)

bars = ax.barh(range(len(methods)), f1_scores, color=colors_list)

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    ax.text(score + 0.01, i, f'{score:.4f}', va='center', fontsize=9, fontweight='bold')

ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods, fontsize=10)
ax.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('F1 Score Comparison - All Methods (final_results.json)', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, 1)
ax.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4ECDC4', label='Semi-Supervised'),
    Patch(facecolor='#95E1D3', label='Baseline'),
    Patch(facecolor='#FF6B6B', label='Unsupervised'),
    Patch(facecolor='#CCCCCC', label='Literature')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "01_f1_scores_all_methods.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 01_f1_scores_all_methods.png")
plt.close()

# ============================================================================
# CHART 2: Precision vs Recall - All Methods
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Collect all methods with metrics
all_data = {}

for method, metrics in final_results['unsupervised'].items():
    all_data[method] = (metrics['precision'], metrics['recall'], 'Unsupervised')

for method, metrics in final_results['semi_supervised'].items():
    all_data[method] = (metrics['precision'], metrics['recall'], 'Semi-Supervised')

for method, metrics in final_results['baseline'].items():
    all_data[method] = (metrics['precision'], metrics['recall'], 'Baseline')

# Color mapping
color_map = {
    'Unsupervised': '#FF6B6B',
    'Semi-Supervised': '#4ECDC4',
    'Baseline': '#95E1D3'
}

for method, (precision, recall, category) in all_data.items():
    ax.scatter(recall, precision, s=200, alpha=0.7, color=color_map[category], label=method, edgecolors='black', linewidth=1)
    ax.annotate(method, (recall, precision), fontsize=8, ha='center', va='bottom')

ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision vs Recall - All Methods', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4ECDC4', label='Semi-Supervised'),
    Patch(facecolor='#95E1D3', label='Baseline'),
    Patch(facecolor='#FF6B6B', label='Unsupervised')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "02_precision_vs_recall.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 02_precision_vs_recall.png")
plt.close()

# ============================================================================
# CHART 3: AUC Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

methods = []
auc_scores = []
colors_list = []

# Unsupervised
for method, metrics in final_results['unsupervised'].items():
    methods.append(method)
    auc_scores.append(metrics['auc'])
    colors_list.append('#FF6B6B')

# Semi-supervised
for method, metrics in final_results['semi_supervised'].items():
    methods.append(method)
    auc_scores.append(metrics['auc'])
    colors_list.append('#4ECDC4')

# Baseline
for method, metrics in final_results['baseline'].items():
    methods.append(method)
    auc_scores.append(metrics['auc'])
    colors_list.append('#95E1D3')

# Sort by AUC
sorted_data = sorted(zip(methods, auc_scores, colors_list), key=lambda x: x[1], reverse=True)
methods, auc_scores, colors_list = zip(*sorted_data)

bars = ax.barh(range(len(methods)), auc_scores, color=colors_list)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, auc_scores)):
    ax.text(score + 0.01, i, f'{score:.4f}', va='center', fontsize=9, fontweight='bold')

ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods, fontsize=10)
ax.set_xlabel('AUC Score', fontsize=12, fontweight='bold')
ax.set_title('AUC Score Comparison - All Methods', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, 1)
ax.grid(axis='x', alpha=0.3)

# Legend
legend_elements = [
    Patch(facecolor='#4ECDC4', label='Semi-Supervised'),
    Patch(facecolor='#95E1D3', label='Baseline'),
    Patch(facecolor='#FF6B6B', label='Unsupervised')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "03_auc_scores.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 03_auc_scores.png")
plt.close()

# ============================================================================
# CHART 4: Unsupervised V2 Results - F1 Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

methods_v2 = []
f1_v2 = []

for method, metrics in unsupervised_results['all_methods'].items():
    methods_v2.append(method)
    f1_v2.append(metrics['f1'])

# Sort by F1
sorted_data = sorted(zip(methods_v2, f1_v2), key=lambda x: x[1], reverse=True)
methods_v2, f1_v2 = zip(*sorted_data)

bars = ax.barh(range(len(methods_v2)), f1_v2, color='#FF6B6B')

# Add value labels
for i, (bar, score) in enumerate(zip(bars, f1_v2)):
    ax.text(score + 0.01, i, f'{score:.4f}', va='center', fontsize=9, fontweight='bold')

ax.set_yticks(range(len(methods_v2)))
ax.set_yticklabels(methods_v2, fontsize=10)
ax.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('F1 Score Comparison - Unsupervised V2 Methods', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, 1)
ax.grid(axis='x', alpha=0.3)

# Highlight best method
best_idx = f1_v2.index(max(f1_v2))
bars[best_idx].set_color('#2ECC71')
bars[best_idx].set_edgecolor('black')
bars[best_idx].set_linewidth(2)

plt.tight_layout()
plt.savefig(output_dir / "04_unsupervised_v2_f1_scores.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 04_unsupervised_v2_f1_scores.png")
plt.close()

# ============================================================================
# CHART 5: Unsupervised V2 - Precision, Recall, F1 (Best Method)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

best_method = unsupervised_results['best_method']['name']
metrics = {
    'Precision': unsupervised_results['best_method']['precision'],
    'Recall': unsupervised_results['best_method']['recall'],
    'F1 Score': unsupervised_results['best_method']['f1']
}

colors = ['#4ECDC4', '#FF6B6B', '#2ECC71']
bars = ax.bar(metrics.keys(), metrics.values(), color=colors, edgecolor='black', linewidth=2)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title(f'Best Unsupervised V2 Method: {best_method}', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "05_unsupervised_v2_best_method.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 05_unsupervised_v2_best_method.png")
plt.close()

# ============================================================================
# CHART 6: Method Comparison Matrix (Heatmap)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 10))

# Prepare data for heatmap
all_methods_list = []
metrics_matrix = []

# Collect all methods
for method in final_results['semi_supervised']:
    all_methods_list.append(method)
for method in final_results['unsupervised']:
    all_methods_list.append(method)

# Create matrix with F1, Precision, Recall
for method in all_methods_list:
    if method in final_results['semi_supervised']:
        metrics_matrix.append([
            final_results['semi_supervised'][method]['f1'],
            final_results['semi_supervised'][method]['precision'],
            final_results['semi_supervised'][method]['recall'],
            final_results['semi_supervised'][method]['auc']
        ])
    else:
        metrics_matrix.append([
            final_results['unsupervised'][method]['f1'],
            final_results['unsupervised'][method]['precision'],
            final_results['unsupervised'][method]['recall'],
            final_results['unsupervised'][method]['auc']
        ])

metrics_matrix = np.array(metrics_matrix)

im = ax.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# Set ticks and labels
ax.set_xticks(np.arange(4))
ax.set_yticks(np.arange(len(all_methods_list)))
ax.set_xticklabels(['F1', 'Precision', 'Recall', 'AUC'], fontsize=11, fontweight='bold')
ax.set_yticklabels(all_methods_list, fontsize=10)

# Rotate the tick labels
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

# Add values in cells
for i in range(len(all_methods_list)):
    for j in range(4):
        text = ax.text(j, i, f'{metrics_matrix[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=9, fontweight='bold')

ax.set_title('Performance Metrics Heatmap - All Methods', fontsize=14, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Score', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "06_metrics_heatmap.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 06_metrics_heatmap.png")
plt.close()

# ============================================================================
# CHART 7: Category Comparison - Semi vs Unsupervised vs Baseline
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

categories = ['Semi-Supervised', 'Unsupervised', 'Baseline']
metric_names = ['F1', 'Precision', 'Recall']
results_data = [final_results['semi_supervised'], final_results['unsupervised'], final_results['baseline']]

for idx, (ax, category, results) in enumerate(zip(axes, categories, results_data)):
    methods = list(results.keys())
    f1_scores = [results[m]['f1'] for m in methods]
    
    colors = ['#2ECC71' if score == max(f1_scores) else '#4ECDC4' for score in f1_scores]
    
    bars = ax.barh(range(len(methods)), f1_scores, color=colors)
    
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        ax.text(score + 0.01, i, f'{score:.4f}', va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel('F1 Score', fontsize=11, fontweight='bold')
    ax.set_title(f'{category}', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)

fig.suptitle('F1 Score by Category', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "07_category_comparison.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 07_category_comparison.png")
plt.close()

print("\n" + "="*50)
print("✅ All visualizations generated successfully!")
print("="*50)
print(f"\n📁 Output directory: {output_dir}")
print(f"\n📊 Generated charts:")
print("   1. 01_f1_scores_all_methods.png")
print("   2. 02_precision_vs_recall.png")
print("   3. 03_auc_scores.png")
print("   4. 04_unsupervised_v2_f1_scores.png")
print("   5. 05_unsupervised_v2_best_method.png")
print("   6. 06_metrics_heatmap.png")
print("   7. 07_category_comparison.png")
