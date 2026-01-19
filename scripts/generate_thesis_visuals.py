"""
Generate Publication-Quality Visualizations for Thesis
======================================================
Creates all necessary figures, graphs, and visuals for thesis paper
with professional styling and better color schemes.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Setup
RESULTS_DIR = Path("results")
METRICS_DIR = RESULTS_DIR / "metrics"
VISUALS_DIR = RESULTS_DIR / "thesis_visuals"
VISUALS_DIR.mkdir(parents=True, exist_ok=True)

# Professional color schemes
COLORS = {
    'primary': '#2E86AB',      # Professional blue
    'secondary': '#A23B72',    # Deep magenta
    'success': '#06A77D',      # Teal green
    'warning': '#F18F01',      # Orange
    'danger': '#C73E1D',       # Red
    'neutral': '#6C757D',      # Gray
    'ensemble': '#8B1E3F',     # Deep red (for best model)
    'lstm': '#3C096C',         # Purple
    'vae': '#5A189A',          # Violet
    'ml': '#F18F01',           # Orange (traditional ML)
}

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Font settings for publication quality
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
})


def load_results():
    """Load all results files."""
    with open(METRICS_DIR / 'unsupervised_v2_results.json', 'r') as f:
        unsupervised_results = json.load(f)
    
    with open(METRICS_DIR / 'confusion_matrices.json', 'r') as f:
        cm_results = json.load(f)
    
    with open(METRICS_DIR / 'jetson_nano_results.json', 'r') as f:
        jetson_results = json.load(f)
    
    return unsupervised_results, cm_results, jetson_results


def plot_confusion_matrix_professional(cm_dict, model_name, save_path, colormap='RdPu'):
    """
    Plot confusion matrix with professional styling and better colors.
    
    Available colormaps:
    - 'RdPu': Red-Purple (good contrast)
    - 'YlGnBu': Yellow-Green-Blue (professional)
    - 'YlOrRd': Yellow-Orange-Red (warm)
    - 'PuBuGn': Purple-Blue-Green (cool)
    """
    cm = np.array([
        [cm_dict['TN'], cm_dict['FP']],
        [cm_dict['FN'], cm_dict['TP']]
    ])
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Use professional colormap
    im = ax.imshow(cm, cmap=colormap, alpha=0.8)
    
    # Add colorbar with label
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Sample Count', rotation=270, labelpad=20, fontsize=11)
    
    # Set ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Anomaly'], fontsize=12)
    ax.set_yticklabels(['Normal', 'Anomaly'], fontsize=12)
    
    # Add text annotations with adaptive color
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > thresh else 'black'
            text = ax.text(j, i, f'{cm[i, j]:,}',
                          ha="center", va="center", color=color,
                          fontsize=16, fontweight='bold')
    
    # Add labels
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    
    # Add title with metrics
    f1 = (2 * cm_dict['TP']) / (2 * cm_dict['TP'] + cm_dict['FP'] + cm_dict['FN'])
    precision = cm_dict['TP'] / (cm_dict['TP'] + cm_dict['FP'])
    recall = cm_dict['TP'] / (cm_dict['TP'] + cm_dict['FN'])
    
    title = f'{model_name}\n'
    title += f'F1: {f1:.4f}  |  Precision: {precision:.4f}  |  Recall: {recall:.4f}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {save_path.name}")


def create_all_confusion_matrices(cm_results):
    """Generate all confusion matrices with better colors."""
    print("\n1. Generating Confusion Matrices (Improved Colors)...")
    print("-" * 80)
    
    cm_dir = VISUALS_DIR / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    
    confusion_matrices = cm_results['confusion_matrices']
    metrics = cm_results['metrics']
    
    # Sort by F1 score
    sorted_models = sorted(metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    # Color schemes for different model types
    colormap_mapping = {
        'Optimized Ensemble': 'RdPu',
        'LSTM Reconstruction': 'YlGnBu',
        'LSTM Latent Distance': 'YlGnBu',
        'VAE Reconstruction': 'PuBuGn',
        'VAE KL Divergence': 'PuBuGn',
        'Feature Autoencoder': 'YlOrRd',
        'Isolation Forest': 'Oranges',
        'Local Outlier Factor': 'Oranges'
    }
    
    for model_name, _ in sorted_models:
        cm_dict = confusion_matrices[model_name]
        safe_name = model_name.replace(' ', '_').replace('/', '_').lower()
        save_path = cm_dir / f"{safe_name}_cm.png"
        colormap = colormap_mapping.get(model_name, 'RdPu')
        plot_confusion_matrix_professional(cm_dict, model_name, save_path, colormap)


def create_performance_comparison_chart(unsupervised_results):
    """Create comprehensive performance comparison chart."""
    print("\n2. Creating Performance Comparison Chart...")
    print("-" * 80)
    
    all_methods = unsupervised_results['all_methods']
    
    # Extract data
    models = []
    f1_scores = []
    precisions = []
    recalls = []
    
    for model, metrics in sorted(all_methods.items(), key=lambda x: x[1]['f1'], reverse=True):
        models.append(model)
        f1_scores.append(metrics['f1'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
    
    # Create figure with grouped bars
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax.bar(x - width, f1_scores, width, label='F1-Score',
                   color=COLORS['primary'], alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, precisions, width, label='Precision',
                   color=COLORS['success'], alpha=0.8, edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, recalls, width, label='Recall',
                   color=COLORS['warning'], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize
    ax.set_xlabel('Models', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance Comparison\nF1-Score, Precision, and Recall', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.tight_layout()
    save_path = VISUALS_DIR / 'performance_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {save_path.name}")


def create_f1_ranking_chart(unsupervised_results):
    """Create F1-score ranking horizontal bar chart."""
    print("\n3. Creating F1-Score Ranking Chart...")
    print("-" * 80)
    
    all_methods = unsupervised_results['all_methods']
    
    # Sort by F1
    sorted_methods = sorted(all_methods.items(), key=lambda x: x[1]['f1'])
    models = [m[0] for m in sorted_methods]
    f1_scores = [m[1]['f1'] for m in sorted_methods]
    
    # Color code by performance
    colors = []
    for score in f1_scores:
        if score >= 0.70:
            colors.append(COLORS['success'])
        elif score >= 0.60:
            colors.append(COLORS['primary'])
        elif score >= 0.55:
            colors.append(COLORS['warning'])
        else:
            colors.append(COLORS['danger'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(models, f1_scores, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        ax.text(score + 0.01, i, f'{score:.4f}', 
               va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('F1-Score', fontsize=13, fontweight='bold')
    ax.set_title('Model Ranking by F1-Score', fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim([0, 1.0])
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add threshold lines
    ax.axvline(x=0.70, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Excellent (≥0.70)')
    ax.axvline(x=0.60, color='blue', linestyle='--', alpha=0.5, linewidth=2, label='Good (≥0.60)')
    ax.axvline(x=0.55, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Fair (≥0.55)')
    ax.legend(loc='lower right', frameon=True, shadow=True)
    
    plt.tight_layout()
    save_path = VISUALS_DIR / 'f1_ranking.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {save_path.name}")


def create_precision_recall_scatter(unsupervised_results):
    """Create precision-recall scatter plot."""
    print("\n4. Creating Precision-Recall Scatter Plot...")
    print("-" * 80)
    
    all_methods = unsupervised_results['all_methods']
    
    models = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for model, metrics in all_methods.items():
        models.append(model)
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Size bubbles by F1-score
    sizes = [f1 * 1000 for f1 in f1_scores]
    
    scatter = ax.scatter(recalls, precisions, s=sizes, alpha=0.6,
                        c=f1_scores, cmap='RdYlGn', edgecolors='black',
                        linewidth=1.5)
    
    # Add model labels
    for i, model in enumerate(models):
        ax.annotate(model, (recalls[i], precisions[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                   facecolor='yellow', alpha=0.3))
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('F1-Score', rotation=270, labelpad=20, fontsize=11)
    
    ax.set_xlabel('Recall (Sensitivity)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax.set_title('Precision-Recall Trade-off\n(Bubble size represents F1-Score)',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim([0.4, 0.85])
    ax.set_ylim([0.4, 0.85])
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add diagonal line (perfect balance)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Perfect Balance')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    save_path = VISUALS_DIR / 'precision_recall_scatter.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {save_path.name}")


def create_error_analysis_chart(cm_results):
    """Create error analysis chart showing FP and FN rates."""
    print("\n5. Creating Error Analysis Chart...")
    print("-" * 80)
    
    confusion_matrices = cm_results['confusion_matrices']
    metrics = cm_results['metrics']
    
    models = []
    fpr = []  # False Positive Rate
    fnr = []  # False Negative Rate
    
    for model in sorted(metrics.keys(), key=lambda x: metrics[x]['f1'], reverse=True):
        cm = confusion_matrices[model]
        models.append(model)
        fpr.append(cm['FP'] / (cm['FP'] + cm['TN']))
        fnr.append(cm['FN'] / (cm['FN'] + cm['TP']))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, fpr, width, label='False Positive Rate (FPR)',
                   color=COLORS['danger'], alpha=0.7, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, fnr, width, label='False Negative Rate (FNR)',
                   color=COLORS['warning'], alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Models', fontsize=13, fontweight='bold')
    ax.set_ylabel('Error Rate', fontsize=13, fontweight='bold')
    ax.set_title('Error Analysis: False Positive vs False Negative Rates\n(Lower is Better)',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    save_path = VISUALS_DIR / 'error_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {save_path.name}")


def create_jetson_performance_chart(jetson_results):
    """Create Jetson Nano performance comparison chart."""
    print("\n6. Creating Edge Device Performance Chart...")
    print("-" * 80)
    
    models = list(jetson_results['models'].keys())
    
    # Extract FP32 and FP16 metrics
    inference_fp32 = [jetson_results['models'][m]['fp32']['inference_time_ms']['mean'] for m in models]
    inference_fp16 = [jetson_results['models'][m]['fp16']['inference_time_ms']['mean'] for m in models]
    
    throughput_fp32 = [jetson_results['models'][m]['fp32']['throughput_samples_per_sec'] for m in models]
    throughput_fp16 = [jetson_results['models'][m]['fp16']['throughput_samples_per_sec'] for m in models]
    
    power_fp32 = [jetson_results['models'][m]['fp32']['power_draw_watts'] for m in models]
    power_fp16 = [jetson_results['models'][m]['fp16']['power_draw_watts'] for m in models]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Edge Device Performance: FP32 vs FP16 Quantization', 
                fontsize=16, fontweight='bold')
    
    x = np.arange(len(models))
    width = 0.35
    
    # Inference Time
    ax = axes[0, 0]
    ax.bar(x - width/2, inference_fp32, width, label='FP32', 
           color=COLORS['primary'], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, inference_fp16, width, label='FP16',
           color=COLORS['success'], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title('Inference Latency (Lower is Better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Throughput
    ax = axes[0, 1]
    ax.bar(x - width/2, throughput_fp32, width, label='FP32',
           color=COLORS['primary'], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, throughput_fp16, width, label='FP16',
           color=COLORS['success'], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Samples/sec', fontweight='bold')
    ax.set_title('Throughput (Higher is Better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Power Draw
    ax = axes[1, 0]
    ax.bar(x - width/2, power_fp32, width, label='FP32',
           color=COLORS['primary'], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, power_fp16, width, label='FP16',
           color=COLORS['success'], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Watts', fontweight='bold')
    ax.set_title('Power Consumption (Lower is Better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Speedup comparison
    ax = axes[1, 1]
    speedups = [jetson_results['models'][m]['speedup_fp16_vs_fp32'] for m in models]
    colors_speedup = [COLORS['success'] if s > 1 else COLORS['warning'] for s in speedups]
    bars = ax.bar(models, speedups, color=colors_speedup, alpha=0.8, 
                  edgecolor='black', linewidth=0.5)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='No Improvement')
    ax.set_ylabel('Speedup Factor', fontweight='bold')
    ax.set_title('FP16 Speedup vs FP32', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_path = VISUALS_DIR / 'jetson_performance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {save_path.name}")


def create_model_category_comparison(unsupervised_results):
    """Create comparison by model category."""
    print("\n7. Creating Model Category Comparison...")
    print("-" * 80)
    
    all_methods = unsupervised_results['all_methods']
    
    # Categorize models
    categories = {
        'Deep Learning\n(LSTM)': ['LSTM Reconstruction', 'LSTM Latent Distance'],
        'Deep Learning\n(VAE)': ['VAE Reconstruction', 'VAE KL Divergence'],
        'Traditional ML': ['Isolation Forest', 'Local Outlier Factor'],
        'Other Neural': ['Feature Autoencoder'],
        'Ensemble': ['Optimized Ensemble']
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x_pos = 0
    bar_positions = []
    bar_labels = []
    bar_colors = []
    bar_heights = []
    
    category_colors = {
        'Deep Learning\n(LSTM)': COLORS['lstm'],
        'Deep Learning\n(VAE)': COLORS['vae'],
        'Traditional ML': COLORS['ml'],
        'Other Neural': COLORS['warning'],
        'Ensemble': COLORS['ensemble']
    }
    
    for category, models in categories.items():
        for model in models:
            if model in all_methods:
                bar_positions.append(x_pos)
                bar_labels.append(model)
                bar_colors.append(category_colors[category])
                bar_heights.append(all_methods[model]['f1'])
                x_pos += 1
        x_pos += 0.5  # Add gap between categories
    
    bars = ax.bar(bar_positions, bar_heights, color=bar_colors, 
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, rotation=45, ha='right')
    ax.set_ylabel('F1-Score', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance by Category', fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim([0, 0.85])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Create legend
    legend_elements = [mpatches.Patch(facecolor=color, edgecolor='black', 
                                     label=cat, alpha=0.8)
                      for cat, color in category_colors.items()]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, shadow=True)
    
    plt.tight_layout()
    save_path = VISUALS_DIR / 'category_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {save_path.name}")


def create_comprehensive_comparison_grid(unsupervised_results, cm_results):
    """Create comprehensive 2x2 grid comparing best models."""
    print("\n8. Creating Best Models Comparison Grid...")
    print("-" * 80)
    
    # Get top 4 models
    all_methods = unsupervised_results['all_methods']
    top_models = sorted(all_methods.items(), key=lambda x: x[1]['f1'], reverse=True)[:4]
    
    confusion_matrices = cm_results['confusion_matrices']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Top 4 Models: Confusion Matrices Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    
    colormaps = ['RdPu', 'YlGnBu', 'PuBuGn', 'YlOrRd']
    
    for idx, ((model_name, metrics), cmap) in enumerate(zip(top_models, colormaps)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        cm_dict = confusion_matrices[model_name]
        cm = np.array([
            [cm_dict['TN'], cm_dict['FP']],
            [cm_dict['FN'], cm_dict['TP']]
        ])
        
        # Plot heatmap
        im = ax.imshow(cm, cmap=cmap, alpha=0.8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)
        
        # Add annotations
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > thresh else 'black'
                ax.text(j, i, f'{cm[i, j]:,}',
                       ha="center", va="center", color=color,
                       fontsize=14, fontweight='bold')
        
        # Set labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Anomaly'], fontsize=11)
        ax.set_yticklabels(['Normal', 'Anomaly'], fontsize=11)
        
        if row == 1:
            ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        if col == 0:
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
        
        # Title with rank
        rank = idx + 1
        marker = "★ " if rank == 1 else f"{rank}. "
        title = f'{marker}{model_name}\n'
        title += f'F1: {metrics["f1"]:.4f} | P: {metrics["precision"]:.4f} | R: {metrics["recall"]:.4f}'
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        
        # Add grid
        ax.set_xticks([0.5], minor=True)
        ax.set_yticks([0.5], minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=2)
    
    plt.tight_layout()
    save_path = VISUALS_DIR / 'top4_comparison_grid.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {save_path.name}")


def create_summary_statistics_table(unsupervised_results, cm_results):
    """Create a visual summary statistics table."""
    print("\n9. Creating Summary Statistics Table...")
    print("-" * 80)
    
    dataset_info = unsupervised_results['dataset']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Dataset statistics
    data = [
        ['Metric', 'Value'],
        ['Total Training Samples', f"{dataset_info['train']:,}"],
        ['Total Test Samples', f"{dataset_info['test']:,}"],
        ['Anomaly Samples (Test)', f"{int(dataset_info['test'] * dataset_info['contamination']):,}"],
        ['Normal Samples (Test)', f"{int(dataset_info['test'] * (1 - dataset_info['contamination'])):,}"],
        ['Contamination Rate', f"{dataset_info['contamination']:.2%}"],
        ['', ''],
        ['Best Model', unsupervised_results['best_method']['name']],
        ['Best F1-Score', f"{unsupervised_results['best_method']['f1']:.4f}"],
        ['Best Precision', f"{unsupervised_results['best_method']['precision']:.4f}"],
        ['Best Recall', f"{unsupervised_results['best_method']['recall']:.4f}"],
    ]
    
    table = ax.table(cellText=data, cellLoc='left', loc='center',
                    colWidths=[0.5, 0.5])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        cell = table[(0, i)]
        cell.set_facecolor(COLORS['primary'])
        cell.set_text_props(weight='bold', color='white')
    
    # Style best model rows
    for i in range(7, 11):
        cell = table[(i, 0)]
        cell.set_facecolor('#E8F4F8')
        cell = table[(i, 1)]
        cell.set_facecolor('#E8F4F8')
        cell.set_text_props(weight='bold')
    
    plt.title('Dataset and Model Statistics Summary', 
             fontsize=14, fontweight='bold', pad=20)
    
    save_path = VISUALS_DIR / 'summary_statistics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {save_path.name}")


def main():
    """Generate all thesis visualizations."""
    print("=" * 80)
    print("GENERATING THESIS-QUALITY VISUALIZATIONS")
    print("=" * 80)
    
    # Load results
    unsupervised_results, cm_results, jetson_results = load_results()
    
    # Generate all visualizations
    create_all_confusion_matrices(cm_results)
    create_performance_comparison_chart(unsupervised_results)
    create_f1_ranking_chart(unsupervised_results)
    create_precision_recall_scatter(unsupervised_results)
    create_error_analysis_chart(cm_results)
    create_jetson_performance_chart(jetson_results)
    create_model_category_comparison(unsupervised_results)
    create_comprehensive_comparison_grid(unsupervised_results, cm_results)
    create_summary_statistics_table(unsupervised_results, cm_results)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ All visualizations saved to: {VISUALS_DIR}/")
    print(f"✓ Generated {len(list(VISUALS_DIR.rglob('*.png')))} publication-quality figures")
    print("\nFigures ready for thesis inclusion:")
    print("  • Confusion matrices with professional color schemes")
    print("  • Performance comparison charts")
    print("  • F1-score rankings")
    print("  • Precision-recall trade-off analysis")
    print("  • Error analysis (FPR vs FNR)")
    print("  • Edge device performance metrics")
    print("  • Model category comparisons")
    print("  • Top models comparison grid")
    print("  • Summary statistics table")
    print("=" * 80)


if __name__ == "__main__":
    main()
