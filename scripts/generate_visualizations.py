"""
Visualization Generator
=======================
Generates all figures and visualizations from saved results JSON.

Usage:
    python scripts/generate_visualizations.py
    python scripts/generate_visualizations.py --results-path results/metrics/final_results.json
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from matplotlib.patches import Patch


def load_results(results_path):
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def generate_roc_curves(results, output_dir, labels, scores_dict):
    """Generate ROC-AUC curves for all methods."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Extract unsupervised and semi-supervised methods
    unsup_methods = list(results['unsupervised'].keys())
    semi_methods = list(results['semi_supervised'].keys())
    
    # 1. ROC Curve - Unsupervised Methods
    ax1 = axes[0, 0]
    for method in unsup_methods:
        if method in scores_dict and method in results['auc_scores']:
            # Recreate ROC curve from scores
            scores = scores_dict[method]
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = results['auc_scores'][method]
            style = '-' if 'NOVEL' in method else '--'
            lw = 2 if 'NOVEL' in method else 1.5
            ax1.plot(fpr, tpr, style, lw=lw, label=f'{method} (AUC={roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve - Unsupervised Methods', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. ROC Curve - Semi-Supervised Methods
    ax2 = axes[0, 1]
    for method in semi_methods:
        if method in scores_dict and method in results['auc_scores']:
            scores = scores_dict[method]
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = results['auc_scores'][method]
            style = '-' if 'Ensemble' in method else '--'
            lw = 2.5 if 'Ensemble' in method else 1.5
            ax2.plot(fpr, tpr, style, lw=lw, label=f'{method} (AUC={roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curve - Semi-Supervised Methods (5% labeled)', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve - Best Methods
    ax3 = axes[1, 0]
    best_unsup = results['best_unsupervised']['method']
    best_semi = results['best_semi_supervised']['method']
    
    if best_unsup in scores_dict:
        scores = scores_dict[best_unsup]
        precision_vals, recall_vals, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        ax3.plot(recall_vals, precision_vals, 'b-', lw=2, label=f'Unsup: {best_unsup} (AP={ap:.3f})')
    
    if best_semi in scores_dict:
        scores = scores_dict[best_semi]
        precision_vals, recall_vals, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        ax3.plot(recall_vals, precision_vals, 'r-', lw=2, label=f'Semi: {best_semi} (AP={ap:.3f})')
    
    ax3.set_xlabel('Recall', fontsize=12)
    ax3.set_ylabel('Precision', fontsize=12)
    ax3.set_title('Precision-Recall Curve - Best Methods', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. F1 Score Comparison
    ax4 = axes[1, 1]
    all_methods = []
    all_f1 = []
    colors = []
    
    for method, metrics in results['unsupervised'].items():
        all_methods.append(method)
        all_f1.append(metrics['f1'])
        if method == best_unsup:
            colors.append('gold')
        else:
            colors.append('steelblue')
    
    for method, metrics in results['semi_supervised'].items():
        all_methods.append(method)
        all_f1.append(metrics['f1'])
        if method == best_semi:
            colors.append('gold')
        else:
            colors.append('coral')
    
    bars = ax4.barh(range(len(all_methods)), all_f1, color=colors, edgecolor='black')
    ax4.set_yticks(range(len(all_methods)))
    ax4.set_yticklabels(all_methods, fontsize=9)
    ax4.set_xlabel('F1 Score', fontsize=12)
    ax4.set_title('F1 Score Comparison (All Methods)', fontsize=14, fontweight='bold')
    ax4.set_xlim([0, 1])
    ax4.invert_yaxis()
    
    for bar, score in zip(bars, all_f1):
        ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=9)
    
    legend_elements = [Patch(facecolor='steelblue', label='Unsupervised'),
                      Patch(facecolor='coral', label='Semi-Supervised'),
                      Patch(facecolor='gold', label='Best')]
    ax4.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    output_path = output_dir / 'roc_auc_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def generate_score_distributions(scores_dict, labels, output_dir):
    """Generate score distribution plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    normal_mask = labels == 0
    attack_mask = labels == 1
    
    # Reconstruction Error
    if 'Reconstruction Error' in scores_dict:
        ax = axes[0]
        scores = scores_dict['Reconstruction Error']
        ax.hist(scores[normal_mask], bins=50, alpha=0.7, label='Normal', color='green', density=True)
        ax.hist(scores[attack_mask], bins=50, alpha=0.7, label='Attack', color='red', density=True)
        ax.set_xlabel('Reconstruction Error')
        ax.set_ylabel('Density')
        ax.set_title('Reconstruction Error Distribution')
        ax.legend()
    
    # Latent Distance
    if 'Latent Distance' in scores_dict:
        ax = axes[1]
        scores = scores_dict['Latent Distance']
        ax.hist(scores[normal_mask], bins=50, alpha=0.7, label='Normal', color='green', density=True)
        ax.hist(scores[attack_mask], bins=50, alpha=0.7, label='Attack', color='red', density=True)
        ax.set_xlabel('Latent Distance')
        ax.set_ylabel('Density')
        ax.set_title('Latent Space Distance Distribution')
        ax.legend()
    
    # Feature Distance
    if 'Feature Distance (NOVEL)' in scores_dict:
        ax = axes[2]
        scores = scores_dict['Feature Distance (NOVEL)']
        ax.hist(scores[normal_mask], bins=50, alpha=0.7, label='Normal', color='green', density=True)
        ax.hist(scores[attack_mask], bins=50, alpha=0.7, label='Attack', color='red', density=True)
        ax.set_xlabel('Feature Distance')
        ax.set_ylabel('Density')
        ax.set_title('Engineered Feature Distance Distribution')
        ax.legend()
    
    plt.tight_layout()
    output_path = output_dir / 'score_distributions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def generate_comparison_table(results, output_dir):
    """Generate comparison table with literature baselines."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    table_data = []
    table_data.append(['Category', 'Method', 'F1', 'Precision', 'Recall', 'AUC'])
    
    # Unsupervised
    for method, metrics in sorted(results['unsupervised'].items(), key=lambda x: x[1]['f1'], reverse=True):
        marker = '★' if 'NOVEL' in method else ''
        table_data.append([
            'Unsupervised',
            f"{marker}{method}",
            f"{metrics['f1']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics.get('auc', 0):.4f}"
        ])
    
    # Semi-supervised
    for method, metrics in sorted(results['semi_supervised'].items(), key=lambda x: x[1]['f1'], reverse=True):
        marker = '★' if 'Ensemble' in method else ''
        table_data.append([
            'Semi-Supervised (5%)',
            f"{marker}{method}",
            f"{metrics['f1']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics.get('auc', 0):.4f}"
        ])
    
    # Literature baselines
    for method, metrics in sorted(results['literature_baselines'].items(), key=lambda x: x[1]['f1'], reverse=True):
        table_data.append([
            'Literature',
            method,
            f"{metrics['f1']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            '-'
        ])
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.18, 0.35, 0.1, 0.12, 0.1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows by category
    row_idx = 1
    for i in range(1, len(table_data)):
        category = table_data[i][0]
        if category == 'Unsupervised':
            color = '#E7E6F7'
        elif category == 'Semi-Supervised (5%)':
            color = '#FFE6CC'
        else:
            color = '#E2F0D9'
        
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor(color)
    
    plt.title('Complete Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    output_path = output_dir / 'comparison_table.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate visualizations from results')
    parser.add_argument('--results-path', type=str, default='results/metrics/final_results.json',
                       help='Path to results JSON file')
    parser.add_argument('--scores-path', type=str, default='results/metrics/anomaly_scores.npz',
                       help='Path to anomaly scores NPZ file')
    parser.add_argument('--output-dir', type=str, default='results/figures',
                       help='Output directory for figures')
    args = parser.parse_args()
    
    results_path = Path(args.results_path)
    scores_path = Path(args.scores_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading results from: {results_path}")
    results = load_results(results_path)
    
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}\n")
    
    # Load scores if available
    scores_dict = {}
    labels = None
    if scores_path.exists():
        print(f"Loading anomaly scores from: {scores_path}")
        data = np.load(scores_path, allow_pickle=True)
        scores_dict = data['scores'].item()
        labels = data['labels']
        print(f"✓ Loaded {len(scores_dict)} score arrays\n")
        
        # Generate plots that require scores
        print("Generating ROC curves...")
        generate_roc_curves(results, output_dir, labels, scores_dict)
        
        print("\nGenerating score distributions...")
        generate_score_distributions(scores_dict, labels, output_dir)
    else:
        print(f"⚠ Warning: Scores file not found at {scores_path}")
        print("  Skipping plots that require score arrays\n")
    
    print("\nGenerating comparison table...")
    generate_comparison_table(results, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✓ All visualizations generated successfully!")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
