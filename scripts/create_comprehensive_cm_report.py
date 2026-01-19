"""
Create Comprehensive Confusion Matrix Report
=============================================
Generate a single comprehensive visualization showing all model confusion matrices.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
METRICS_DIR = Path("results/metrics")
CONFUSION_DIR = METRICS_DIR / "confusion_matrices"


def create_comprehensive_report():
    """
    Create a comprehensive report with all confusion matrices side-by-side.
    """
    # Load confusion matrices
    cm_file = METRICS_DIR / 'confusion_matrices.json'
    with open(cm_file, 'r') as f:
        data = json.load(f)
    
    confusion_matrices = data['confusion_matrices']
    metrics = data['metrics']
    dataset_info = data['dataset']
    
    # Sort models by F1 score
    sorted_models = sorted(metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    # Create a large figure with subplots
    n_models = len(sorted_models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, model_metrics) in enumerate(sorted_models):
        ax = axes[idx]
        cm_dict = confusion_matrices[model_name]
        
        # Create confusion matrix array
        cm = np.array([
            [cm_dict['TN'], cm_dict['FP']],
            [cm_dict['FN'], cm_dict['TP']]
        ])
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'],
                    cbar=True, cbar_kws={'label': 'Count'})
        
        # Add title with metrics
        title = f"{model_name}\n"
        title += f"F1: {model_metrics['f1']:.4f} | "
        title += f"P: {model_metrics['precision']:.4f} | "
        title += f"R: {model_metrics['recall']:.4f} | "
        title += f"Acc: {model_metrics['accuracy']:.4f}"
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=9)
        ax.set_xlabel('Predicted Label', fontsize=9)
    
    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    # Add overall title
    fig.suptitle(f'Confusion Matrices - All Models\n'
                 f'Dataset: {dataset_info["total_samples"]} samples | '
                 f'Anomalies: {dataset_info["anomalies"]} ({dataset_info["contamination"]:.2%})',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    save_path = CONFUSION_DIR / 'all_models_confusion_matrices.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comprehensive report saved: {save_path}")
    
    # Create a detailed text report
    report_path = CONFUSION_DIR / 'confusion_matrix_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("CONFUSION MATRIX REPORT - ALL MODELS\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("DATASET INFORMATION\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total Test Samples:  {dataset_info['total_samples']:>10,}\n")
        f.write(f"Normal Samples:      {dataset_info['normal']:>10,} ({100*(1-dataset_info['contamination']):.2f}%)\n")
        f.write(f"Anomaly Samples:     {dataset_info['anomalies']:>10,} ({100*dataset_info['contamination']:.2f}%)\n")
        f.write(f"Contamination Rate:  {dataset_info['contamination']:>10.4f}\n")
        f.write("\n\n")
        
        f.write("MODEL PERFORMANCE SUMMARY (Sorted by F1-Score)\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Rank':<6}{'Model':<30}{'F1':<10}{'Precision':<12}{'Recall':<10}{'Accuracy':<10}\n")
        f.write("-" * 100 + "\n")
        
        for rank, (model_name, model_metrics) in enumerate(sorted_models, 1):
            marker = "★" if rank == 1 else " "
            f.write(f"{marker}{rank:<5}{model_name:<30}"
                   f"{model_metrics['f1']:<10.4f}"
                   f"{model_metrics['precision']:<12.4f}"
                   f"{model_metrics['recall']:<10.4f}"
                   f"{model_metrics['accuracy']:<10.4f}\n")
        
        f.write("\n\n")
        f.write("DETAILED CONFUSION MATRICES\n")
        f.write("=" * 100 + "\n\n")
        
        for rank, (model_name, model_metrics) in enumerate(sorted_models, 1):
            cm_dict = confusion_matrices[model_name]
            
            f.write(f"{rank}. {model_name.upper()}\n")
            f.write("-" * 100 + "\n")
            f.write(f"Performance Metrics:\n")
            f.write(f"  F1-Score:   {model_metrics['f1']:.4f}\n")
            f.write(f"  Precision:  {model_metrics['precision']:.4f}\n")
            f.write(f"  Recall:     {model_metrics['recall']:.4f}\n")
            f.write(f"  Accuracy:   {model_metrics['accuracy']:.4f}\n\n")
            
            f.write(f"Confusion Matrix:\n")
            f.write(f"                          Predicted\n")
            f.write(f"                    Normal        Anomaly\n")
            f.write(f"  Actual Normal   {cm_dict['TN']:>8,}      {cm_dict['FP']:>8,}\n")
            f.write(f"  Actual Anomaly  {cm_dict['FN']:>8,}      {cm_dict['TP']:>8,}\n\n")
            
            # Calculate additional metrics
            specificity = cm_dict['TN'] / (cm_dict['TN'] + cm_dict['FP'])
            npv = cm_dict['TN'] / (cm_dict['TN'] + cm_dict['FN']) if (cm_dict['TN'] + cm_dict['FN']) > 0 else 0
            fpr = cm_dict['FP'] / (cm_dict['FP'] + cm_dict['TN'])
            fnr = cm_dict['FN'] / (cm_dict['FN'] + cm_dict['TP'])
            
            f.write(f"Additional Metrics:\n")
            f.write(f"  True Positives (TP):       {cm_dict['TP']:>8,}  (Correctly detected anomalies)\n")
            f.write(f"  True Negatives (TN):       {cm_dict['TN']:>8,}  (Correctly detected normal)\n")
            f.write(f"  False Positives (FP):      {cm_dict['FP']:>8,}  (Normal misclassified as anomaly)\n")
            f.write(f"  False Negatives (FN):      {cm_dict['FN']:>8,}  (Missed anomalies)\n")
            f.write(f"  Specificity (TNR):         {specificity:>8.4f}  (TN / (TN + FP))\n")
            f.write(f"  False Positive Rate (FPR): {fpr:>8.4f}  (FP / (FP + TN))\n")
            f.write(f"  False Negative Rate (FNR): {fnr:>8.4f}  (FN / (FN + TP))\n")
            f.write(f"  Negative Predictive Value: {npv:>8.4f}  (TN / (TN + FN))\n")
            f.write("\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")
    
    print(f"✓ Text report saved: {report_path}")


if __name__ == "__main__":
    create_comprehensive_report()
