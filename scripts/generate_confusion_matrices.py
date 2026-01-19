"""
Generate Confusion Matrices for All Evaluated Models
=====================================================
This script computes confusion matrices for all models in the evaluation results.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
RESULTS_DIR = Path("results")
METRICS_DIR = RESULTS_DIR / "metrics"
CONFUSION_DIR = METRICS_DIR / "confusion_matrices"
CONFUSION_DIR.mkdir(parents=True, exist_ok=True)


def calculate_confusion_matrix_from_metrics(precision, recall, total_samples, contamination):
    """
    Calculate confusion matrix components from precision, recall, and dataset stats.
    
    Parameters:
    - precision: Precision score
    - recall: Recall score
    - total_samples: Total number of test samples
    - contamination: Contamination rate (proportion of anomalies)
    
    Returns:
    - confusion matrix as dict with TP, FP, TN, FN
    """
    # Calculate actual number of anomalies and normal samples
    actual_anomalies = int(total_samples * contamination)
    actual_normal = total_samples - actual_anomalies
    
    # From recall: TP = recall * actual_anomalies
    TP = int(recall * actual_anomalies)
    
    # From precision: TP / (TP + FP) = precision
    # Therefore: FP = TP/precision - TP
    FP = int(TP / precision - TP) if precision > 0 else 0
    
    # FN = actual_anomalies - TP
    FN = actual_anomalies - TP
    
    # TN = actual_normal - FP
    TN = actual_normal - FP
    
    return {
        'TP': TP,  # True Positives
        'FP': FP,  # False Positives
        'TN': TN,  # True Negatives
        'FN': FN   # False Negatives
    }


def plot_confusion_matrix(cm_dict, model_name, save_path):
    """
    Plot and save confusion matrix visualization.
    """
    # Create 2x2 matrix
    cm = np.array([
        [cm_dict['TN'], cm_dict['FP']],
        [cm_dict['FN'], cm_dict['TP']]
    ])
    
    # Create figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization: {save_path}")


def generate_confusion_matrices():
    """
    Generate confusion matrices for all models in the results files.
    """
    print("=" * 80)
    print("GENERATING CONFUSION MATRICES")
    print("=" * 80)
    
    # Load unsupervised evaluation results
    results_file = METRICS_DIR / 'unsupervised_v2_results.json'
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    all_methods = results['all_methods']
    dataset_info = results['dataset']
    total_samples = dataset_info['test']
    contamination = dataset_info['contamination']
    
    print(f"\nDataset Information:")
    print(f"  Total test samples: {total_samples}")
    print(f"  Contamination rate: {contamination:.4f}")
    print(f"  Anomalies: {int(total_samples * contamination)}")
    print(f"  Normal samples: {int(total_samples * (1 - contamination))}")
    print()
    
    # Store all confusion matrices
    all_confusion_matrices = {}
    
    print("\nGenerating Confusion Matrices for All Models:")
    print("-" * 80)
    
    for model_name, metrics in all_methods.items():
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1']
        
        # Calculate confusion matrix
        cm_dict = calculate_confusion_matrix_from_metrics(
            precision, recall, total_samples, contamination
        )
        
        # Store in results
        all_confusion_matrices[model_name] = cm_dict
        
        # Print confusion matrix
        print(f"\n{model_name}:")
        print(f"  F1-Score: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"  Confusion Matrix:")
        print(f"                    Predicted")
        print(f"                Normal  Anomaly")
        print(f"  Actual Normal   {cm_dict['TN']:>6}  {cm_dict['FP']:>7}")
        print(f"  Actual Anomaly  {cm_dict['FN']:>6}  {cm_dict['TP']:>7}")
        
        # Calculate accuracy
        accuracy = (cm_dict['TP'] + cm_dict['TN']) / total_samples
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Plot and save visualization
        safe_name = model_name.replace(' ', '_').replace('/', '_').lower()
        plot_path = CONFUSION_DIR / f"{safe_name}_confusion_matrix.png"
        plot_confusion_matrix(cm_dict, model_name, plot_path)
    
    # Save all confusion matrices to JSON
    output = {
        'dataset': {
            'total_samples': total_samples,
            'contamination': contamination,
            'anomalies': int(total_samples * contamination),
            'normal': int(total_samples * (1 - contamination))
        },
        'confusion_matrices': all_confusion_matrices,
        'metrics': {
            model_name: {
                'f1': metrics['f1'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'accuracy': (all_confusion_matrices[model_name]['TP'] + 
                           all_confusion_matrices[model_name]['TN']) / total_samples
            }
            for model_name, metrics in all_methods.items()
        }
    }
    
    output_file = METRICS_DIR / 'confusion_matrices.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"SUMMARY")
    print("=" * 80)
    print(f"✓ Generated confusion matrices for {len(all_methods)} models")
    print(f"✓ Saved visualizations to: {CONFUSION_DIR}/")
    print(f"✓ Saved JSON results to: {output_file}")
    print()
    
    # Print best model summary
    best_model = results['best_method']['name']
    best_cm = all_confusion_matrices[best_model]
    print(f"Best Model: {best_model}")
    print(f"  F1-Score: {results['best_method']['f1']:.4f}")
    print(f"  TP: {best_cm['TP']}, FP: {best_cm['FP']}, TN: {best_cm['TN']}, FN: {best_cm['FN']}")
    print("=" * 80)


if __name__ == "__main__":
    generate_confusion_matrices()
