"""
Data Pipeline Visualization
============================
Shows the complete data transformation from raw dataset to final windowed data.
Visualizes data reduction at each preprocessing step.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Setup
RESULTS_DIR = Path("results")
VISUALS_DIR = RESULTS_DIR / "thesis_visuals"
VISUALS_DIR.mkdir(parents=True, exist_ok=True)

# Professional styling
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.titlesize': 16,
    'font.family': 'serif',
})

COLORS = {
    'train': '#2E86AB',
    'val': '#06A77D',
    'test': '#F18F01',
    'removed': '#C73E1D',
    'kept': '#06A77D',
}


def load_data_stats():
    """Load or calculate all data statistics."""
    
    # Original datasets
    print("Loading original datasets...")
    train_df = pd.read_csv('datasets/WADI_14days_new.csv')
    test_df = pd.read_csv('datasets/WADI_attackdataLABLE.csv', low_memory=False)
    
    # Preprocessed data
    print("Loading preprocessed data...")
    train_windows = np.load('data/preprocessed/train_windows.npy')
    val_windows = np.load('data/preprocessed/val_windows.npy')
    test_windows = np.load('data/preprocessed/test_windows.npy')
    
    with open('data/preprocessed/features.json', 'r') as f:
        features_info = json.load(f)
    
    stats = {
        # Original data
        'original_train_samples': len(train_df),
        'original_train_features': train_df.shape[1],
        'original_test_samples': len(test_df),
        'original_test_features': test_df.shape[1],
        
        # After preprocessing (before windowing)
        # Approximate from windows: n_windows * stride + window_size
        'preprocessed_train_samples': train_windows.shape[0] * 5 + 100,
        'preprocessed_test_samples': test_windows.shape[0] * 5 + 100,
        'preprocessed_features': features_info['n_features'],
        
        # Final windowed data
        'final_train_windows': train_windows.shape[0],
        'final_val_windows': val_windows.shape[0],
        'final_test_windows': test_windows.shape[0],
        'window_size': features_info['window_size'],
        'stride': features_info['stride'],
        
        # Feature reduction
        'initial_features': train_df.shape[1],
        'final_features': features_info['n_features'],
        'features_removed': train_df.shape[1] - features_info['n_features'],
    }
    
    return stats


def create_data_pipeline_flowchart(stats):
    """Create comprehensive data pipeline visualization."""
    print("\n1. Creating Data Pipeline Flowchart...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Define stages
    stages = [
        {
            'title': 'Stage 1: Raw Dataset',
            'y': 0.85,
            'train': f"Training: {stats['original_train_samples']:,} samples × {stats['original_train_features']} features",
            'test': f"Test: {stats['original_test_samples']:,} samples × {stats['original_test_features']} features",
            'total': f"Total: {stats['original_train_samples'] + stats['original_test_samples']:,} samples",
            'color': COLORS['train'],
            'description': '• Raw WADI dataset from water treatment plant\n• 14 days of training data (normal operation)\n• 2 days of test data (with attacks)'
        },
        {
            'title': 'Stage 2: Feature Engineering',
            'y': 0.65,
            'train': f"Training: ~{stats['preprocessed_train_samples']:,} samples × {stats['preprocessed_features']} features",
            'test': f"Test: ~{stats['preprocessed_test_samples']:,} samples × {stats['preprocessed_features']} features",
            'total': f"Removed {stats['features_removed']} features ({stats['features_removed']/stats['initial_features']*100:.1f}%)",
            'color': COLORS['kept'],
            'description': f"• Removed stabilization period (first 6 hours)\n• Removed {stats['features_removed']} constant/low-variance features\n• K-S test for feature selection\n• Outlier clipping & min-max normalization [0,1]"
        },
        {
            'title': 'Stage 3: Sliding Window Creation',
            'y': 0.45,
            'train': f"Training: {stats['final_train_windows']:,} windows",
            'test': f"Test: {stats['final_test_windows']:,} windows",
            'total': f"Window size: {stats['window_size']} timesteps, Stride: {stats['stride']}",
            'color': COLORS['test'],
            'description': f"• Created temporal sequences of length {stats['window_size']}\n• Stride of {stats['stride']} for temporal overlap\n• Each window: [{stats['window_size']} × {stats['preprocessed_features']}] shape"
        },
        {
            'title': 'Stage 4: Final Dataset',
            'y': 0.25,
            'train': f"Train: {stats['final_train_windows']:,} windows  |  Val: {stats['final_val_windows']:,} windows",
            'test': f"Test: {stats['final_test_windows']:,} windows",
            'total': f"Total: {stats['final_train_windows'] + stats['final_val_windows'] + stats['final_test_windows']:,} windows for model training",
            'color': COLORS['val'],
            'description': f"• Train/Val split: 90/10 from training data\n• Ready for LSTM/VAE/Autoencoder models\n• Shape: (windows, {stats['window_size']}, {stats['preprocessed_features']})"
        }
    ]
    
    # Draw stages
    for i, stage in enumerate(stages):
        y = stage['y']
        
        # Box
        box = plt.Rectangle((0.05, y - 0.08), 0.9, 0.15, 
                           facecolor=stage['color'], alpha=0.2,
                           edgecolor=stage['color'], linewidth=3)
        ax.add_patch(box)
        
        # Title
        ax.text(0.5, y + 0.05, stage['title'],
               fontsize=14, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor=stage['color'], linewidth=2))
        
        # Content
        ax.text(0.08, y + 0.01, stage['train'], fontsize=11, fontweight='bold')
        ax.text(0.08, y - 0.02, stage['test'], fontsize=11, fontweight='bold')
        ax.text(0.08, y - 0.05, stage['total'], fontsize=10, style='italic')
        
        # Description
        ax.text(0.92, y, stage['description'],
               fontsize=9, ha='right', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.5))
        
        # Arrow to next stage (except last)
        if i < len(stages) - 1:
            ax.annotate('', xy=(0.5, stages[i+1]['y'] + 0.07), 
                       xytext=(0.5, y - 0.08),
                       arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    # Title
    ax.text(0.5, 0.98, 'Data Preprocessing Pipeline: From Raw Dataset to Model Input',
           fontsize=16, fontweight='bold', ha='center')
    
    # Summary box at bottom
    summary_text = f"""
    Data Reduction Summary:
    • Samples: {stats['original_train_samples'] + stats['original_test_samples']:,} → {stats['final_train_windows'] + stats['final_val_windows'] + stats['final_test_windows']:,} windows
    • Features: {stats['initial_features']} → {stats['preprocessed_features']} ({(1 - stats['preprocessed_features']/stats['initial_features'])*100:.1f}% reduction)
    • Window-based transformation allows temporal pattern learning for anomaly detection
    """
    ax.text(0.5, 0.05, summary_text.strip(),
           fontsize=10, ha='center', va='center',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.3))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    save_path = VISUALS_DIR / 'data_pipeline_flowchart.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {save_path.name}")


def create_sample_reduction_chart(stats):
    """Create bar chart showing sample reduction at each stage."""
    print("\n2. Creating Sample Reduction Chart...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Sample counts
    stages = ['Raw\nDataset', 'After Feature\nEngineering', 'Windowed\nData', 'Final Split']
    train_counts = [
        stats['original_train_samples'],
        stats['preprocessed_train_samples'],
        stats['final_train_windows'],
        stats['final_train_windows']
    ]
    val_counts = [0, 0, 0, stats['final_val_windows']]
    test_counts = [
        stats['original_test_samples'],
        stats['preprocessed_test_samples'],
        stats['final_test_windows'],
        stats['final_test_windows']
    ]
    
    x = np.arange(len(stages))
    width = 0.25
    
    bars1 = ax1.bar(x - width, train_counts, width, label='Training',
                    color=COLORS['train'], alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x, val_counts, width, label='Validation',
                    color=COLORS['val'], alpha=0.8, edgecolor='black', linewidth=0.5)
    bars3 = ax1.bar(x + width, test_counts, width, label='Test',
                    color=COLORS['test'], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Pipeline Stage', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Sample/Window Count', fontsize=13, fontweight='bold')
    ax1.set_title('Data Volume Through Pipeline', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.legend(loc='upper right', frameon=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.annotate(f'{int(height):,}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, rotation=0)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # Right: Feature counts
    feature_stages = ['Raw\nDataset', 'After Feature\nSelection']
    feature_counts = [stats['initial_features'], stats['preprocessed_features']]
    removed = [0, stats['features_removed']]
    
    x2 = np.arange(len(feature_stages))
    bars_kept = ax2.bar(x2, feature_counts, color=COLORS['kept'], alpha=0.8,
                        edgecolor='black', linewidth=0.5, label='Features Kept')
    bars_removed = ax2.bar(x2, removed, bottom=feature_counts, color=COLORS['removed'],
                          alpha=0.8, edgecolor='black', linewidth=0.5, label='Features Removed')
    
    ax2.set_xlabel('Pipeline Stage', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Feature Count', fontsize=13, fontweight='bold')
    ax2.set_title('Feature Reduction', fontsize=14, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(feature_stages)
    ax2.legend(loc='upper right', frameon=True, shadow=True)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars_kept, bars_removed]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height/2),
                           ha='center', va='center', fontsize=12, fontweight='bold',
                           color='white' if height > 10 else 'black')
    
    # Add percentage
    reduction_pct = (stats['features_removed'] / stats['initial_features']) * 100
    ax2.text(1, stats['initial_features'] + 5, f'{reduction_pct:.1f}% reduction',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
    
    plt.suptitle('Dataset Transformation Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = VISUALS_DIR / 'data_reduction_chart.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {save_path.name}")


def create_windowing_visualization(stats):
    """Create visualization explaining the windowing process."""
    print("\n3. Creating Windowing Process Visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Simulate a time series
    time = np.arange(0, 200)
    values = np.sin(time / 10) + np.random.normal(0, 0.1, len(time))
    
    # Plot the time series
    ax.plot(time, values, 'k-', linewidth=1, alpha=0.3, label='Raw Time Series')
    
    # Show windows
    window_size = 20
    stride = 5
    windows_to_show = 6
    colors_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i in range(windows_to_show):
        start = i * stride
        end = start + window_size
        
        if end <= len(time):
            # Highlight window
            ax.axvspan(time[start], time[end], alpha=0.3, color=colors_list[i],
                      label=f'Window {i+1}' if i < 3 else None)
            
            # Mark start and end
            ax.plot([time[start], time[start]], [min(values)-0.5, max(values)+0.5],
                   'k--', linewidth=1, alpha=0.5)
    
    # Add annotations
    ax.annotate('', xy=(time[window_size], max(values)+0.3), 
               xytext=(time[0], max(values)+0.3),
               arrowprops=dict(arrowstyle='<->', lw=2, color='red'))
    ax.text(time[window_size//2], max(values)+0.5, 
           f'Window Size = {stats["window_size"]}',
           ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', linewidth=2))
    
    ax.annotate('', xy=(time[stride], min(values)-0.3), 
               xytext=(time[0], min(values)-0.3),
               arrowprops=dict(arrowstyle='<->', lw=2, color='blue'))
    ax.text(time[stride//2], min(values)-0.5, 
           f'Stride = {stats["stride"]}',
           ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='blue', linewidth=2))
    
    ax.set_xlabel('Time Steps', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sensor Value', fontsize=13, fontweight='bold')
    ax.set_title(f'Sliding Window Process\n'
                f'Creates {stats["final_test_windows"]:,} windows from {stats["preprocessed_test_samples"]:,} samples',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add explanation box
    explanation = f"""
    How Windowing Works:
    
    1. Take {stats['window_size']} consecutive timesteps → Create 1 window
    2. Slide forward by {stats['stride']} timesteps → Create next window
    3. Repeat until end of data
    
    Result: From {stats['preprocessed_test_samples']:,} raw samples
            → {stats['final_test_windows']:,} overlapping windows
    
    Each window captures temporal patterns for anomaly detection
    """
    
    ax.text(0.98, 0.05, explanation.strip(),
           transform=ax.transAxes,
           fontsize=10, ha='right', va='bottom',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    save_path = VISUALS_DIR / 'windowing_process.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {save_path.name}")


def create_data_shape_comparison(stats):
    """Create comprehensive shape comparison."""
    print("\n4. Creating Data Shape Comparison...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Raw data shape
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.7, 'Raw Dataset', fontsize=16, fontweight='bold', ha='center')
    shape_text = f"Training: [{stats['original_train_samples']:,} × {stats['original_train_features']}]\n"
    shape_text += f"Test: [{stats['original_test_samples']:,} × {stats['original_test_features']}]"
    ax1.text(0.5, 0.4, shape_text, fontsize=12, ha='center',
            bbox=dict(boxstyle='round,pad=1', facecolor=COLORS['train'], alpha=0.3))
    ax1.text(0.5, 0.1, f"Total: {stats['original_train_samples'] + stats['original_test_samples']:,} samples",
            fontsize=11, ha='center', fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # 2. After feature engineering
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.7, 'After Feature Engineering', fontsize=16, fontweight='bold', ha='center')
    shape_text = f"Training: [~{stats['preprocessed_train_samples']:,} × {stats['preprocessed_features']}]\n"
    shape_text += f"Test: [~{stats['preprocessed_test_samples']:,} × {stats['preprocessed_features']}]"
    ax2.text(0.5, 0.4, shape_text, fontsize=12, ha='center',
            bbox=dict(boxstyle='round,pad=1', facecolor=COLORS['kept'], alpha=0.3))
    ax2.text(0.5, 0.1, f"Features reduced: {stats['initial_features']} → {stats['preprocessed_features']}",
            fontsize=11, ha='center', fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # 3. After windowing
    ax3 = fig.add_subplot(gs[1, :])
    ax3.text(0.5, 0.8, 'After Sliding Window Creation', fontsize=16, fontweight='bold', ha='center')
    
    window_text = f"Train: [{stats['final_train_windows']:,} × {stats['window_size']} × {stats['preprocessed_features']}]\n\n"
    window_text += f"Val: [{stats['final_val_windows']:,} × {stats['window_size']} × {stats['preprocessed_features']}]\n\n"
    window_text += f"Test: [{stats['final_test_windows']:,} × {stats['window_size']} × {stats['preprocessed_features']}]"
    
    ax3.text(0.5, 0.4, window_text, fontsize=12, ha='center',
            bbox=dict(boxstyle='round,pad=1', facecolor=COLORS['test'], alpha=0.3))
    
    dim_explain = f"3D Shape: (num_windows, timesteps_per_window, features_per_timestep)"
    ax3.text(0.5, 0.05, dim_explain, fontsize=11, ha='center', style='italic')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # 4. Visual representation of 3D tensor
    ax4 = fig.add_subplot(gs[2, :], projection='3d')
    
    # Create a visual representation of the tensor
    n_windows = 5
    n_timesteps = 10
    n_features = 3
    
    for w in range(n_windows):
        for t in range(n_timesteps):
            for f in range(n_features):
                color = plt.cm.viridis(np.random.random())
                ax4.scatter(w, t, f, c=[color], s=50, alpha=0.6, edgecolors='black')
    
    ax4.set_xlabel(f'Windows\n({stats["final_test_windows"]:,} total)', fontsize=11, fontweight='bold')
    ax4.set_ylabel(f'Timesteps\n({stats["window_size"]} per window)', fontsize=11, fontweight='bold')
    ax4.set_zlabel(f'Features\n({stats["preprocessed_features"]} total)', fontsize=11, fontweight='bold')
    ax4.set_title('3D Tensor Structure (Simplified View)', fontsize=13, fontweight='bold')
    
    plt.suptitle('Data Shape Evolution Through Pipeline', fontsize=18, fontweight='bold', y=0.98)
    
    save_path = VISUALS_DIR / 'data_shape_evolution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {save_path.name}")


def create_summary_table(stats):
    """Create detailed summary table."""
    print("\n5. Creating Summary Statistics Table...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    data = [
        ['Pipeline Stage', 'Training Data', 'Test Data', 'Features', 'Notes'],
        ['', '', '', '', ''],
        ['1. Raw Dataset', 
         f"{stats['original_train_samples']:,} samples",
         f"{stats['original_test_samples']:,} samples",
         f"{stats['original_train_features']} features",
         'Original WADI dataset'],
        ['', '', '', '', ''],
        ['2. Feature Engineering',
         f"~{stats['preprocessed_train_samples']:,} samples",
         f"~{stats['preprocessed_test_samples']:,} samples",
         f"{stats['preprocessed_features']} features",
         f"Removed {stats['features_removed']} features"],
        ['', '', '', '', ''],
        ['3. Windowing',
         f"{stats['final_train_windows']:,} windows",
         f"{stats['final_test_windows']:,} windows",
         f"{stats['window_size']}×{stats['preprocessed_features']} shape",
         f"Stride: {stats['stride']} timesteps"],
        ['', '', '', '', ''],
        ['4. Train/Val Split',
         f"Train: {stats['final_train_windows']:,}\nVal: {stats['final_val_windows']:,}",
         f"{stats['final_test_windows']:,} windows",
         f"{stats['window_size']}×{stats['preprocessed_features']} shape",
         '90/10 split from training'],
    ]
    
    table = ax.table(cellText=data, cellLoc='left', loc='center',
                    colWidths=[0.2, 0.22, 0.22, 0.18, 0.18])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Color rows
    colors = ['white', 'white', '#E8F4F8', 'white', '#F0F8E8', 'white', '#FFF8E8', 'white', '#F8E8F8']
    for i in range(1, len(data)):
        for j in range(5):
            table[(i, j)].set_facecolor(colors[i])
    
    plt.title('Complete Data Pipeline Statistics', fontsize=16, fontweight='bold', pad=20)
    
    # Add summary note
    summary = f"""
    Why the data is "smaller" now:
    
    1. Feature Reduction: {stats['initial_features']} → {stats['preprocessed_features']} features ({stats['features_removed']} removed)
       - Removed constant features (solenoid valves)
       - Removed stabilization period (first 6 hours)
       - K-S test and variance filtering
    
    2. Window Transformation: Individual samples → Temporal sequences
       - From {stats['preprocessed_test_samples']:,} individual samples → {stats['final_test_windows']:,} windows
       - Each window contains {stats['window_size']} consecutive timesteps
       - Windows overlap by {stats['window_size'] - stats['stride']} timesteps (stride = {stats['stride']})
    
    3. Result: More efficient representation for temporal pattern learning
       - Original: 1D time series per feature
       - Final: 3D tensor (windows × timesteps × features)
       - Enables LSTM/VAE to learn temporal anomaly patterns
    """
    
    ax.text(0.5, -0.05, summary.strip(),
           transform=ax.transAxes,
           fontsize=10, ha='center', va='top',
           bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.7))
    
    save_path = VISUALS_DIR / 'data_pipeline_summary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {save_path.name}")


def main():
    """Generate all data pipeline visualizations."""
    print("=" * 80)
    print("GENERATING DATA PIPELINE VISUALIZATIONS")
    print("=" * 80)
    
    # Load statistics
    stats = load_data_stats()
    
    print("\nData Pipeline Summary:")
    print(f"  Original: {stats['original_train_samples'] + stats['original_test_samples']:,} samples × {stats['initial_features']} features")
    print(f"  After preprocessing: ~{stats['preprocessed_train_samples'] + stats['preprocessed_test_samples']:,} samples × {stats['preprocessed_features']} features")
    print(f"  Final windows: {stats['final_train_windows'] + stats['final_val_windows'] + stats['final_test_windows']:,} windows")
    print(f"  Feature reduction: {stats['features_removed']} features removed ({stats['features_removed']/stats['initial_features']*100:.1f}%)")
    
    # Create visualizations
    create_data_pipeline_flowchart(stats)
    create_sample_reduction_chart(stats)
    create_windowing_visualization(stats)
    create_data_shape_comparison(stats)
    create_summary_table(stats)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ All data pipeline visualizations saved to: {VISUALS_DIR}/")
    print("\nGenerated figures:")
    print("  • Data pipeline flowchart (complete transformation)")
    print("  • Sample reduction chart (volumes at each stage)")
    print("  • Windowing process visualization (how it works)")
    print("  • Data shape evolution (2D → 3D tensor)")
    print("  • Complete pipeline summary table")
    print("\nThese figures explain why we have 34,541 test 'samples' (windows)")
    print("instead of 172,804 raw samples - it's transformation, not reduction!")
    print("=" * 80)


if __name__ == "__main__":
    main()
