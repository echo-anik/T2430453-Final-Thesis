"""
Thesis-Quality Visualizations: Data Analysis
=============================================
Simplified script for generating publication-ready visualizations.
Focuses on key insights for thesis presentation.

Author: Thesis Project
Date: 2026-01-24
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Setup directories
DATA_DIR = Path("data")
RAW_DATA_DIR = Path("datasets")
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
RESULTS_DIR = Path("results")
VISUALS_DIR = RESULTS_DIR / "thesis_visuals"
VISUALS_DIR.mkdir(parents=True, exist_ok=True)

# Professional styling
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')
sns.set_palette("husl")

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
})

# Color scheme
COLORS = {
    'before': '#C73E1D',      # Red (raw)
    'after': '#06A77D',       # Teal (processed)
    'normal': '#2E86AB',      # Blue
    'attack': '#F18F01',      # Orange
    'grid': '#E0E0E0',        # Light gray
}


def load_raw_data(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load raw training data."""
    print("Loading raw training data...")
    df = pd.read_csv(RAW_DATA_DIR / "WADI_14days_new.csv", nrows=nrows)
    print(f"  Raw data shape: {df.shape}")
    return df


def load_preprocessed_data() -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load preprocessed training data and features."""
    print("Loading preprocessed data...")
    
    # Load windows and labels
    train_windows = np.load(PREPROCESSED_DIR / "train_windows.npy")
    train_labels = np.load(PREPROCESSED_DIR / "train_labels.npy")
    
    # Load feature names
    with open(PREPROCESSED_DIR / "features.json", 'r') as f:
        features_data = json.load(f)
        if 'final_features' in features_data:
            feature_names = features_data['final_features']
        else:
            feature_names = features_data.get('features', [])
    
    print(f"  Preprocessed windows shape: {train_windows.shape}")
    print(f"  Number of features: {len(feature_names)}")
    
    return train_windows, train_labels, {'features': feature_names}


def plot_data_distribution_comparison(raw_df: pd.DataFrame, 
                                     preprocessed_windows: np.ndarray,
                                     feature_idx: int = 0,
                                     raw_feature_name: str = None,
                                     save_path: Optional[Path] = None):
    """
    Compare distribution of a feature before and after preprocessing.
    
    Args:
        raw_df: Raw data DataFrame
        preprocessed_windows: Preprocessed windows array
        feature_idx: Index of feature in preprocessed data
        raw_feature_name: Name of feature in raw data (if None, use first numeric column)
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Get raw feature data
    if raw_feature_name is None:
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
        raw_feature_name = numeric_cols[3]  # Skip Row, Date, Time
    
    raw_data = raw_df[raw_feature_name].dropna()
    
    # Get preprocessed feature data (flatten all windows for this feature)
    processed_data = preprocessed_windows[:, :, feature_idx].flatten()
    
    # 1. Histogram comparison
    axes[0, 0].hist(raw_data, bins=50, alpha=0.7, color=COLORS['before'], 
                    label='Before Preprocessing', edgecolor='black')
    axes[0, 0].set_xlabel('Value', fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title('Distribution: Before Preprocessing', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(processed_data, bins=50, alpha=0.7, color=COLORS['after'], 
                    label='After Preprocessing', edgecolor='black')
    axes[0, 1].set_xlabel('Value', fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontweight='bold')
    axes[0, 1].set_title('Distribution: After Preprocessing', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    box_data = [raw_data, processed_data]
    bp = axes[1, 0].boxplot(box_data, labels=['Before', 'After'],
                            patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor(COLORS['before'])
    bp['boxes'][1].set_facecolor(COLORS['after'])
    axes[1, 0].set_ylabel('Value', fontweight='bold')
    axes[1, 0].set_title('Box Plot Comparison', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 3. Q-Q plot
    stats.probplot(processed_data, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (After Preprocessing)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Feature: {raw_feature_name}', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_statistics_comparison(raw_df: pd.DataFrame, 
                              preprocessed_windows: np.ndarray,
                              num_features: int = 10,
                              save_path: Optional[Path] = None):
    """
    Compare statistical properties before and after preprocessing.
    
    Args:
        raw_df: Raw data DataFrame
        preprocessed_windows: Preprocessed windows array
        num_features: Number of features to compare
        save_path: Path to save the figure
    """
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns[3:]  # Skip Row, Date, Time
    num_features = min(num_features, len(numeric_cols), preprocessed_windows.shape[2])
    
    # Calculate statistics
    stats_before = []
    stats_after = []
    feature_names = []
    
    for i in range(num_features):
        # Before preprocessing
        raw_data = raw_df[numeric_cols[i]].dropna()
        stats_before.append({
            'mean': raw_data.mean(),
            'std': raw_data.std(),
            'min': raw_data.min(),
            'max': raw_data.max(),
            'skew': stats.skew(raw_data),
            'kurt': stats.kurtosis(raw_data)
        })
        
        # After preprocessing
        processed_data = preprocessed_windows[:, :, i].flatten()
        stats_after.append({
            'mean': processed_data.mean(),
            'std': processed_data.std(),
            'min': processed_data.min(),
            'max': processed_data.max(),
            'skew': stats.skew(processed_data),
            'kurt': stats.kurtosis(processed_data)
        })
        
        feature_names.append(numeric_cols[i][:15])  # Truncate long names
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics = ['mean', 'std', 'min', 'max', 'skew', 'kurt']
    titles = ['Mean', 'Std Dev', 'Minimum', 'Maximum', 'Skewness', 'Kurtosis']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]
        
        before_vals = [s[metric] for s in stats_before]
        after_vals = [s[metric] for s in stats_after]
        
        x = np.arange(len(feature_names))
        width = 0.35
        
        ax.bar(x - width/2, before_vals, width, label='Before', 
               color=COLORS['before'], alpha=0.8, edgecolor='black')
        ax.bar(x + width/2, after_vals, width, label='After', 
               color=COLORS['after'], alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Features', fontweight='bold')
        ax.set_ylabel(title, fontweight='bold')
        ax.set_title(f'{title} Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Statistical Properties: Before vs After Preprocessing', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_correlation_matrix_comparison(raw_df: pd.DataFrame,
                                       preprocessed_windows: np.ndarray,
                                       num_features: int = 15,
                                       save_path: Optional[Path] = None):
    """
    Compare correlation matrices before and after preprocessing.
    
    Args:
        raw_df: Raw data DataFrame
        preprocessed_windows: Preprocessed windows array
        num_features: Number of features to include
        save_path: Path to save the figure
    """
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns[3:]
    num_features = min(num_features, len(numeric_cols), preprocessed_windows.shape[2])
    
    # Before preprocessing
    corr_before = raw_df[numeric_cols[:num_features]].corr()
    
    # After preprocessing
    processed_flat = preprocessed_windows[:1000, :, :num_features].reshape(-1, num_features)
    corr_after = pd.DataFrame(processed_flat).corr()
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Before
    sns.heatmap(corr_before, annot=False, cmap='coolwarm', center=0, 
                vmin=-1, vmax=1, square=True, ax=ax1, cbar_kws={'label': 'Correlation'})
    ax1.set_title('Correlation Matrix: Before Preprocessing', fontweight='bold', pad=15)
    
    # After
    sns.heatmap(corr_after, annot=False, cmap='coolwarm', center=0,
                vmin=-1, vmax=1, square=True, ax=ax2, cbar_kws={'label': 'Correlation'})
    ax2.set_title('Correlation Matrix: After Preprocessing', fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_time_series_comparison(raw_df: pd.DataFrame,
                                preprocessed_windows: np.ndarray,
                                feature_idx: int = 0,
                                raw_feature_name: str = None,
                                num_samples: int = 1000,
                                save_path: Optional[Path] = None):
    """
    Compare time series visualization before and after preprocessing.
    
    Args:
        raw_df: Raw data DataFrame
        preprocessed_windows: Preprocessed windows array
        feature_idx: Feature index to visualize
        raw_feature_name: Name of feature in raw data
        num_samples: Number of samples to plot
        save_path: Path to save the figure
    """
    if raw_feature_name is None:
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
        raw_feature_name = numeric_cols[3]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Before preprocessing
    raw_data = raw_df[raw_feature_name][:num_samples]
    ax1.plot(raw_data, color=COLORS['before'], linewidth=1, alpha=0.8)
    ax1.set_xlabel('Time (samples)', fontweight='bold')
    ax1.set_ylabel('Value', fontweight='bold')
    ax1.set_title(f'Time Series: Before Preprocessing - {raw_feature_name}', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # After preprocessing (flatten first window's time series)
    num_windows = min(num_samples // 100, preprocessed_windows.shape[0])
    processed_data = preprocessed_windows[:num_windows, :, feature_idx].flatten()
    ax2.plot(processed_data, color=COLORS['after'], linewidth=1, alpha=0.8)
    ax2.set_xlabel('Time (samples)', fontweight='bold')
    ax2.set_ylabel('Normalized Value', fontweight='bold')
    ax2.set_title(f'Time Series: After Preprocessing - Feature {feature_idx}', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_outlier_analysis(raw_df: pd.DataFrame,
                          preprocessed_windows: np.ndarray,
                          num_features: int = 8,
                          save_path: Optional[Path] = None):
    """
    Analyze outliers before and after preprocessing.
    
    Args:
        raw_df: Raw data DataFrame
        preprocessed_windows: Preprocessed windows array
        num_features: Number of features to analyze
        save_path: Path to save the figure
    """
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns[3:]
    num_features = min(num_features, len(numeric_cols), preprocessed_windows.shape[2])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calculate outlier percentages
    outlier_pcts_before = []
    outlier_pcts_after = []
    feature_names = []
    
    for i in range(num_features):
        # Before
        raw_data = raw_df[numeric_cols[i]].dropna()
        Q1, Q3 = raw_data.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers_before = ((raw_data < Q1 - 1.5*IQR) | (raw_data > Q3 + 1.5*IQR)).sum()
        outlier_pcts_before.append(100 * outliers_before / len(raw_data))
        
        # After
        processed_data = preprocessed_windows[:, :, i].flatten()
        Q1, Q3 = np.percentile(processed_data, [25, 75])
        IQR = Q3 - Q1
        outliers_after = ((processed_data < Q1 - 1.5*IQR) | (processed_data > Q3 + 1.5*IQR)).sum()
        outlier_pcts_after.append(100 * outliers_after / len(processed_data))
        
        feature_names.append(numeric_cols[i][:12])
    
    # Bar plot
    x = np.arange(len(feature_names))
    width = 0.35
    
    ax1.bar(x - width/2, outlier_pcts_before, width, label='Before',
            color=COLORS['before'], alpha=0.8, edgecolor='black')
    ax1.bar(x + width/2, outlier_pcts_after, width, label='After',
            color=COLORS['after'], alpha=0.8, edgecolor='black')
    
    ax1.set_xlabel('Features', fontweight='bold')
    ax1.set_ylabel('Outlier Percentage (%)', fontweight='bold')
    ax1.set_title('Outlier Analysis', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(feature_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Reduction comparison
    reduction = np.array(outlier_pcts_before) - np.array(outlier_pcts_after)
    colors_bar = [COLORS['after'] if r > 0 else COLORS['before'] for r in reduction]
    
    ax2.bar(x, reduction, color=colors_bar, alpha=0.8, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Features', fontweight='bold')
    ax2.set_ylabel('Outlier Reduction (%)', fontweight='bold')
    ax2.set_title('Outlier Reduction After Preprocessing', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(feature_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_data_quality_summary(raw_df: pd.DataFrame,
                              preprocessed_windows: np.ndarray,
                              save_path: Optional[Path] = None):
    """
    Create comprehensive data quality summary.
    
    Args:
        raw_df: Raw data DataFrame
        preprocessed_windows: Preprocessed windows array
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Missing values
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns[3:]
    missing_before = raw_df[numeric_cols].isnull().sum().sum()
    missing_after = 0  # Preprocessed data has no missing values
    
    axes[0, 0].bar(['Before', 'After'], [missing_before, missing_after],
                   color=[COLORS['before'], COLORS['after']], alpha=0.8, edgecolor='black')
    axes[0, 0].set_ylabel('Count', fontweight='bold')
    axes[0, 0].set_title('Missing Values', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Number of features
    features_before = len(numeric_cols)
    features_after = preprocessed_windows.shape[2]
    
    axes[0, 1].bar(['Before', 'After'], [features_before, features_after],
                   color=[COLORS['before'], COLORS['after']], alpha=0.8, edgecolor='black')
    axes[0, 1].set_ylabel('Count', fontweight='bold')
    axes[0, 1].set_title('Number of Features', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Data volume
    samples_before = len(raw_df)
    samples_after = preprocessed_windows.shape[0] * preprocessed_windows.shape[1]
    
    axes[1, 0].bar(['Before', 'After'], [samples_before, samples_after],
                   color=[COLORS['before'], COLORS['after']], alpha=0.8, edgecolor='black')
    axes[1, 0].set_ylabel('Samples', fontweight='bold')
    axes[1, 0].set_title('Data Volume', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 4. Data shape info
    info_text = f"""
    BEFORE PREPROCESSING:
    • Shape: {raw_df.shape}
    • Features: {features_before}
    • Missing: {missing_before}
    • Duration: ~14 days
    
    AFTER PREPROCESSING:
    • Windows: {preprocessed_windows.shape[0]:,}
    • Window size: {preprocessed_windows.shape[1]}
    • Features: {features_after}
    • Missing: 0
    """
    
    axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Data Summary', fontweight='bold')
    
    plt.suptitle('Data Quality: Before vs After Preprocessing', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def main():
    """Generate thesis-quality visualizations."""
    
    print("\n" + "="*60)
    print("THESIS-QUALITY VISUALIZATIONS")
    print("="*60 + "\n")
    
    try:
        # Load data (use subset for speed)
        print("Loading data...")
        raw_df = load_raw_data(nrows=20000)  # First ~5.5 hours
        preprocessed_windows, preprocessed_labels, features_info = load_preprocessed_data()
        
        print("\n" + "-"*60)
        print("Generating thesis visualizations...")
        print("-"*60 + "\n")
        
        generated_files = []
        
        # 1. Data quality summary (most important for thesis)
        try:
            print("1. Data quality summary...")
            plot_data_quality_summary(
                raw_df, preprocessed_windows,
                save_path=VISUALS_DIR / "thesis_data_quality.png"
            )
            generated_files.append("thesis_data_quality.png")
        except Exception as e:
            print(f"  Warning: Could not generate data quality summary: {e}")
        
        # 2. Distribution comparison (key for showing preprocessing effect)
        try:
            print("\n2. Distribution comparison...")
            plot_data_distribution_comparison(
                raw_df, preprocessed_windows, feature_idx=0,
                save_path=VISUALS_DIR / "thesis_distribution.png"
            )
            generated_files.append("thesis_distribution.png")
        except Exception as e:
            print(f"  Warning: Could not generate distribution comparison: {e}")
        
        # 3. Time series comparison (visual impact)
        try:
            print("\n3. Time series comparison...")
            plot_time_series_comparison(
                raw_df, preprocessed_windows, feature_idx=0, num_samples=1000,
                save_path=VISUALS_DIR / "thesis_timeseries.png"
            )
            generated_files.append("thesis_timeseries.png")
        except Exception as e:
            print(f"  Warning: Could not generate time series: {e}")
        
        # 4. Outlier analysis (shows preprocessing effectiveness)
        try:
            print("\n4. Outlier reduction analysis...")
            plot_outlier_analysis(
                raw_df, preprocessed_windows, num_features=6,
                save_path=VISUALS_DIR / "thesis_outlier_reduction.png"
            )
            generated_files.append("thesis_outlier_reduction.png")
        except Exception as e:
            print(f"  Warning: Could not generate outlier analysis: {e}")
        
        # 5. Statistics comparison (for detailed analysis section)
        try:
            print("\n5. Statistical comparison...")
            plot_statistics_comparison(
                raw_df, preprocessed_windows, num_features=8,
                save_path=VISUALS_DIR / "thesis_statistics.png"
            )
            generated_files.append("thesis_statistics.png")
        except Exception as e:
            print(f"  Warning: Could not generate statistics: {e}")
        
        print("\n" + "="*60)
        print(f"✓ Generated {len(generated_files)} thesis visualizations!")
        print(f"✓ Saved to: {VISUALS_DIR}")
        print("="*60 + "\n")
        
        if generated_files:
            print("Generated files:")
            for file in generated_files:
                print(f"  • {file}")
        else:
            print("⚠ No files were generated. Please check the errors above.")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure:")
        print("  1. Preprocessed data exists in data/preprocessed/")
        print("  2. Raw data exists in datasets/")
        print("  3. All required packages are installed")


if __name__ == "__main__":
    main()
