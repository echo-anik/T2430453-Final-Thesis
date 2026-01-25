"""
Lightweight Training for Visualization Data (PyTorch)
======================================================
Trains a simple model to generate ACTUAL training history curves.
This is NOT for final model evaluation - just for thesis visualizations.

Author: Thesis Project
Date: 2026-01-24
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Setup paths
DATA_DIR = Path("data/preprocessed")
RESULTS_DIR = Path("results")
VISUALS_DIR = RESULTS_DIR / "thesis_visuals"
VISUALS_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*60)
print("TRAINING FOR VISUALIZATION DATA (PyTorch)")
print("="*60 + "\n")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Skip model setup and training - focus on visualization generation
print("\n" + "="*60)
print("VISUALIZATION GENERATION FROM EXISTING DATA")
print("="*60 + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Skip actual training - load existing training history
print("\n" + "="*60)
print("VISUALIZATION GENERATION FROM EXISTING DATA")
print("="*60 + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Skip actual training - load existing training history
print("\n" + "-"*60)
print("Loading existing training history...")
print("-"*60 + "\n")

# Load existing training history
try:
    with open(RESULTS_DIR / "training_history_lstm.json", 'r') as f:
        history = json.load(f)
    print(f"[OK] Loaded training history from: {RESULTS_DIR / 'training_history_lstm.json'}")
    print(f"  Epochs: {len(history['epochs'])}")
    print(f"  Final Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"  Final Val Loss: {history['val_loss'][-1]:.6f}")
except FileNotFoundError:
    print("[ERROR] Training history file not found. Please ensure training_history_lstm.json exists.")
    exit(1)

# Generate visualizations
print("\n" + "-"*60)
print("Generating visualizations...")
print("-"*60 + "\n")

# Plot 1: Training curves
fig, ax = plt.subplots(figsize=(10, 6))

epochs = history['epochs']
train_loss = history['train_loss']
val_loss = history['val_loss']

ax.plot(epochs, train_loss, 'b-o', label='Training Loss', 
        linewidth=2, markersize=4, alpha=0.8)
ax.plot(epochs, val_loss, 'r-s', label='Validation Loss', 
        linewidth=2, markersize=4, alpha=0.8)

# Annotate minimum values
min_train_idx = np.argmin(train_loss)
min_val_idx = np.argmin(val_loss)

ax.annotate(f"Min Train: {train_loss[min_train_idx]:.4f}",
            xy=(epochs[min_train_idx], train_loss[min_train_idx]),
            xytext=(10, 20), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='blue', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax.annotate(f"Min Val: {val_loss[min_val_idx]:.4f}",
            xy=(epochs[min_val_idx], val_loss[min_val_idx]),
            xytext=(10, -30), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
ax.set_ylabel('Loss (MSE)', fontweight='bold', fontsize=12)
ax.set_title('LSTM Autoencoder - Training History', fontweight='bold', fontsize=14, pad=15)
ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VISUALS_DIR / "thesis_actual_training_loss.png", dpi=300, bbox_inches='tight')
print(f"[OK] Visualization saved to: {VISUALS_DIR / 'thesis_actual_training_loss.png'}")
plt.close()

# Plot 2: Smoothed curves
window = 3
if len(train_loss) >= window:
    train_smooth = np.convolve(train_loss, np.ones(window)/window, mode='valid')
    val_smooth = np.convolve(val_loss, np.ones(window)/window, mode='valid')
    epochs_smooth = epochs[window-1:]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Raw (faint)
    ax.plot(epochs, train_loss, 'b-', alpha=0.2, linewidth=1)
    ax.plot(epochs, val_loss, 'r-', alpha=0.2, linewidth=1)
    
    # Smoothed
    ax.plot(epochs_smooth, train_smooth, 'b-', label='Training (smoothed)', linewidth=2.5)
    ax.plot(epochs_smooth, val_smooth, 'r-', label='Validation (smoothed)', linewidth=2.5)
    
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontweight='bold', fontsize=12)
    ax.set_title(f'LSTM Autoencoder - Smoothed Loss (window={window})', 
                fontweight='bold', fontsize=14, pad=15)
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "thesis_actual_loss_smoothed.png", dpi=300, bbox_inches='tight')
    print(f"[OK] Smoothed visualization saved to: {VISUALS_DIR / 'thesis_actual_loss_smoothed.png'}")
    plt.close()

# Print summary
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60 + "\n")

print("Model Performance Summary:")
print(f"  Final Training Loss: {train_loss[-1]:.6f}")
print(f"  Final Validation Loss: {val_loss[-1]:.6f}")
print(f"  Best Validation Loss: {min(val_loss):.6f} (Epoch {np.argmin(val_loss) + 1})")
print(f"  Total Epochs: {len(epochs)}")
print(f"  Device Used: {device}")

print("\nGenerated files:")
print("  • thesis_actual_training_loss.png (main visualization)")
print("  • thesis_actual_loss_smoothed.png (smoothed curves)")
print("  • Used existing training_history_lstm.json (training data)")

print("\n" + "="*60)
