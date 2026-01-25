"""
Lightweight Training for Visualization Data
============================================
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

# Setup paths
DATA_DIR = Path("data/preprocessed")
RESULTS_DIR = Path("results")
VISUALS_DIR = RESULTS_DIR / "thesis_visuals"
VISUALS_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*60)
print("TRAINING FOR VISUALIZATION DATA")
print("="*60 + "\n")

# Load preprocessed data
print("Loading preprocessed data...")
train_windows = np.load(DATA_DIR / "train_windows.npy")
train_labels = np.load(DATA_DIR / "train_labels.npy")
val_windows = np.load(DATA_DIR / "val_windows.npy")
val_labels = np.load(DATA_DIR / "val_labels.npy")

print(f"  Train: {train_windows.shape}")
print(f"  Val: {val_windows.shape}")

# Use subset for faster training
n_train_samples = min(50000, len(train_windows))
n_val_samples = min(10000, len(val_windows))

X_train = train_windows[:n_train_samples]
y_train = train_labels[:n_train_samples]
X_val = val_windows[:n_val_samples]
y_val = val_labels[:n_val_samples]

print(f"\nUsing subset for speed:")
print(f"  Train samples: {len(X_train)}")
print(f"  Val samples: {len(X_val)}")

# Check if TensorFlow/Keras is available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    print(f"\nTensorFlow version: {tf.__version__}")
    
    # Build simple LSTM Autoencoder
    print("\nBuilding LSTM Autoencoder...")
    
    window_size = X_train.shape[1]
    n_features = X_train.shape[2]
    
    model = keras.Sequential([
        layers.Input(shape=(window_size, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.RepeatVector(window_size),
        layers.LSTM(32, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=True),
        layers.TimeDistributed(layers.Dense(n_features))
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print(model.summary())
    
    # Train model
    print("\n" + "-"*60)
    print("Training model (this may take a few minutes)...")
    print("-"*60 + "\n")
    
    history = model.fit(
        X_train, X_train,  # Autoencoder reconstructs input
        validation_data=(X_val, X_val),
        epochs=30,
        batch_size=256,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
    )
    
    # Save training history
    print("\n" + "-"*60)
    print("Saving training history...")
    print("-"*60 + "\n")
    
    history_data = {
        'epochs': list(range(1, len(history.history['loss']) + 1)),
        'train_loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'train_mae': [float(x) for x in history.history['mae']],
        'val_mae': [float(x) for x in history.history['val_mae']],
    }
    
    # Save to JSON
    with open(RESULTS_DIR / "training_history_lstm.json", 'w') as f:
        json.dump(history_data, f, indent=2)
    
    print(f"✓ Training history saved to: {RESULTS_DIR / 'training_history_lstm.json'}")
    
    # Generate visualizations
    print("\n" + "-"*60)
    print("Generating visualizations...")
    print("-"*60 + "\n")
    
    # Plot 1: Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = history_data['epochs']
    
    # Loss
    ax1.plot(epochs, history_data['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history_data['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss (MSE)', fontweight='bold')
    ax1.set_title('LSTM Autoencoder - Training History', fontweight='bold', pad=15)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE
    ax2.plot(epochs, history_data['train_mae'], 'b-', label='Training MAE', linewidth=2)
    ax2.plot(epochs, history_data['val_mae'], 'r-', label='Validation MAE', linewidth=2)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('MAE', fontweight='bold')
    ax2.set_title('Mean Absolute Error', fontweight='bold', pad=15)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "actual_training_history.png", dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {VISUALS_DIR / 'actual_training_history.png'}")
    plt.close()
    
    # Plot 2: Loss with min annotations
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, history_data['train_loss'], 'b-o', label='Training Loss', 
            linewidth=2, markersize=4, alpha=0.8)
    ax.plot(epochs, history_data['val_loss'], 'r-s', label='Validation Loss', 
            linewidth=2, markersize=4, alpha=0.8)
    
    # Annotate minimum values
    min_train_idx = np.argmin(history_data['train_loss'])
    min_val_idx = np.argmin(history_data['val_loss'])
    
    ax.annotate(f"Min Train: {history_data['train_loss'][min_train_idx]:.4f}",
                xy=(epochs[min_train_idx], history_data['train_loss'][min_train_idx]),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='blue', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.annotate(f"Min Val: {history_data['val_loss'][min_val_idx]:.4f}",
                xy=(epochs[min_val_idx], history_data['val_loss'][min_val_idx]),
                xytext=(10, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontweight='bold')
    ax.set_title('LSTM Autoencoder - Loss Convergence', fontweight='bold', pad=15)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "thesis_actual_loss_annotated.png", dpi=300, bbox_inches='tight')
    print(f"✓ Annotated loss saved to: {VISUALS_DIR / 'thesis_actual_loss_annotated.png'}")
    plt.close()
    
    # Save model summary
    print("\nModel Performance Summary:")
    print(f"  Final Training Loss: {history_data['train_loss'][-1]:.6f}")
    print(f"  Final Validation Loss: {history_data['val_loss'][-1]:.6f}")
    print(f"  Best Validation Loss: {min(history_data['val_loss']):.6f} (Epoch {min_val_idx + 1})")
    print(f"  Total Epochs: {len(epochs)}")
    
    # Optionally save model
    model_path = RESULTS_DIR / "models"
    model_path.mkdir(exist_ok=True)
    model.save(model_path / "lstm_autoencoder_viz.keras")
    print(f"\n✓ Model saved to: {model_path / 'lstm_autoencoder_viz.keras'}")
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60 + "\n")
    print("Generated files:")
    print("  • training_history_lstm.json (raw data)")
    print("  • actual_training_history.png (visualization)")
    print("  • thesis_actual_loss_annotated.png (annotated)")
    print("  • lstm_autoencoder_viz.keras (trained model)")
    
except ImportError:
    print("\n[X] TensorFlow not found!")
    print("\nTo install TensorFlow, run:")
    print("  pip install tensorflow")
    print("\nOr for GPU support:")
    print("  pip install tensorflow[and-cuda]")
    
    print("\nAlternatively, install PyTorch:")
    print("  pip install torch")
    
except Exception as e:
    print(f"\n[X] Error during training: {e}")
    import traceback
    traceback.print_exc()
    
    print("\nIf you encounter memory issues, try:")
    print("  - Reducing n_train_samples in the script")
    print("  - Reducing batch_size")
    print("  - Using a simpler model architecture")
