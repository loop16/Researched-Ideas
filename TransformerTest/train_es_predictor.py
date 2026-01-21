#!/usr/bin/env python3
"""
Training Script for ES Futures Price Prediction Model

This script orchestrates the training pipeline:
1. Load and preprocess data
2. Create sequences for prediction
3. Build and train Transformer model
4. Evaluate and visualize results
5. Save model and artifacts

Usage:
    python train_es_predictor.py --scenario 1 --epochs 50
    python train_es_predictor.py --scenario 2 --epochs 50 --test-run
"""

import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# Import our custom modules
from data_loader import ESDataLoader
from es_transformer_model import (
    build_es_model, 
    build_es_model_single_output,
    compile_model,
    MaskedMSELoss
)


def setup_gpu(use_cpu: bool = False):
    """Configure GPU memory growth to avoid OOM errors."""
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.set_visible_devices([], 'GPU')
        print("✓ Using CPU only (GPU disabled)")
        return
        
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU configured: {len(gpus)} GPU(s) available")
        except RuntimeError as e:
            print(f"✗ GPU setup error: {e}")
    else:
        print("✗ No GPU detected, using CPU")


def create_callbacks(logdir: str, 
                     patience: int = 15,
                     model_name: str = 'es_transformer') -> list:
    """
    Create training callbacks.
    
    Args:
        logdir: Directory for logs and checkpoints
        patience: Early stopping patience
        model_name: Name for saved model
        
    Returns:
        List of Keras callbacks
    """
    callbacks = []
    
    # TensorBoard logging
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch',
        profile_batch=0  # Disable profiling for speed
    )
    callbacks.append(tensorboard_cb)
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Model checkpoint
    checkpoint_path = os.path.join(logdir, f'{model_name}_best.keras')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Learning rate reduction
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(lr_reducer)
    
    return callbacks


def plot_training_history(history, logdir: str):
    """
    Plot and save training history.
    
    Args:
        history: Keras training history object
        logdir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train')
    axes[0].plot(history.history['val_loss'], label='Validation')
    axes[0].set_title('Loss (MSE)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    if 'mae' in history.history:
        axes[1].plot(history.history['mae'], label='Train')
        axes[1].plot(history.history['val_mae'], label='Validation')
        axes[1].set_title('Mean Absolute Error')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # RMSE
    if 'rmse' in history.history:
        axes[2].plot(history.history['rmse'], label='Train')
        axes[2].plot(history.history['val_rmse'], label='Validation')
        axes[2].set_title('Root Mean Squared Error')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('RMSE')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, 'training_history.png'), dpi=150)
    plt.close()
    print(f"✓ Training history plot saved")


def plot_predictions(y_true: np.ndarray, 
                     y_pred: np.ndarray, 
                     logdir: str,
                     num_samples: int = 5):
    """
    Plot sample predictions vs actual values.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        logdir: Directory to save plots
        num_samples: Number of sample predictions to plot
    """
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    
    # Select random samples
    indices = np.random.choice(len(y_true), size=min(num_samples, len(y_true)), replace=False)
    
    for i, idx in enumerate(indices):
        ax = axes[i] if num_samples > 1 else axes
        
        true_seq = y_true[idx]
        pred_seq = y_pred[idx]
        
        # Filter out NaN indicator values
        valid_mask = true_seq != -999.0
        
        ax.plot(true_seq[valid_mask], 'b-', label='Actual', linewidth=2)
        ax.plot(pred_seq[valid_mask], 'r--', label='Predicted', linewidth=2)
        ax.set_title(f'Sample {idx}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('close_ar_pct (normalized)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, 'sample_predictions.png'), dpi=150)
    plt.close()
    print(f"✓ Sample predictions plot saved")


def plot_scatter(y_true: np.ndarray, 
                 y_pred: np.ndarray, 
                 logdir: str):
    """
    Plot scatter plot of predictions vs actuals.
    
    Args:
        y_true: True values (num_samples, seq_len)
        y_pred: Predicted values (num_samples, seq_len)
        logdir: Directory to save plot
    """
    # Calculate directional accuracy first (before flattening)
    directional_acc = calculate_directional_accuracy(y_true, y_pred)
    
    # Flatten and filter out NaN indicators
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    valid_mask = y_true_flat != -999.0
    y_true_valid = y_true_flat[valid_mask]
    y_pred_valid = y_pred_flat[valid_mask]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true_valid, y_pred_valid, alpha=0.3, s=5)
    
    # Add diagonal line
    min_val = min(y_true_valid.min(), y_pred_valid.min())
    max_val = max(y_true_valid.max(), y_pred_valid.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate correlation
    correlation = np.corrcoef(y_true_valid, y_pred_valid)[0, 1]
    
    ax.set_xlabel('Actual close_ar_pct')
    ax.set_ylabel('Predicted close_ar_pct')
    ax.set_title(f'Correlation: {correlation:.4f} | Directional Accuracy: {directional_acc:.1f}%')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, 'prediction_scatter.png'), dpi=150)
    plt.close()
    print(f"✓ Scatter plot saved (Correlation: {correlation:.4f}, Directional Accuracy: {directional_acc:.1f}%)")


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy: does predicted direction match actual?
    Direction = whether last candle > first candle in the sequence.
    
    Args:
        y_true: True values (num_samples, seq_len)
        y_pred: Predicted values (num_samples, seq_len)
        
    Returns:
        Directional accuracy as percentage
    """
    correct = 0
    total = 0
    
    for i in range(len(y_true)):
        # Get first and last valid values
        true_seq = y_true[i]
        pred_seq = y_pred[i]
        
        # Find valid (non-NaN indicator) positions
        valid_mask = true_seq != -999.0
        if np.sum(valid_mask) < 2:
            continue
            
        # Get first and last valid values
        valid_indices = np.where(valid_mask)[0]
        first_idx, last_idx = valid_indices[0], valid_indices[-1]
        
        # Actual direction: last > first
        actual_direction = true_seq[last_idx] > true_seq[first_idx]
        
        # Predicted direction: last > first  
        pred_direction = pred_seq[last_idx] > pred_seq[first_idx]
        
        if actual_direction == pred_direction:
            correct += 1
        total += 1
    
    if total == 0:
        return 0.0
        
    return (correct / total) * 100.0


def train_scenario1(args):
    """
    Train model for Scenario 1: Predict TR session.
    
    Input: 9:30 AM to next day 8:25 AM
    Target: TR session (08:30-09:25) close_ar_pct path
    """
    print("\n" + "=" * 70)
    print("SCENARIO 1: Predict TR Session (08:30-09:25)")
    print("=" * 70)
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("logs", f"scenario1_{timestamp}")
    os.makedirs(logdir, exist_ok=True)
    print(f"✓ Log directory: {logdir}")
    
    # Load data
    print("\n[1/5] Loading and preprocessing data...")
    loader = ESDataLoader()
    loader.fit_transform()
    
    # Create sequences
    print("\n[2/5] Creating sequences...")
    max_samples = 500 if args.test_run else None
    X, y, dates = loader.create_sequences_scenario1(max_samples=max_samples)
    
    if len(X) == 0:
        print("✗ No valid sequences created. Check data.")
        return
    
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Features: {len(loader.all_features)}")
    
    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = loader.split_data(
        X, y, dates,
        train_end_date='2020-12-31',
        val_end_date='2022-12-31'
    )
    
    # Build model
    print("\n[3/5] Building model...")
    seq_length = X.shape[1]
    num_features = X.shape[2]
    target_length = y.shape[1]
    
    model = build_es_model(
        seq_length=seq_length,
        num_features=num_features,
        target_length=target_length,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        dropout_rate=args.dropout
    )
    model = compile_model(model, learning_rate=args.learning_rate)
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks(logdir, patience=args.patience, model_name='scenario1')
    
    # Train model
    print("\n[4/5] Training model...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  TensorBoard: tensorboard --logdir {logdir}")
    
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n[5/5] Evaluating model...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss: {test_results[0]:.6f}")
    print(f"  Test MAE: {test_results[1]:.6f}")
    print(f"  Test RMSE: {test_results[2]:.6f}")
    
    # Generate predictions for visualization
    y_pred_test = model.predict(X_test, verbose=0)
    
    # Create plots
    plot_training_history(history, logdir)
    plot_predictions(y_test, y_pred_test, logdir)
    plot_scatter(y_test, y_pred_test, logdir)
    
    # Save final model
    model.save(os.path.join(logdir, 'scenario1_final.keras'))
    print(f"\n✓ Model saved to {logdir}")
    
    return model, history


def train_scenario2(args):
    """
    Train model for Scenario 2: Predict RS session.
    
    Input: 9:30 AM to next day 10:25 AM
    Target: RS session (10:30-15:55) close_ar_pct path
    """
    print("\n" + "=" * 70)
    print("SCENARIO 2: Predict RS Session (10:30-15:55)")
    print("=" * 70)
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("logs", f"scenario2_{timestamp}")
    os.makedirs(logdir, exist_ok=True)
    print(f"✓ Log directory: {logdir}")
    
    # Load data
    print("\n[1/5] Loading and preprocessing data...")
    loader = ESDataLoader()
    loader.fit_transform()
    
    # Create sequences
    print("\n[2/5] Creating sequences...")
    max_samples = 500 if args.test_run else None
    X, y, dates = loader.create_sequences_scenario2(max_samples=max_samples)
    
    if len(X) == 0:
        print("✗ No valid sequences created. Check data.")
        return
    
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Features: {len(loader.all_features)}")
    
    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = loader.split_data(
        X, y, dates,
        train_end_date='2020-12-31',
        val_end_date='2022-12-31'
    )
    
    # Build model
    print("\n[3/5] Building model...")
    seq_length = X.shape[1]
    num_features = X.shape[2]
    target_length = y.shape[1]
    
    model = build_es_model(
        seq_length=seq_length,
        num_features=num_features,
        target_length=target_length,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        dropout_rate=args.dropout
    )
    model = compile_model(model, learning_rate=args.learning_rate)
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks(logdir, patience=args.patience, model_name='scenario2')
    
    # Train model
    print("\n[4/5] Training model...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  TensorBoard: tensorboard --logdir {logdir}")
    
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n[5/5] Evaluating model...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss: {test_results[0]:.6f}")
    print(f"  Test MAE: {test_results[1]:.6f}")
    print(f"  Test RMSE: {test_results[2]:.6f}")
    
    # Generate predictions for visualization
    y_pred_test = model.predict(X_test, verbose=0)
    
    # Create plots
    plot_training_history(history, logdir)
    plot_predictions(y_test, y_pred_test, logdir)
    plot_scatter(y_test, y_pred_test, logdir)
    
    # Save final model
    model.save(os.path.join(logdir, 'scenario2_final.keras'))
    print(f"\n✓ Model saved to {logdir}")
    
    return model, history


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train ES Futures Transformer Prediction Model'
    )
    
    # Scenario selection
    parser.add_argument('--scenario', type=int, default=1, choices=[1, 2],
                        help='Prediction scenario (1=TR session, 2=RS session)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    
    # Model architecture
    parser.add_argument('--embed-dim', type=int, default=256,
                        help='Transformer embedding dimension')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--ff-dim', type=int, default=1024,
                        help='Feed-forward network dimension')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Testing mode
    parser.add_argument('--test-run', action='store_true',
                        help='Run with small dataset for testing')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU only (disable GPU)')
    
    args = parser.parse_args()
    
    # Setup
    setup_gpu(use_cpu=args.cpu)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run training
    if args.scenario == 1:
        train_scenario1(args)
    else:
        train_scenario2(args)
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
