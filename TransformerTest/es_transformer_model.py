#!/usr/bin/env python3
"""
ES Futures Transformer Model for Price Path Prediction

This module implements a Transformer-based model adapted from main_commented.py
to handle multi-feature time series input and predict price paths (sequence-to-sequence).

Key modifications from base model:
1. Multi-feature input (100+ features per timestep)
2. Sequence-to-sequence output (predict price path, not single value)
3. Positional encoding for time awareness
4. Handles variable-length sequences with masking
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Dense, LayerNormalization, Dropout, 
    GlobalAveragePooling1D, Masking
)
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from typing import Tuple, Optional


class PositionalEncoding(Layer):
    """
    Adds positional information to the input embeddings.
    
    Transformers have no inherent notion of position/order in a sequence.
    Positional encoding adds this information using sinusoidal functions,
    allowing the model to learn patterns based on position in the sequence.
    
    The encoding uses sin/cos functions at different frequencies:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, max_seq_length: int, embed_dim: int, **kwargs):
        """
        Initialize positional encoding.
        
        Args:
            max_seq_length: Maximum sequence length to support
            embed_dim: Embedding dimension (must match transformer embed_dim)
        """
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        
        # Pre-compute positional encodings
        self.pos_encoding = self._compute_positional_encoding()
        
    def _compute_positional_encoding(self) -> tf.Tensor:
        """Compute the positional encoding matrix."""
        positions = np.arange(self.max_seq_length)[:, np.newaxis]
        dims = np.arange(self.embed_dim)[np.newaxis, :]
        
        # Compute angles
        angles = positions / np.power(10000, (2 * (dims // 2)) / self.embed_dim)
        
        # Apply sin to even indices, cos to odd indices
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        
        # Add batch dimension
        pos_encoding = angles[np.newaxis, :, :]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        """
        Add positional encoding to inputs.
        
        Args:
            inputs: Tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_seq_length': self.max_seq_length,
            'embed_dim': self.embed_dim
        })
        return config


class MultiHeadSelfAttention(Layer):
    """
    Multi-Head Self-Attention mechanism.
    
    Allows the model to jointly attend to information from different
    representation subspaces at different positions.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, **kwargs):
        """
        Initialize Multi-Head Self-Attention.
        
        Args:
            embed_dim: Dimensionality of input/output embeddings
            num_heads: Number of parallel attention heads
        """
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        
        # Q, K, V projection layers
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        
        # Output projection
        self.combine_heads = Dense(embed_dim)
        
    def attention(self, query, key, value, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor (batch, heads, seq_len, depth)
            key: Key tensor (batch, heads, seq_len, depth)
            value: Value tensor (batch, heads, seq_len, depth)
            mask: Optional mask tensor
            
        Returns:
            Attention output and weights
        """
        # Compute attention scores
        score = tf.matmul(query, key, transpose_b=True)
        
        # Scale scores
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        
        # Apply mask if provided (for padding)
        if mask is not None:
            scaled_score += (mask * -1e9)
        
        # Softmax to get attention weights
        weights = tf.nn.softmax(scaled_score, axis=-1)
        
        # Apply attention to values
        output = tf.matmul(weights, value)
        
        return output, weights
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, projection_dim)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs, mask=None):
        """
        Forward pass of Multi-Head Self-Attention.
        
        Args:
            inputs: Input tensor (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch_size, seq_len, embed_dim)
        """
        batch_size = tf.shape(inputs)[0]
        
        # Project to Q, K, V
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        # Split into multiple heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # Compute attention
        attention_output, _ = self.attention(query, key, value, mask)
        
        # Concatenate heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.embed_dim))
        
        # Final projection
        return self.combine_heads(concat_attention)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads
        })
        return config


class TransformerBlock(Layer):
    """
    Single Transformer encoder block.
    
    Components:
    1. Multi-Head Self-Attention + Add & Norm
    2. Feed-Forward Network + Add & Norm
    """
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, 
                 dropout_rate: float = 0.1, **kwargs):
        """
        Initialize Transformer Block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward network inner dimension
            dropout_rate: Dropout rate for regularization
        """
        super(TransformerBlock, self).__init__(**kwargs)
        
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu"),  # GELU often works better than ReLU
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        
    def call(self, inputs, training=False, mask=None):
        """
        Forward pass of Transformer Block.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            mask: Optional attention mask
            
        Returns:
            Output tensor
        """
        # Multi-head attention with residual connection
        attn_output = self.att(inputs, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        return config


class TransformerEncoder(Layer):
    """
    Stack of Transformer encoder blocks.
    """
    
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int, 
                 ff_dim: int, max_seq_length: int, dropout_rate: float = 0.1, **kwargs):
        """
        Initialize Transformer Encoder.
        
        Args:
            num_layers: Number of transformer blocks
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            max_seq_length: Maximum sequence length
            dropout_rate: Dropout rate
        """
        super(TransformerEncoder, self).__init__(**kwargs)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(max_seq_length, embed_dim)
        
        # Stack of transformer blocks
        self.enc_layers = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.dropout = Dropout(dropout_rate)
        
    def call(self, inputs, training=False, mask=None):
        """
        Forward pass through encoder.
        
        Args:
            inputs: Input tensor (batch, seq_len, embed_dim)
            training: Whether in training mode
            mask: Optional attention mask
            
        Returns:
            Encoded output tensor
        """
        # Add positional encoding
        x = self.pos_encoding(inputs)
        x = self.dropout(x, training=training)
        
        # Pass through transformer blocks
        for layer in self.enc_layers:
            x = layer(x, training=training, mask=mask)
            
        return x


def build_es_model(
    seq_length: int,
    num_features: int,
    target_length: int,
    embed_dim: int = 256,
    num_heads: int = 8,
    ff_dim: int = 1024,
    num_layers: int = 4,
    dropout_rate: float = 0.1,
    nan_indicator: float = -999.0
) -> Model:
    """
    Build a Transformer model for ES futures price path prediction.
    
    Architecture:
    1. Input projection: (seq_len, num_features) -> (seq_len, embed_dim)
    2. Transformer Encoder: Captures temporal patterns
    3. Output projection: (seq_len, embed_dim) -> (target_length,) via pooling + dense
    
    Args:
        seq_length: Length of input sequence (number of candles)
        num_features: Number of features per timestep
        target_length: Length of output sequence (price path)
        embed_dim: Transformer embedding dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward network dimension
        num_layers: Number of transformer blocks
        dropout_rate: Dropout rate
        nan_indicator: Value used to indicate NaN/missing data
        
    Returns:
        Compiled Keras Model
    """
    # Input layer
    inputs = Input(shape=(seq_length, num_features), name='input_features')
    
    # Create mask for NaN values (where all features are NaN indicator)
    # This helps the model ignore padded positions
    mask_value = nan_indicator
    
    # Initial projection to embedding dimension
    x = Dense(embed_dim, name='input_projection')(inputs)
    
    # Add masking layer to handle padded sequences
    # The masking will propagate through the network
    
    # Transformer encoder
    encoder = TransformerEncoder(
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        max_seq_length=seq_length,
        dropout_rate=dropout_rate
    )
    x = encoder(x)  # Shape: (batch, seq_len, embed_dim)
    
    # Aggregate temporal information for sequence output
    # Option 1: Global average pooling + expand to target length
    pooled = GlobalAveragePooling1D()(x)  # (batch, embed_dim)
    
    # Option 2: Use last N timesteps for prediction
    # This can be done with x[:, -N:, :] but we'll use pooling for simplicity
    
    # Expand to target sequence length
    x = Dense(ff_dim, activation='gelu', name='expand_1')(pooled)
    x = Dropout(dropout_rate)(x)
    x = Dense(target_length * 64, activation='gelu', name='expand_2')(x)
    x = tf.keras.layers.Reshape((target_length, 64))(x)  # (batch, target_len, 64)
    
    # Final projection to single value per timestep (close_ar_pct)
    x = Dense(1, name='output_projection')(x)  # (batch, target_len, 1)
    outputs = tf.keras.layers.Reshape((target_length,))(x)  # (batch, target_len)
    
    model = Model(inputs=inputs, outputs=outputs, name='ES_Transformer')
    
    return model


def build_es_model_single_output(
    seq_length: int,
    num_features: int,
    embed_dim: int = 256,
    num_heads: int = 8,
    ff_dim: int = 1024,
    num_layers: int = 4,
    dropout_rate: float = 0.1
) -> Model:
    """
    Build a Transformer model for single value prediction (e.g., final close_ar_pct).
    
    This is a simpler model that predicts a single value instead of a sequence.
    Useful for predicting just the final price level.
    
    Args:
        seq_length: Length of input sequence
        num_features: Number of features per timestep
        embed_dim: Transformer embedding dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward network dimension
        num_layers: Number of transformer blocks
        dropout_rate: Dropout rate
        
    Returns:
        Compiled Keras Model
    """
    # Input layer
    inputs = Input(shape=(seq_length, num_features), name='input_features')
    
    # Initial projection
    x = Dense(embed_dim, name='input_projection')(inputs)
    
    # Transformer encoder
    encoder = TransformerEncoder(
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        max_seq_length=seq_length,
        dropout_rate=dropout_rate
    )
    x = encoder(x)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Output projection
    x = Dense(ff_dim // 2, activation='gelu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='ES_Transformer_Single')
    
    return model


class MaskedMSELoss(tf.keras.losses.Loss):
    """
    Custom MSE loss that ignores NaN indicator values.
    """
    
    def __init__(self, nan_indicator: float = -999.0, name='masked_mse'):
        super().__init__(name=name)
        self.nan_indicator = nan_indicator
        
    def call(self, y_true, y_pred):
        """
        Compute MSE only on valid (non-NaN) positions.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Masked MSE loss
        """
        # Create mask for valid values
        mask = tf.not_equal(y_true, self.nan_indicator)
        mask = tf.cast(mask, tf.float32)
        
        # Compute squared error
        squared_error = tf.square(y_true - y_pred)
        
        # Apply mask and compute mean
        masked_error = squared_error * mask
        
        # Avoid division by zero
        num_valid = tf.maximum(tf.reduce_sum(mask), 1.0)
        
        return tf.reduce_sum(masked_error) / num_valid


def compile_model(model: Model, 
                  learning_rate: float = 1e-4,
                  nan_indicator: float = -999.0,
                  clipnorm: float = 1.0) -> Model:
    """
    Compile the model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        nan_indicator: NaN indicator value for masked loss
        clipnorm: Gradient clipping norm
        
    Returns:
        Compiled model
    """
    # Use legacy optimizer for M1/M2 Macs (TF 2.15 recommendation)
    # Also add gradient clipping to prevent exploding gradients
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=learning_rate,
        clipnorm=clipnorm
    )
    
    # Use masked loss to handle NaN values in targets
    loss = MaskedMSELoss(nan_indicator=nan_indicator)
    
    # Metrics
    metrics = [
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
        tf.keras.metrics.RootMeanSquaredError(name='rmse')
    ]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model


def test_model():
    """Test model building and forward pass."""
    print("=" * 60)
    print("Testing ES Transformer Model")
    print("=" * 60)
    
    # Model parameters
    seq_length = 220  # ~23 hours of 5-min candles
    num_features = 150  # Approximate number of features
    target_length = 12  # TR session (12 candles)
    
    print(f"\nBuilding model...")
    print(f"  Input: ({seq_length}, {num_features})")
    print(f"  Output: ({target_length},)")
    
    # Build sequence-to-sequence model
    model = build_es_model(
        seq_length=seq_length,
        num_features=num_features,
        target_length=target_length,
        embed_dim=128,
        num_heads=8,
        ff_dim=512,
        num_layers=4,
        dropout_rate=0.1
    )
    
    # Compile model
    model = compile_model(model)
    
    # Print summary
    model.summary()
    
    # Test forward pass
    print("\n" + "=" * 60)
    print("Testing forward pass...")
    print("=" * 60)
    
    batch_size = 4
    X_test = np.random.randn(batch_size, seq_length, num_features).astype(np.float32)
    
    y_pred = model.predict(X_test, verbose=0)
    print(f"Input shape: {X_test.shape}")
    print(f"Output shape: {y_pred.shape}")
    print(f"Output sample: {y_pred[0][:5]}...")
    
    # Test single output model
    print("\n" + "=" * 60)
    print("Testing single output model...")
    print("=" * 60)
    
    model_single = build_es_model_single_output(
        seq_length=seq_length,
        num_features=num_features,
        embed_dim=128,
        num_heads=8,
        ff_dim=512,
        num_layers=4
    )
    model_single = compile_model(model_single)
    model_single.summary()
    
    y_pred_single = model_single.predict(X_test, verbose=0)
    print(f"Single output shape: {y_pred_single.shape}")


if __name__ == "__main__":
    test_model()
