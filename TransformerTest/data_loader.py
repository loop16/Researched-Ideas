#!/usr/bin/env python3
"""
Data Loader for ES Futures Transformer Prediction Model

This module handles:
1. Loading and merging feature + probability parquet files
2. Encoding categorical features while preserving NaN values
3. Creating sequence windows for prediction (23 hours: 9:30 AM -> next day 8:25 AM)
4. Generating targets for price path prediction
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class ESDataLoader:
    """
    Data loader for ES futures prediction.
    
    Input Window: 9:30 AM to next day 8:25 AM (23 hours = ~222 candles)
    This captures: RR, RS, TOUT, TA, AR, AS, TO, OR, OS sessions
    
    Target Windows:
    - Scenario 1: Predict TR session (08:30-09:25) - 12 candles
    - Scenario 2: Predict TR+RR sessions (08:30-10:25) - 24 candles
    """
    
    # Session order within a trading day (NY time)
    SESSION_ORDER = ['AS', 'TO', 'OR', 'OS', 'TR', 'RR', 'RS', 'TOUT', 'TA', 'AR']
    
    # Sessions in the input window (from 9:30 AM day 1 to 8:25 AM day 2)
    INPUT_SESSIONS_DAY1 = ['RR', 'RS', 'TOUT', 'TA', 'AR']  # After 9:30 AM
    INPUT_SESSIONS_DAY2 = ['AS', 'TO', 'OR', 'OS']  # Before 8:25 AM next day
    
    # Target sessions  
    TARGET_SESSIONS_TR = ['TR']  # 08:30-09:25
    TARGET_SESSIONS_TR_RR = ['TR', 'RR']  # 08:30-10:25
    
    # NaN indicator values
    # For features: Use 0.0 (neutral after z-score normalization)
    # For targets: Use -999.0 to identify and mask in loss function
    FEATURE_NAN_INDICATOR = 0.0
    TARGET_NAN_INDICATOR = -999.0
    
    def __init__(self, 
                 features_path: str = 'consolidated_live_features_ES_2008-01-10_2025-09-25_merged.parquet',
                 probabilities_path: str = 'consolidated_probabilities_ES_2008-01-10_2025-09-25_merged.parquet'):
        """
        Initialize the data loader.
        
        Args:
            features_path: Path to the live features parquet file
            probabilities_path: Path to the probabilities parquet file
        """
        self.features_path = features_path
        self.probabilities_path = probabilities_path
        
        # Storage for encoders and scalers
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_stats: Dict[str, Tuple[float, float]] = {}  # (mean, std) for each feature
        
        # Feature lists (populated during loading)
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.all_features: List[str] = []
        
        # Columns to exclude from features
        self.exclude_cols = ['timestamp', 'ts', 'symbol', 'ny_day']
        
        # Target column
        self.target_col = 'close_ar_pct'
        
        # Data storage
        self.df: Optional[pd.DataFrame] = None
        self.is_fitted = False
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and merge the feature and probability dataframes.
        
        Returns:
            Merged dataframe with all features
        """
        print("Loading feature data...")
        df_features = pd.read_parquet(self.features_path)
        print(f"  Features shape: {df_features.shape}")
        
        print("Loading probability data...")
        df_probs = pd.read_parquet(self.probabilities_path)
        print(f"  Probabilities shape: {df_probs.shape}")
        
        # Merge on 'ts' column (timestamp in seconds)
        # Drop duplicate columns from probabilities before merge
        prob_cols_to_use = [c for c in df_probs.columns if c not in ['timestamp', 'symbol', 'session'] or c == 'ts']
        df_probs_subset = df_probs[prob_cols_to_use]
        
        print("Merging dataframes on 'ts' column...")
        self.df = pd.merge(df_features, df_probs_subset, on='ts', how='left', suffixes=('', '_prob'))
        print(f"  Merged shape: {self.df.shape}")
        
        # Parse timestamp
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Sort by timestamp
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        return self.df
    
    def _identify_features(self):
        """Identify numeric and categorical features."""
        all_cols = self.df.columns.tolist()
        
        # Separate numeric and categorical
        self.numeric_features = []
        self.categorical_features = []
        
        for col in all_cols:
            if col in self.exclude_cols or col == self.target_col:
                continue
                
            dtype = self.df[col].dtype
            if dtype in ['float64', 'int64', 'float32', 'int32']:
                self.numeric_features.append(col)
            elif dtype in ['object', 'bool']:
                self.categorical_features.append(col)
        
        print(f"Identified {len(self.numeric_features)} numeric features")
        print(f"Identified {len(self.categorical_features)} categorical features")
        
    def _encode_categorical_features(self):
        """
        Encode categorical features using LabelEncoder.
        Preserves NaN values by encoding them separately.
        Normalizes encoded values to 0-1 range to prevent large values.
        """
        print("Encoding categorical features...")
        
        for col in self.categorical_features:
            # Create label encoder
            le = LabelEncoder()
            
            # Get non-null values for fitting
            non_null_mask = self.df[col].notna()
            non_null_values = self.df.loc[non_null_mask, col].astype(str).values
            
            # Fit encoder on unique non-null values
            if len(non_null_values) > 0:
                unique_values = np.unique(non_null_values)
                le.fit(unique_values)
                
                # Transform non-null values
                encoded_col = np.full(len(self.df), self.FEATURE_NAN_INDICATOR, dtype=np.float32)
                encoded_values = le.transform(non_null_values).astype(np.float32)
                
                # Normalize to 0-1 range to prevent large values
                max_val = len(unique_values) - 1
                if max_val > 0:
                    encoded_values = encoded_values / max_val
                
                encoded_col[non_null_mask] = encoded_values
                
                # Store encoder
                self.label_encoders[col] = le
                
                # Create new column with encoded values
                self.df[f'{col}_encoded'] = encoded_col
            else:
                # All null column
                self.df[f'{col}_encoded'] = self.FEATURE_NAN_INDICATOR
                
        print(f"  Created {len(self.categorical_features)} encoded columns")
    
    def _normalize_numeric_features(self):
        """
        Normalize numeric features using z-score normalization.
        Preserves NaN values by replacing them with FEATURE_NAN_INDICATOR after normalization.
        Also clips extreme values to prevent NaN in training.
        """
        print("Normalizing numeric features...")
        
        # Clipping threshold for normalized values (standard deviations)
        CLIP_STD = 10.0
        
        for col in self.numeric_features:
            # Calculate mean and std on non-null values
            non_null_mask = self.df[col].notna()
            raw_values = self.df.loc[non_null_mask, col].values.astype(np.float64)
            
            # Replace inf values with NaN for statistics
            raw_values = np.where(np.isinf(raw_values), np.nan, raw_values)
            non_null_values = raw_values[~np.isnan(raw_values)]
            
            if len(non_null_values) > 0:
                mean_val = np.nanmean(non_null_values)
                std_val = np.nanstd(non_null_values)
                if std_val == 0 or np.isnan(std_val):
                    std_val = 1.0  # Avoid division by zero
                
                # Store stats for inverse transform
                self.feature_stats[col] = (mean_val, std_val)
                
                # Normalize and clip
                normalized = np.full(len(self.df), self.FEATURE_NAN_INDICATOR, dtype=np.float32)
                norm_values = (raw_values - mean_val) / std_val
                
                # Clip extreme values to prevent NaN in training
                norm_values = np.clip(norm_values, -CLIP_STD, CLIP_STD)
                
                # Replace any remaining NaN/inf with 0
                norm_values = np.where(np.isfinite(norm_values), norm_values, 0.0)
                
                normalized[non_null_mask] = norm_values.astype(np.float32)
                
                self.df[f'{col}_norm'] = normalized
            else:
                self.feature_stats[col] = (0.0, 1.0)
                self.df[f'{col}_norm'] = self.FEATURE_NAN_INDICATOR
                
        print(f"  Normalized {len(self.numeric_features)} numeric features")
    
    def fit_transform(self) -> pd.DataFrame:
        """
        Load data, identify features, encode categoricals, and normalize numerics.
        
        Returns:
            Transformed dataframe
        """
        if self.df is None:
            self.load_data()
            
        self._identify_features()
        self._encode_categorical_features()
        self._normalize_numeric_features()
        
        # Build final feature list (normalized numeric + encoded categorical)
        self.all_features = [f'{col}_norm' for col in self.numeric_features]
        self.all_features += [f'{col}_encoded' for col in self.categorical_features]
        
        print(f"\nTotal features for model: {len(self.all_features)}")
        self.is_fitted = True
        
        return self.df
    
    def _get_trading_days(self) -> List[str]:
        """Get list of unique trading days."""
        return sorted(self.df['ny_day'].unique().tolist())
    
    def create_sequences_scenario1(self, 
                                    max_samples: Optional[int] = None,
                                    sample_evenly: bool = True
                                   ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create sequences for Scenario 1: Predict TR session.
        
        Input: 9:30 AM day N to 8:25 AM day N+1 (covers RR, RS, TOUT, TA, AR, AS, TO, OR, OS)
        Target: TR session on day N+1 (08:30-09:25) - predict close_ar_pct path (12 candles)
        
        Args:
            max_samples: Maximum number of samples to create (None for all)
            sample_evenly: If True and max_samples is set, sample evenly across date range
        
        Returns:
            X: Input sequences of shape (num_samples, seq_len, num_features)
            y: Target sequences of shape (num_samples, target_len)
            dates: List of target dates for each sample
        """
        if not self.is_fitted:
            self.fit_transform()
            
        trading_days = self._get_trading_days()
        
        # If sampling evenly, determine which days to use
        if max_samples and sample_evenly and len(trading_days) > max_samples:
            step = len(trading_days) // max_samples
            selected_indices = list(range(0, len(trading_days) - 1, step))[:max_samples]
        else:
            selected_indices = list(range(len(trading_days) - 1))
        
        X_samples = []
        y_samples = []
        sample_dates = []
        
        print(f"Creating sequences for {len(trading_days)} trading days (sampling {len(selected_indices)} days)...")
        
        for i in selected_indices:
            day1 = trading_days[i]
            day2 = trading_days[i + 1]
            
            # Get input data: day1 from 9:30 AM onwards + day2 until 8:25 AM
            day1_data = self.df[self.df['ny_day'] == day1].copy()
            day2_data = self.df[self.df['ny_day'] == day2].copy()
            
            # Filter day1: sessions from RR onwards (9:30 AM+)
            day1_input = day1_data[day1_data['session'].isin(self.INPUT_SESSIONS_DAY1)]
            
            # Filter day2: sessions before TR (before 8:30 AM)
            day2_input = day2_data[day2_data['session'].isin(self.INPUT_SESSIONS_DAY2)]
            
            # Combine input window
            input_data = pd.concat([day1_input, day2_input], ignore_index=True)
            
            # Get target: TR session on day2
            target_data = day2_data[day2_data['session'].isin(self.TARGET_SESSIONS_TR)]
            
            # Skip if insufficient data
            if len(input_data) < 100 or len(target_data) < 10:
                continue
                
            # Extract features for input
            X = input_data[self.all_features].values.astype(np.float32)
            # Replace any remaining NaN/inf with 0 in features
            X = np.where(np.isfinite(X), X, 0.0)
            
            # Extract target (close_ar_pct for TR session)
            if f'{self.target_col}_norm' in target_data.columns:
                y = target_data[f'{self.target_col}_norm'].values.astype(np.float32)
            else:
                y = target_data[self.target_col].values.astype(np.float32)
            
            # Replace any NaN/inf in target with TARGET_NAN_INDICATOR
            y = np.where(np.isfinite(y), y, self.TARGET_NAN_INDICATOR)
            
            X_samples.append(X)
            y_samples.append(y)
            sample_dates.append(day2)
            
            if max_samples and not sample_evenly and len(X_samples) >= max_samples:
                break
                
        print(f"Created {len(X_samples)} samples")
        
        # Pad sequences to same length
        X_padded, y_padded = self._pad_sequences(X_samples, y_samples)
        
        return X_padded, y_padded, sample_dates
    
    def create_sequences_scenario2(self,
                                    max_samples: Optional[int] = None,
                                    sample_evenly: bool = True
                                   ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create sequences for Scenario 2: Predict RS session.
        
        Input: 9:30 AM day N to 10:25 AM day N+1 (includes TR+RR on day N+1)
        Target: RS session on day N+1 (10:30-15:55) - predict close_ar_pct path (66 candles)
        
        Args:
            max_samples: Maximum number of samples to create (None for all)
            sample_evenly: If True and max_samples is set, sample evenly across date range
        
        Returns:
            X: Input sequences of shape (num_samples, seq_len, num_features)
            y: Target sequences of shape (num_samples, target_len)
            dates: List of target dates for each sample
        """
        if not self.is_fitted:
            self.fit_transform()
            
        trading_days = self._get_trading_days()
        
        # If sampling evenly, determine which days to use
        if max_samples and sample_evenly and len(trading_days) > max_samples:
            step = len(trading_days) // max_samples
            selected_indices = list(range(0, len(trading_days) - 1, step))[:max_samples]
        else:
            selected_indices = list(range(len(trading_days) - 1))
        
        X_samples = []
        y_samples = []
        sample_dates = []
        
        print(f"Creating sequences for {len(trading_days)} trading days (sampling {len(selected_indices)} days)...")
        
        for i in selected_indices:
            day1 = trading_days[i]
            day2 = trading_days[i + 1]
            
            # Get data for both days
            day1_data = self.df[self.df['ny_day'] == day1].copy()
            day2_data = self.df[self.df['ny_day'] == day2].copy()
            
            # Filter day1: sessions from RR onwards (9:30 AM+)
            day1_input = day1_data[day1_data['session'].isin(self.INPUT_SESSIONS_DAY1)]
            
            # Filter day2: sessions up to and including RR (through 10:25 AM)
            day2_input_sessions = self.INPUT_SESSIONS_DAY2 + ['TR', 'RR']
            day2_input = day2_data[day2_data['session'].isin(day2_input_sessions)]
            
            # Combine input window
            input_data = pd.concat([day1_input, day2_input], ignore_index=True)
            
            # Get target: RS session on day2
            target_data = day2_data[day2_data['session'] == 'RS']
            
            # Skip if insufficient data
            if len(input_data) < 100 or len(target_data) < 30:
                continue
                
            # Extract features for input
            X = input_data[self.all_features].values.astype(np.float32)
            # Replace any remaining NaN/inf with 0 in features
            X = np.where(np.isfinite(X), X, 0.0)
            
            # Extract target (close_ar_pct for RS session)
            if f'{self.target_col}_norm' in target_data.columns:
                y = target_data[f'{self.target_col}_norm'].values.astype(np.float32)
            else:
                y = target_data[self.target_col].values.astype(np.float32)
            
            # Replace any NaN/inf in target with TARGET_NAN_INDICATOR
            y = np.where(np.isfinite(y), y, self.TARGET_NAN_INDICATOR)
            
            X_samples.append(X)
            y_samples.append(y)
            sample_dates.append(day2)
            
            if max_samples and not sample_evenly and len(X_samples) >= max_samples:
                break
                
        print(f"Created {len(X_samples)} samples")
        
        # Pad sequences to same length
        X_padded, y_padded = self._pad_sequences(X_samples, y_samples)
        
        return X_padded, y_padded, sample_dates
    
    def _pad_sequences(self, 
                       X_samples: List[np.ndarray], 
                       y_samples: List[np.ndarray]
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad sequences to uniform length using NAN_INDICATOR.
        
        Args:
            X_samples: List of input arrays with varying lengths
            y_samples: List of target arrays with varying lengths
            
        Returns:
            X_padded: Padded input array (num_samples, max_seq_len, num_features)
            y_padded: Padded target array (num_samples, max_target_len)
        """
        if len(X_samples) == 0:
            return np.array([]), np.array([])
        
        # Find max lengths
        max_X_len = max(x.shape[0] for x in X_samples)
        max_y_len = max(y.shape[0] for y in y_samples)
        num_features = X_samples[0].shape[1]
        
        print(f"Padding to max lengths: X={max_X_len}, y={max_y_len}")
        
        # Create padded arrays
        X_padded = np.full((len(X_samples), max_X_len, num_features), 
                          self.FEATURE_NAN_INDICATOR, dtype=np.float32)
        y_padded = np.full((len(y_samples), max_y_len), 
                          self.TARGET_NAN_INDICATOR, dtype=np.float32)
        
        # Fill in actual values
        for i, (x, y) in enumerate(zip(X_samples, y_samples)):
            X_padded[i, :x.shape[0], :] = x
            y_padded[i, :y.shape[0]] = y
            
        return X_padded, y_padded
    
    def split_data(self, 
                   X: np.ndarray, 
                   y: np.ndarray, 
                   dates: List[str],
                   train_end_date: str = '2020-12-31',
                   val_end_date: str = '2022-12-31'
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                            np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data chronologically into train/validation/test sets.
        
        Args:
            X: Input features
            y: Targets
            dates: Date strings for each sample
            train_end_date: End date for training data
            val_end_date: End date for validation data
            
        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
        """
        dates_arr = np.array(dates)
        
        train_mask = dates_arr <= train_end_date
        val_mask = (dates_arr > train_end_date) & (dates_arr <= val_end_date)
        test_mask = dates_arr > val_end_date
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[val_mask]
        y_val = y[val_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        print(f"Data split:")
        print(f"  Train: {len(X_train)} samples (up to {train_end_date})")
        print(f"  Val: {len(X_val)} samples ({train_end_date} to {val_end_date})")
        print(f"  Test: {len(X_test)} samples (after {val_end_date})")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def get_feature_names(self) -> List[str]:
        """Get the list of all feature names used in the model."""
        return self.all_features
    
    def inverse_transform_target(self, y_normalized: np.ndarray) -> np.ndarray:
        """
        Convert normalized target values back to original scale.
        
        Args:
            y_normalized: Normalized target values
            
        Returns:
            Original scale target values
        """
        if self.target_col in self.feature_stats:
            mean_val, std_val = self.feature_stats[self.target_col]
            # Handle NaN indicators
            y_original = np.where(
                y_normalized == self.TARGET_NAN_INDICATOR,
                np.nan,
                y_normalized * std_val + mean_val
            )
            return y_original
        return y_normalized


def test_data_loader():
    """Test the data loader functionality."""
    print("=" * 60)
    print("Testing ESDataLoader")
    print("=" * 60)
    
    loader = ESDataLoader()
    
    # Load and transform data
    df = loader.fit_transform()
    print(f"\nTransformed dataframe shape: {df.shape}")
    print(f"Number of features: {len(loader.all_features)}")
    
    # Create sequences for scenario 1 (small sample for testing)
    print("\n" + "=" * 60)
    print("Testing Scenario 1 (Predict TR session)")
    print("=" * 60)
    X1, y1, dates1 = loader.create_sequences_scenario1(max_samples=100)
    print(f"X shape: {X1.shape}")
    print(f"y shape: {y1.shape}")
    
    # Create sequences for scenario 2
    print("\n" + "=" * 60)
    print("Testing Scenario 2 (Predict RS session)")
    print("=" * 60)
    X2, y2, dates2 = loader.create_sequences_scenario2(max_samples=100)
    print(f"X shape: {X2.shape}")
    print(f"y shape: {y2.shape}")
    
    # Test data splitting
    if len(X1) > 0:
        print("\n" + "=" * 60)
        print("Testing data split")
        print("=" * 60)
        X_train, y_train, X_val, y_val, X_test, y_test = loader.split_data(
            X1, y1, dates1, 
            train_end_date='2020-12-31', 
            val_end_date='2022-12-31'
        )


if __name__ == "__main__":
    test_data_loader()
