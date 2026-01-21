"""
Breakout Strategy EDA
=====================

This script analyzes the best performing strategy (ES 03:00 -> 15:55) to understand *why* it works.
It performs:
1. Feature Correlation Analysis
2. Hidden Markov Model (HMM) Regime Detection
3. Volatility Clustering
4. K-Means Clustering on features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("../../consolidated_live_outputs")
SYMBOL = 'ES'

def run_eda():
    print(f"Running EDA for {SYMBOL}...")
    
    # 1. Load Data (Simplified loading of same parquet)
    path = list(DATA_DIR.glob(f"*{SYMBOL}*features*merged.parquet"))[0]
    df = pd.read_parquet(path)
    
    # Preprocess
    df['ts_dt'] = pd.to_datetime(df['timestamp'], utc=True)
    df['ts_et'] = df['ts_dt'].dt.tz_convert('US/Eastern')
    df['time_et'] = df['ts_et'].dt.strftime('%H:%M')
    df['date_et'] = df['ts_et'].dt.date
    
    # Filter for Strategy Time (03:00)
    df_idx = df[df['time_et'] == '03:00'].copy().set_index('date_et')
    df_exit = df[df['time_et'] == '15:55'].copy().set_index('date_et')
    
    common = df_idx.index.intersection(df_exit.index)
    df_idx = df_idx.loc[common]
    df_exit = df_exit.loc[common]
    
    # Calculate Returns
    returns = (df_exit['close'] / df_idx['close'] - 1) * 100
    df_idx['return'] = returns
    df_idx = df_idx[pd.to_datetime(df_idx.index) >= '2016-01-01']
    
    # Numeric Features
    exclude = ['timestamp', 'ts', 'ts_dt', 'ts_et', 'ts_rounded', 'symbol', 'ny_day', 
               'time_et', 'date_et', 'hour_et', 'open', 'high', 'low', 'close', 'return']
    features = [c for c in df_idx.columns if c not in exclude and df_idx[c].dtype in ['float64', 'float32']]
    
    X = df_idx[features].fillna(0).values
    y = df_idx['return'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. HMM Analysis
    print("\n--- HMM Regime Detection ---")
    model = hmm.GaussianHMM(n_components=3, covariance_type='full', n_iter=100, random_state=42)
    # HMM on returns
    model.fit(y.reshape(-1, 1))
    states = model.predict(y.reshape(-1, 1))
    
    for i in range(3):
        mask = states == i
        r = y[mask]
        print(f"State {i}: N={len(r)}, Mean Ret={r.mean():.3f}%, WinRate={(r>0).mean():.1%}")
        
    # 3. K-Means Clustering Analysis
    print("\n--- K-Means Clustering on Features ---")
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    for i in range(4):
        mask = clusters == i
        r = y[mask]
        print(f"Cluster {i}: N={len(r)}, Mean Ret={r.mean():.3f}%, WinRate={(r>0).mean():.1%}")
        
    print("\nEDA Complete.")

if __name__ == "__main__":
    run_eda()
