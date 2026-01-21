# Repository Structure Tree

This document provides a comprehensive tree diagram of the repository structure and explains what each component does.

## Repository Overview

This workspace contains multiple trading/ML research threads focused on AI-driven futures trading systems. The primary repository is `ai-trader-agent/`, which implements a live market state builder, model adapters, probability stacking, and higher-level trade reasoning workflows.

## Full Directory Tree

```
Researched Ideas/
│
├── ai-trader-agent/                    # MAIN REPOSITORY - AI Trading System
│   │
│   ├── src/                            # Core Library Code
│   │   ├── world/                      # Market World State Primitives
│   │   │   ├── fusion/                 # Fusion Layer (state → features/probabilities)
│   │   │   │   ├── live_world.py      # LiveWorld: incremental state builder
│   │   │   │   ├── model_adapters_v3.py # Pure reader adapters (V3)
│   │   │   │   ├── consolidated_live_engine.py # Consolidated row exporter
│   │   │   │   ├── probability.py     # LikelihoodStacker for probability stacking
│   │   │   │   ├── orchestrator.py    # Combines adapter outputs → decisions
│   │   │   │   ├── contracts.py       # Data contracts
│   │   │   │   └── feature_store.py   # Feature storage utilities
│   │   │   │
│   │   │   ├── engines/                # Event/Interaction Engines
│   │   │   │   ├── interaction_engine.py # Bar + levels → interaction/rejection events
│   │   │   │   ├── interaction_live_engine.py # Live interaction engine
│   │   │   │   ├── level_store.py     # Level registry + merging
│   │   │   │   ├── rejection_engine.py # Rejection event detection
│   │   │   │   └── engines/           # Nested engine implementations
│   │   │   │
│   │   │   ├── ranges.py              # Range geometry logic
│   │   │   ├── sessions.py            # Session management (AR/OR/RR/AS/OS/RS)
│   │   │   ├── seq_intraday.py        # Intraday sequencing (04:00→15:55)
│   │   │   ├── seq_daycycle.py        # Day cycle sequencing
│   │   │   ├── seq_asia.py            # Asia session sequencing
│   │   │   ├── box_daily.py           # Box daily logic
│   │   │   ├── wddrb.py               # WDDRB level derivations
│   │   │   ├── modeldb.py             # Model database logic
│   │   │   ├── range_state.py         # Range state tracking
│   │   │   ├── first_range_candle.py  # First range candle logic
│   │   │   ├── weekly.py              # Weekly analysis
│   │   │   ├── clusters.py            # Cluster analysis
│   │   │   ├── calendars.py           # Trading calendar utilities
│   │   │   ├── constants.py           # Trading constants
│   │   │   ├── io.py                  # I/O utilities
│   │   │   ├── action_menu.py        # Action menu logic
│   │   │   │
│   │   │   └── tmpi/                  # TMPI Agent Code
│   │   │       └── session_decision_agent.py # Decision-point logic
│   │   │
│   │   └── trading/                    # Trade Archetype Logic
│   │       └── archetype_logic.py     # Archetype tagging, WDDRB, bias scoring
│   │
│   ├── scripts/                        # CLI Entry Points
│   │   ├── validate/                  # Validation Jobs
│   │   │   ├── consolidated_live/    # Consolidated engine tests
│   │   │   ├── range/                # Range validation
│   │   │   ├── adapters/             # Adapter validation
│   │   │   └── seq/                  # Sequencing validation
│   │   │
│   │   ├── tmpi/                      # TMPI Training/Testing
│   │   │   ├── test_tmpi_minimal.py
│   │   │   ├── test_tmpi_full.py
│   │   │   ├── train_tmpi_full.py
│   │   │   ├── train_tmpi_session_level.py
│   │   │   └── train_tmpi_comprehensive.py
│   │   │
│   │   ├── experiments/
│   │   │   └── mcp_assistant/        # MCP Assistant Scripts
│   │   │       ├── generate_trade_ideas_2016.py
│   │   │       └── backtest_trade_ideas.py
│   │   │
│   │   ├── maintenance/
│   │   │   └── trim_time_parquets.py # Parquet file trimming
│   │   │
│   │   └── [50+ utility scripts]     # Various tests, exports, analyses
│   │       ├── test_*.py             # Test scripts
│   │       ├── export_*.py           # Export scripts
│   │       ├── generate_*.py         # Generation scripts
│   │       ├── build_*.py            # Build scripts
│   │       └── train_*.py            # Training scripts
│   │
│   ├── experiments/                    # Legacy Research
│   │   └── single_row/                # Single-row supervised models
│   │       └── [XGBoost/LightGBM/NN/HMM/LSTM variants]
│   │
│   ├── live/                          # Live Trading Infrastructure
│   │   ├── live_builder.py           # Live world builder
│   │   ├── run_live.py               # Live execution runner
│   │   └── tv_fetcher.py             # TradingView data fetcher
│   │
│   ├── data/                          # Data Storage
│   │   ├── raw/                      # Raw parquet files (5-min bars)
│   │   ├── processed/                # Processed tables
│   │   │   └── <SYMBOL>/            # Per-symbol processed data
│   │   │       ├── ranges/          # Range tables
│   │   │       ├── clusters/        # Cluster tables
│   │   │       └── first_range/     # First range DBs
│   │   ├── csv/                      # CSV datasets
│   │   ├── news/                     # News data
│   │   └── *.parquet                 # Dataset parquet files
│   │
│   ├── models/                        # Trained Model Artifacts
│   │   └── [*.joblib, *.json, *.pt] # Model files and configs
│   │
│   ├── consolidated_live_outputs/     # Consolidated Feature/Probability Exports
│   │   ├── *_features_*.csv         # Feature rows (time-windowed)
│   │   ├── *_probabilities_*.csv    # Probability rows
│   │   ├── *_levels_*.csv           # Level data
│   │   ├── *_rejection_*.csv        # Rejection event data
│   │   └── [dated export files]
│   │
│   ├── artifacts/                     # Workflow Inputs/Outputs
│   │   ├── inputs/                   # Manual inputs (CSV files)
│   │   ├── outputs/                  # Generated outputs (CSV, JSON)
│   │   └── models/                   # Training artifacts (PKL files)
│   │
│   ├── reports/                       # Analysis Outputs
│   │   ├── [*.csv]                  # Calibration/summary CSVs
│   │   ├── [*.json]                 # Analysis JSONs
│   │   └── [*.png]                  # Plot images
│   │
│   ├── tests/                         # Unit Tests
│   │   └── [test_*.py]
│   │
│   ├── docs/                         # Documentation
│   │   └── EXPERIMENTS_INDEX.md     # "You are here" map
│   │
│   ├── mcp_server.py                 # MCP Server (Model Context Protocol)
│   │                                   # Exposes consolidated state via tools
│   │
│   ├── pyproject.toml                # Project configuration
│   │
│   └── [Architecture Documentation]
│       ├── README.md                 # Main repository README
│       ├── V2_ARCHITECTURE.md       # LiveWorld + Adapter separation
│       ├── LIVE_FEATURE_ENGINE.md   # Live 5-min feature engineering
│       ├── MCP_SERVER_PLAN.md       # MCP server tools/plan
│       ├── WDDRB_PATCHES_SUMMARY.md # WDDRB patch notes
│       └── LIVEWORLD_PATCHES_SUMMARY.md # LiveWorld patch notes
│
├── backtest_results_rdr_box_breakout/ # RDR Box Breakout Strategy Results
│   ├── backtest_rdr_box_breakout.py  # 5R strategy with 1-tick slippage
│   ├── backtest_results.txt          # Results summary
│   ├── daily_r_analysis.py          # Daily R analysis
│   ├── trades_rdr_box_breakout_5R.csv
│   ├── equity_curves_by_asset.png
│   ├── equity_with_commissions.png
│   └── README.md
│
├── backtest_results_rdr_session/     # Two Bullet Strategy Results
│   ├── backtest_simple_two_bullet.py # 2.5 SD target strategy
│   ├── backtest_results.txt
│   ├── daily_r_analysis.py
│   ├── trades_two_bullet_2.5_sd.csv
│   ├── equity_curves_by_asset.png
│   ├── equity_with_commissions.png
│   └── README.md
│
├── quarterly_and_downloader_bundle/  # Quarterly Range Breakout Strategies
│   ├── Quarterly_Range_Backtest.py   # Direct trading quarterly breakout backtest
│   ├── Options_Quarterly_Backtest.py # Options-based quarterly breakout backtest
│   ├── SPX_Quarterly_Analysis.py     # Quarterly range analysis script
│   ├── Multi_Asset_Quarterly_Analysis.py # Multi-asset quarterly analysis
│   ├── Downloader.py                 # Generic data downloader
│   ├── SP500_Downloader.py           # SPX-specific data downloader
│   ├── SPX_1D.csv                    # SPX daily price data
│   ├── quarterly_expiration_strategy_performance.png # Direct trading equity curves
│   ├── quarterly_range_backtest_performance.png      # Direct trading performance
│   ├── options_strategy_performance.png              # Options cumulative PnL
│   └── README.md                     # Quarterly strategy documentation
│
├── breakout_optimization/            # Breakout Strategy Optimization & Analysis
│   ├── run_strategy_optimization.py  # Main optimization script with walk-forward validation
│   ├── run_strategy_eda.py           # Exploratory data analysis script
│   ├── equity_curve_0.8pct_stops.png # Equity curve with 0.8% stop loss
│   ├── combined_es_cl_gc_equity.png  # Combined equity curves across assets
│   ├── walk_forward_es_cl_gc.png     # Walk-forward validation results
│   └── README.md                     # Optimization strategy documentation
│
├── TransformerTest/                  # Transformer Model Experiments
│   ├── es_transformer_model.py       # ES transformer implementation
│   ├── train_es_predictor.py         # Training script
│   ├── data_loader.py                # Data loading utilities
│   ├── main_commented.py             # Main script (commented)
│   ├── notebook.ipynb                # Jupyter notebook
│   └── logs/                         # Training logs & checkpoints
│       └── [scenario directories with training artifacts]
│
├── Example Data ES.csv               # Example dataset
│
└── README.md                         # Workspace overview
```

## Key Components Explained

### Core Architecture (ai-trader-agent/src/world/)

#### Fusion Layer (`src/world/fusion/`)
- **live_world.py**: The central LiveWorld system that ingests 5-minute bars and builds immutable state snapshots
- **model_adapters_v3.py**: Pure reader adapters that consume snapshots and produce probabilities
- **consolidated_live_engine.py**: Exports consolidated rows combining features + probabilities
- **probability.py**: LikelihoodStacker for stacking probabilities across multiple models
- **orchestrator.py**: Combines adapter outputs and optional rejection signals into decisions

#### State Primitives
- **ranges.py**: Range geometry calculations (IDR, DR, box_std, etc.)
- **sessions.py**: Session management (AR/OR/RR for ranges, AS/OS/RS for follow sessions)
- **seq_intraday.py**: Intraday sequencing tracking (04:00→15:55 high/low)
- **box_daily.py**: Box daily calculations
- **wddrb.py**: WDDRB (Width-Depth-Density Range Break) level derivations
- **modeldb.py**: Model database queries and probability lookups

#### Engines (`src/world/engines/`)
- **interaction_engine.py**: Converts bars + level registries into structured interaction/rejection events
- **level_store.py**: Level registry with merging helpers

### Scripts Organization

#### Validation Scripts (`scripts/validate/`)
Domain-specific validation jobs:
- Consolidated live engine tests
- Range validation tests
- Adapter validation tests
- Sequencing validation tests

#### TMPI Scripts (`scripts/tmpi/`)
Temporal Market Program Induction:
- Training scripts for learning rule-like decision logic
- Testing scripts for evaluation

#### MCP Assistant (`scripts/experiments/mcp_assistant/`)
Trade idea generation and evaluation:
- Generate trade ideas from historical data
- Backtest trade ideas

### Data Organization

#### `data/processed/<SYMBOL>/`
Processed tables organized by symbol:
- **ranges/**: Range session data
- **clusters/**: Cluster analysis data
- **first_range/**: First range database tables

#### `consolidated_live_outputs/`
Time-windowed consolidated exports:
- Feature rows (all features in one row per bar)
- Probability rows (all model probabilities per bar)
- Level data (support/resistance levels)
- Rejection event data (rejection patterns)

### Supporting Projects

#### Backtest Results
- **rdr_box_breakout/**: RDR box breakout strategy (5R target, 1-tick slippage)
- **rdr_session/**: Two bullet strategy (2.5 SD target)
- **quarterly_and_downloader_bundle/**: Quarterly range breakout strategies (direct trading and options variants) with equity curve generation
- **breakout_optimization/**: Breakout strategy optimization with XGBoost classifiers and walk-forward validation (ES, CL, GC)

#### TransformerTest
Experimental transformer models for price prediction

## Data Flow

```
5-minute bars 
    ↓
LiveWorld.ingest(bar)
    ↓
LiveSnapshot (immutable state)
    ↓
ModelAdapters (pure readers)
    ↓
Probabilities (per model)
    ↓
LikelihoodStacker
    ↓
Stacked Probabilities
    ↓
Orchestrator
    ↓
Trade Decisions
```

## Quick Navigation

- **Main README**: `ai-trader-agent/README.md`
- **Architecture Docs**: `ai-trader-agent/V2_ARCHITECTURE.md`
- **Experiments Index**: `ai-trader-agent/docs/EXPERIMENTS_INDEX.md`
- **Entry Point**: Start with `ai-trader-agent/README.md` for full repo map

## Common Entry Points

- **Validations**: `python scripts/validate/<domain>/test_*.py`
- **TMPI**: `python scripts/tmpi/test_tmpi_full.py`
- **MCP Server**: `python mcp_server.py`
- **Live Trading**: `python live/run_live.py`
