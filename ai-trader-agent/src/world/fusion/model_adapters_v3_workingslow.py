##wed nov 19 2025
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from numbers import Real
from typing import Any, Callable, Dict, Optional, Tuple, List
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

from .contracts import (
    Bar,
    RangeSessionOut,
    SeqIntradayOut,
    SeqDaycycleOut,
    SeqAsiaOut,
    ModelDBOut,
    BoxDailyOut,
    FirstRangeCandleOut,
    WeeklyOut,
    WddrbOut,
    WddrbModelOut,
)
from .consolidated_live_engine import ConsolidatedLiveEngine
from ..constants import NY_TZ
from ..calendars import prev_business_day, next_business_day, trading_week_tuesday
from ..weekly import DAY_NAME


ALL_MODELS: tuple[str, ...] = ("RC", "RX", "UX", "DX", "UXP", "DXP", "U", "D")

# Feature toggle: include RR model (NextRDR) when filtering intraday history.
# Keep disabled by default until the RR cohort is fully vetted.
INCLUDE_RR_MODEL_FILTER: bool = True

# Allowed forward transitions by current model (context guard for next-range probabilities)
ALLOWED_NEXT: Dict[str, tuple[str, ...]] = {
    # Absorbing
    "RX": ("RX",),

    # Single-path funnels
    "U":  ("RX",),
    "D":  ("RX",),

    # Two-branch funnels
    "UX": ("U", "RX"),
    "DX": ("D", "RX"),

    # Multi-branch (but no gaps)
    "RC": ("RC", "U", "D", "UX", "DX", "RX"),

    # Gap/extension branches
    "UXP": ("UX", "U", "RX"),
    "DXP": ("DX", "D", "RX"),
}

WDDRB_MODEL_CODES: tuple[str, ...] = ("UG", "U", "UX", "O", "I", "DX", "D", "DG")
WDDRB_OPEN_SD_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (-float("inf"), -1.5, "below_low"),
    (-1.5, -1.25, "neg_150_125"),
    (-1.25, -1.0, "neg_125_100"),
    (-1.0, -0.75, "neg_100_075"),
    (-0.75, -0.5, "neg_075_050"),
    (-0.5, -0.25, "neg_050_025"),
    (-0.25, 0.0, "neg_025_000"),
    (0.0, 0.25, "pos_000_025"),
    (0.25, 0.5, "pos_025_050"),
    (0.5, 0.75, "pos_050_075"),
    (0.75, 1.0, "pos_075_100"),
    (1.0, 1.25, "pos_100_125"),
    (1.25, 1.5, "pos_125_150"),
    (1.5, float("inf"), "above_high"),
)

TRACKED_SESSIONS: tuple[str, ...] = ("AS", "TO", "OR", "OS", "TR", "RR", "RS")

@dataclass
class _Last:
    """Lightweight holder for last consolidated output per (ny_day, symbol)."""
    ny_day: date
    symbol: str
    out: Dict


class ModelAdaptersV3:
    """V3 adapters powered by ConsolidatedLiveEngine outputs.

    This class ingests 5m bars into ConsolidatedLiveEngine and maps its
    consolidated features to adapter contracts. Historical probability
    lookups can be layered later; for now we produce neutral probabilities
    when history is unavailable.
    """

    def __init__(self, data_dir: str):
        # data_dir should point to the ai-trader-agent/data directory
        self.engine = ConsolidatedLiveEngine(data_dir)
        # Cache the most recent consolidated output per (ny_day, symbol)
        self._last: Dict[Tuple[date, str], _Last] = {}
        # Load range session metrics for historical percentiles
        self.data_dir = data_dir
        self.range_metrics_df = self._load_data("range_session_metrics.csv")
        # Per-symbol metrics cache (prefer instrument-scoped parquet)
        self._metrics_cache: Dict[str, pd.DataFrame] = {}
        # ModelDB history (per symbol) for mid-break and model-context filtering
        self._modeldb_cache: Dict[str, pd.DataFrame] = {}
        # Seq intraday history cache (per symbol)
        self._seq_intraday_cache: Dict[str, pd.DataFrame] = {}
        # Seq daycycle history cache (per symbol)
        self._seq_daycycle_cache: Dict[str, pd.DataFrame] = {}
        # Seq asia history cache (per symbol)
        self._seq_asia_cache: Dict[str, pd.DataFrame] = {}
        # First range candle history cache (per symbol)
        self._first_range_cache: Dict[str, pd.DataFrame] = {}
        # Weekly model history cache (per symbol)
        self._weekly_cache: Dict[str, pd.DataFrame] = {}
        # WDDRB history cache (per symbol)
        self._wddrb_cache: Dict[str, pd.DataFrame] = {}
        # Cache for AR/OR models and RR box direction (keyed by (anchor_day, symbol))
        self._daycycle_model_cache: Dict[Tuple[date, str], Dict[str, Optional[str]]] = {}
        # Running Asia extremes keyed by (anchor_day, symbol)
        self._asia_extrema: Dict[Tuple[date, str], Dict[str, Any]] = {}
        # Box daily historical caches
        self._box_day_cache: Dict[str, pd.DataFrame] = {}
        self._box_session_cache: Dict[str, pd.DataFrame] = {}
        self._box_range_quantiles: Dict[str, Dict[str, float]] = {}
        # Live session trackers for Asia-normalized SD calculations
        self._box_session_tracker: Dict[Tuple[date, str], Dict[str, Dict[str, Any]]] = {}
        self.MIN_SEQ_ROWS = 10
        self.include_rdr_filter = INCLUDE_RR_MODEL_FILTER
        self.wddrb_progress_buckets = 6.0
        self._wddrb_hold_cache: Dict[str, Dict[str, Any]] = {}
        self._wddrb_last_adr: Dict[str, Optional[str]] = {}
        self._wddrb_last_day: Dict[str, date] = {}
        self.MIN_WDDRB_MODEL_ROWS = 40
        self._wddrb_model_state: Dict[str, Dict[str, Any]] = {}

    def _load_data(self, filename: str) -> pd.DataFrame:
        """Load a metrics table, with fallbacks for instrument-scoped files.

        Tries generic processed/csv paths first, then searches recursively for
        files like processed/*/*_range_session_metrics.parquet or csv.
        """
        from pathlib import Path
        base = Path(self.data_dir)
        pq = base / "processed" / filename.replace(".csv", ".parquet")
        if pq.exists():
            return pd.read_parquet(pq)
        csv = base / "csv" / filename
        if csv.exists():
            return pd.read_csv(csv)
        # Instrument-scoped fallbacks
        frames: list[pd.DataFrame] = []
        try:
            for f in (base / "processed").rglob("*_range_session_metrics.parquet"):
                try:
                    frames.append(pd.read_parquet(f))
                except Exception:
                    pass
            if not frames:
                for f in (base / "csv").rglob("*_range_session_metrics.csv"):
                    try:
                        frames.append(pd.read_csv(f))
                    except Exception:
                        pass
        except Exception:
            pass
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _get_metrics_for_symbol(self, symbol: str) -> pd.DataFrame:
        """Return metrics dataframe for a given symbol, preferring parquet in processed/<SYM>/.

        Falls back to CSV in csv/<SYM>/, then any instrument-scoped files, then the generic table.
        Results are cached per symbol.
        """
        if symbol in self._metrics_cache:
            return self._metrics_cache[symbol]
        from pathlib import Path
        base = Path(self.data_dir)
        # 1) Direct instrument parquet
        p_main = base / "processed" / symbol / f"{symbol}_range_session_metrics.parquet"
        try:
            if p_main.exists():
                df = pd.read_parquet(p_main)
                self._metrics_cache[symbol] = df
                return df
        except Exception:
            pass
        # 2) Any parquet under processed/<symbol>/ matching pattern
        try:
            sym_dir = base / "processed" / symbol
            if sym_dir.exists():
                for f in sym_dir.glob("*_range_session_metrics.parquet"):
                    try:
                        df = pd.read_parquet(f)
                        self._metrics_cache[symbol] = df
                        return df
                    except Exception:
                        continue
        except Exception:
            pass
        # 3) CSV under csv/<symbol>/
        try:
            c_main = base / "csv" / symbol / f"{symbol}_range_session_metrics.csv"
            if c_main.exists():
                df = pd.read_csv(c_main)
                self._metrics_cache[symbol] = df
                return df
            sym_csv_dir = base / "csv" / symbol
            if sym_csv_dir.exists():
                for f in sym_csv_dir.glob("*_range_session_metrics.csv"):
                    try:
                        df = pd.read_csv(f)
                        self._metrics_cache[symbol] = df
                        return df
                    except Exception:
                        continue
        except Exception:
            pass
        # 4) Any instrument-scoped parquet under processed/**
        try:
            for f in (base / "processed").rglob(f"*{symbol}*_range_session_metrics.parquet"):
                try:
                    df = pd.read_parquet(f)
                    self._metrics_cache[symbol] = df
                    return df
                except Exception:
                    continue
        except Exception:
            pass
        # 5) Fallback to generic loaded table
        self._metrics_cache[symbol] = self.range_metrics_df
        return self.range_metrics_df

    def _get_modeldb_for_symbol(self, symbol: str) -> pd.DataFrame:
        """Load model_db history for a symbol."""
        if symbol in self._modeldb_cache:
            return self._modeldb_cache[symbol]

        from pathlib import Path

        base = Path(self.data_dir)

        def _read(path: Path) -> Optional[pd.DataFrame]:
            try:
                if path.suffix == ".parquet":
                    return pd.read_parquet(path)
                if path.suffix in (".csv", ".txt"):
                    return pd.read_csv(path)
            except Exception:
                return None
            return None

        candidates: List[Path] = []
        candidates.extend([
            base / "processed" / symbol / f"{symbol}_model_db.parquet",
            base / "processed" / symbol / f"{symbol}_model_db.csv",
            base / "processed" / symbol / "model_db.parquet",
            base / "processed" / symbol / "model_db.csv",
        ])
        candidates.extend([
            base / "processed" / f"{symbol}_model_db.parquet",
            base / "processed" / f"{symbol}_model_db.csv",
            base / "processed" / "model_db.parquet",
            base / "processed" / "model_db.csv",
        ])
        candidates.extend([
            base / "csv" / symbol / f"{symbol}_model_db.csv",
            base / "csv" / symbol / "model_db.csv",
            base / "csv" / "model_db.csv",
        ])

        for path in candidates:
            if path.exists() and path.is_file():
                df = _read(path)
                if df is not None and not df.empty:
                    df = self._prepare_modeldb_df(df)
                    self._modeldb_cache[symbol] = df
                    return df

        try:
            for pattern in (f"*{symbol}*_model_db.parquet", f"*{symbol}*_model_db.csv"):
                for path in (base / "processed").rglob(pattern):
                    if path.is_file():
                        df = _read(path)
                        if df is not None and not df.empty:
                            df = self._prepare_modeldb_df(df)
                            self._modeldb_cache[symbol] = df
                            return df
        except Exception:
            pass

        self._modeldb_cache[symbol] = pd.DataFrame()
        return self._modeldb_cache[symbol]

    def _prepare_modeldb_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame() if df is None else df
        if "_day_date" in df.columns:
            return df.copy()
        df = df.copy()
        day_col = None
        for candidate in ("day_date", "day_id"):
            if candidate in df.columns:
                day_col = candidate
                break
        if day_col:
            day_ts = pd.to_datetime(df[day_col], errors="coerce")
            df["_day_date"] = day_ts.dt.date
        return df

    def _modeldb_row_for_day(self, symbol: Optional[str], ny_day: Any) -> Optional[pd.Series]:
        if not symbol or ny_day is None:
            return None
        if isinstance(ny_day, datetime):
            target_day = ny_day.date()
        elif isinstance(ny_day, date):
            target_day = ny_day
        else:
            try:
                ts = pd.to_datetime(ny_day)
            except Exception:
                return None
            if pd.isna(ts):
                return None
            target_day = ts.date()

        df = self._get_modeldb_for_symbol(symbol)
        if df.empty or "_day_date" not in df.columns or target_day is None:
            return None

        exact = df[df["_day_date"] == target_day]
        if not exact.empty:
            return exact.iloc[-1]

        prior = df[df["_day_date"] < target_day]
        if prior.empty:
            return None
        prior = prior.sort_values("_day_date")
        return prior.iloc[-1]

    def _get_first_range_candle_df(self, symbol: str) -> pd.DataFrame:
        """Load first range candle database rows for a symbol."""
        if symbol in self._first_range_cache:
            return self._first_range_cache[symbol]

        from pathlib import Path

        base = Path(self.data_dir)

        def _read(path: Path) -> Optional[pd.DataFrame]:
            try:
                if path.suffix == ".parquet":
                    return pd.read_parquet(path)
                if path.suffix in (".csv", ".txt"):
                    return pd.read_csv(path)
            except Exception:
                return None
            return None

        candidates: List[Path] = []
        # Symbol-scoped files first (processed/<SYM>/...)
        candidates.extend([
            base / "processed" / symbol / f"{symbol}_first_range_candle_db.parquet",
            base / "processed" / symbol / f"{symbol}_first_range_candle_db.csv",
            base / "processed" / symbol / "first_range_candle_db.parquet",
            base / "processed" / symbol / "first_range_candle_db.csv",
        ])
        # Generic processed files
        candidates.extend([
            base / "processed" / f"{symbol}_first_range_candle_db.parquet",
            base / "processed" / f"{symbol}_first_range_candle_db.csv",
            base / "processed" / "first_range_candle_db.parquet",
            base / "processed" / "first_range_candle_db.csv",
        ])
        # CSV fallbacks
        candidates.extend([
            base / "csv" / symbol / f"{symbol}_first_range_candle_db.csv",
            base / "csv" / symbol / "first_range_candle_db.csv",
            base / "csv" / "first_range_candle_db.csv",
        ])

        for path in candidates:
            if path.exists() and path.is_file():
                df = _read(path)
                if df is not None and not df.empty:
                    self._first_range_cache[symbol] = df
                    return df

        self._first_range_cache[symbol] = pd.DataFrame()
        return self._first_range_cache[symbol]

    def _get_weekly_model_for_symbol(self, symbol: str) -> pd.DataFrame:
        """Load weekly model aggregates for a symbol."""
        if symbol in self._weekly_cache:
            return self._weekly_cache[symbol]

        from pathlib import Path

        base = Path(self.data_dir)

        def _read(path: Path) -> Optional[pd.DataFrame]:
            try:
                if path.suffix == ".parquet":
                    return pd.read_parquet(path)
                if path.suffix in (".csv", ".txt"):
                    return pd.read_csv(path)
            except Exception:
                return None
            return None

        candidates: List[Path] = [
            base / "processed" / symbol / f"{symbol}_weekly_model.parquet",
            base / "processed" / symbol / f"{symbol}_weekly_model.csv",
            base / "processed" / f"{symbol}_weekly_model.parquet",
            base / "processed" / f"{symbol}_weekly_model.csv",
            base / "processed" / "weekly_model.parquet",
            base / "processed" / "weekly_model.csv",
            base / "csv" / symbol / f"{symbol}_weekly_model.csv",
            base / "csv" / f"{symbol}_weekly_model.csv",
            base / "csv" / "weekly_model.csv",
        ]

        for path in candidates:
            if path.exists() and path.is_file():
                df = _read(path)
                if df is not None:
                    self._weekly_cache[symbol] = df
                    return df

        # Recursive fallback in processed/
        try:
            for pattern in (f"*{symbol}*_weekly_model.parquet", f"*{symbol}*_weekly_model.csv"):
                for path in (base / "processed").rglob(pattern):
                    if path.is_file():
                        df = _read(path)
                        if df is not None:
                            self._weekly_cache[symbol] = df
                            return df
        except Exception:
            pass

        self._weekly_cache[symbol] = pd.DataFrame()
        return self._weekly_cache[symbol]

        # Recursive search as a final fallback
        try:
            for pattern in (f"*{symbol}*_first_range_candle_db.parquet", f"*{symbol}*_first_range_candle_db.csv"):
                for path in (base / "processed").rglob(pattern):
                    if path.is_file():
                        df = _read(path)
                        if df is not None and not df.empty:
                            self._first_range_cache[symbol] = df
                            return df
        except Exception:
            pass

        self._first_range_cache[symbol] = pd.DataFrame()
        return self._first_range_cache[symbol]

    @staticmethod
    def _distribution(series: pd.Series, allowed: Tuple[str, ...]) -> Dict[str, float]:
        if series is None:
            total = len(allowed)
            return {val: round(1.0 / total, 4) for val in allowed} if total else {}
        cleaned = series.dropna().astype(str)
        if cleaned.empty:
            total = len(allowed)
            return {val: round(1.0 / total, 4) for val in allowed} if total else {}
        cleaned = cleaned[~cleaned.str.lower().isin({"none", "nan", ""})]
        if cleaned.empty:
            total = len(allowed)
            return {val: round(1.0 / total, 4) for val in allowed} if total else {}
        counts = cleaned.value_counts()
        total = counts.sum()
        if total == 0:
            total = len(allowed)
            return {val: round(1.0 / total, 4) for val in allowed} if total else {}
        return {
            val: float(round(counts.get(val, 0) / total, 4))
            for val in allowed
        }

    @staticmethod
    def _bool_probability(series: pd.Series) -> float:
        if series is None:
            return 0.5
        cleaned = series.dropna()
        if cleaned.empty:
            return 0.5
        if cleaned.dtype == object:
            lower = cleaned.astype(str).str.lower()
            mapping = {"true": True, "false": False, "1": True, "0": False}
            cleaned = lower.map(mapping)
            cleaned = cleaned.dropna()
            if cleaned.empty:
                return 0.5
        try:
            arr = cleaned.astype(bool)
        except Exception:
            return 0.5
        if arr.empty:
            return 0.5
        return float(round(arr.mean(), 4))

    def _get_seq_intraday_for_symbol(self, symbol: str) -> pd.DataFrame:
        if symbol in self._seq_intraday_cache:
            return self._seq_intraday_cache[symbol]
        from pathlib import Path
        base = Path(self.data_dir)
        candidates = [
            base / "processed" / symbol / f"{symbol}_seq_intraday_pairs.parquet",
            base / "processed" / symbol / f"{symbol}_seq_intraday_pairs.csv",
        ]
        df = pd.DataFrame()
        for path in candidates:
            if path.exists():
                try:
                    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
                    break
                except Exception:
                    continue
        if df.empty:
            # Fallback to global files
            for path in [base / "processed" / "seq_intraday_pairs.parquet", base / "csv" / "seq_intraday_pairs.csv"]:
                if path.exists():
                    try:
                        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
                        df = df[df.get("symbol") == symbol]
                        break
                    except Exception:
                        continue
        self._seq_intraday_cache[symbol] = df
        return df

    def _get_seq_daycycle_for_symbol(self, symbol: str) -> pd.DataFrame:
        """Load seq_daycycle_pairs history for a symbol, preferring processed/<SYM>/<SYM>_seq_daycycle_pairs.parquet."""
        if symbol in self._seq_daycycle_cache:
            return self._seq_daycycle_cache[symbol]
        from pathlib import Path
        base = Path(self.data_dir)
        candidates = [
            base / "processed" / symbol / f"{symbol}_seq_daycycle_pairs.parquet",
            base / "processed" / symbol / f"{symbol}_seq_daycycle_pairs.csv",
        ]
        df = pd.DataFrame()
        for path in candidates:
            if path.exists():
                try:
                    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
                    break
                except Exception:
                    continue
        if df.empty:
            # Fallback to global files
            for path in [base / "processed" / "seq_daycycle_pairs.parquet", base / "csv" / "seq_daycycle_pairs.csv"]:
                if path.exists():
                    try:
                        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
                        df = df[df.get("symbol") == symbol]
                        break
                    except Exception:
                        continue
        self._seq_daycycle_cache[symbol] = df
        return df

    def _get_seq_asia_for_symbol(self, symbol: str) -> pd.DataFrame:
        """Load seq_asia_pairs history for a symbol, preferring processed/<SYM>/<SYM>_seq_asia_pairs.parquet."""
        if symbol in self._seq_asia_cache:
            return self._seq_asia_cache[symbol]
        from pathlib import Path
        base = Path(self.data_dir)
        candidates = [
            base / "processed" / symbol / f"{symbol}_seq_asia_pairs.parquet",
            base / "processed" / symbol / f"{symbol}_seq_asia_pairs.csv",
        ]
        df = pd.DataFrame()
        for path in candidates:
            if path.exists():
                try:
                    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
                    break
                except Exception:
                    continue
        if df.empty:
            # Fallback to global files
            for path in [base / "processed" / "seq_asia_pairs.parquet", base / "csv" / "seq_asia_pairs.csv"]:
                if path.exists():
                    try:
                        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
                        df = df[df.get("symbol") == symbol]
                        break
                    except Exception:
                        continue
        self._seq_asia_cache[symbol] = df
        return df

    def _get_wddrb_for_symbol(self, symbol: str) -> pd.DataFrame:
        if symbol in self._wddrb_cache:
            return self._wddrb_cache[symbol]
        from pathlib import Path
        base = Path(self.data_dir)
        candidates = [
            base / "processed" / symbol / f"{symbol}_wddrb.parquet",
            base / "processed" / symbol / "wddrb.parquet",
            base / "processed" / f"{symbol}_wddrb.parquet",
            base / "processed" / "wddrb.parquet",
            base / "csv" / symbol / f"{symbol}_wddrb.csv",
            base / "csv" / "wddrb.csv",
        ]
        df = pd.DataFrame()
        for path in candidates:
            if path.exists():
                try:
                    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
                    break
                except Exception:
                    continue
        if df.empty:
            self._wddrb_cache[symbol] = pd.DataFrame()
            return self._wddrb_cache[symbol]
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol].copy()
        else:
            df = df.copy()

        if "day_date" in df.columns and "_day_date" not in df.columns:
            day_dt = pd.to_datetime(df["day_date"], errors="coerce")
            df["_day_date"] = day_dt.dt.date
        if "_day_date" in df.columns:
            df = df.sort_values("_day_date").reset_index(drop=True)

        def _resolve_bucket(row: pd.Series) -> Optional[str]:
            bucket = self._wddrb_open_bucket_static(
                row.get("next_open"),
                row.get("rdr_lo"),
                row.get("rdr_hi"),
                row.get("rdr_q"),
            )
            if bucket is not None:
                return bucket
            existing = row.get("open_bucket_quarters")
            if isinstance(existing, str) and existing.strip():
                return existing.strip()
            return None

        df["open_bucket_quarters"] = df.apply(_resolve_bucket, axis=1)

        def _normalize(col: str) -> None:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .replace({"": np.nan, "NAN": np.nan})
                )

        for col in [
            "model_rdr_trade_direction",
            "model_rdr_box_direction",
            "next_model_rdr_trade_direction",
            "next_model_rdr_box_direction",
            "asia_range_model_hilo",
            "first_break_side",
            "open_bucket_quarters",
        ]:
            _normalize(col)

        if "model_rdr_session_true" in df.columns:
            df["model_rdr_session_true"] = df["model_rdr_session_true"].astype(bool)
        if "next_model_rdr_session_true" in df.columns:
            df["next_model_rdr_session_true"] = df["next_model_rdr_session_true"].astype(bool)

        if "open_bucket_quarters" not in df.columns:
            df["open_bucket_quarters"] = df.apply(
                lambda row: self._wddrb_open_bucket_static(
                    row.get("next_open"),
                    row.get("rdr_lo"),
                    row.get("rdr_hi"),
                    row.get("rdr_q"),
                ),
                axis=1,
            )

        self._wddrb_cache[symbol] = df
        return df

    @staticmethod
    def _normalize_wddrb_value(value: Optional[Any]) -> Optional[str]:
        if value is None:
            return None
        try:
            txt = str(value).strip().upper()
        except Exception:
            return None
        if txt in {"", "NA", "N/A", "NONE"}:
            return None
        return txt

    def _prior_wddrb_row(self, symbol: str, ny_day: date) -> Optional[pd.Series]:
        history = self._get_wddrb_for_symbol(symbol)
        if history.empty or "_day_date" not in history.columns:
            return None
        mask = history["_day_date"] == ny_day
        if mask.any():
            idx = history.loc[mask].index.max()
            return history.loc[idx]
        prior = history[history["_day_date"] < ny_day]
        if prior.empty:
            return None
        idx = prior.index.max()
        return history.loc[idx]

    def _wddrb_row_for_day(self, symbol: str, ny_day: Optional[date]) -> Optional[pd.Series]:
        if ny_day is None:
            return None
        history = self._get_wddrb_for_symbol(symbol)
        if history.empty or "_day_date" not in history.columns:
            return None
        mask = history["_day_date"] == ny_day
        if not mask.any():
            return None
        return history.loc[mask].iloc[-1]

    def _wddrb_active_day(self, ts_epoch: int, ny_day: Optional[date]) -> Optional[date]:
        """Return the WDDRB day (16:00 prior day → 09:25 current day window)."""
        try:
            bar_dt = datetime.fromtimestamp(ts_epoch, tz=ZoneInfo(NY_TZ))
        except Exception:
            return ny_day

        base_day = ny_day or bar_dt.date()
        t = bar_dt.time()

        if t >= time(16, 0):
            return next_business_day(base_day)
        if t < time(9, 30):
            return base_day
        return bar_dt.date()

    def _wddrb_model_phase(self, bar_dt: datetime) -> str:
        t = bar_dt.time()
        if time(8, 25) <= t <= time(9, 25):
            return "preopen"
        if time(9, 30) <= t < time(15, 55):
            return "intraday"
        return "inactive"

    def _wddrb_model_reference_day(self, ny_day: date) -> date:
        return prev_business_day(ny_day)

    def _current_open_bucket(self, out: Dict[str, Any], ref_row: Optional[pd.Series]) -> Optional[str]:
        if ref_row is None or out is None:
            return None
        price = None
        if bool(out.get("wddrb_open_finalized")):
            price = out.get("wddrb_open_price")
        else:
            price = out.get("wddrb_preopen_price")
        return self._wddrb_open_bucket_static(
            price,
            ref_row.get("rdr_lo"),
            ref_row.get("rdr_hi"),
            ref_row.get("rdr_q"),
        )

    def _update_wddrb_model_state(self, symbol: str, ny_day: date, out: Dict[str, Any]) -> Dict[str, Any]:
        state = self._wddrb_model_state.get(symbol)
        if state is None or state.get("day") != ny_day:
            state = {"day": ny_day}
            self._wddrb_model_state[symbol] = state
        session = (out.get("current_session") or "").strip().upper()
        session_break = (out.get("current_session_break_out_direction") or "").strip().upper()
        if session == "RR" and bool(out.get("current_session_break_out")):
            if session_break in {"LONG", "SHORT"}:
                state["next_model_rdr_trade_direction"] = session_break
        if session == "RS":
            box_dir = (out.get("current_session_box_dir") or "").strip().upper()
            if box_dir in {"UP", "DOWN"}:
                state["next_model_rdr_box_direction"] = box_dir
            if out.get("current_session_true") is not None:
                state["next_model_rdr_session_true"] = bool(out.get("current_session_true"))
        return state

    def _update_intraday_touch_flags(
        self,
        state: Optional[Dict[str, Any]],
        out: Optional[Dict[str, Any]],
        ref_row: Optional[pd.Series],
    ) -> None:
        if state is None or out is None or ref_row is None:
            return
        prev_hi = self._safe_float(ref_row.get("rdr_hi"))
        prev_lo = self._safe_float(ref_row.get("rdr_lo"))
        bar_low = self._safe_float(out.get("low"))
        bar_high = self._safe_float(out.get("high"))
        if prev_hi is not None and bar_low is not None and np.isfinite(prev_hi) and np.isfinite(bar_low):
            if bar_low <= prev_hi:
                state["touched_prev_hi"] = True
        if prev_lo is not None and bar_high is not None and np.isfinite(prev_lo) and np.isfinite(bar_high):
            if bar_high >= prev_lo:
                state["touched_prev_lo"] = True

    def _apply_wddrb_model_filters(
        self,
        history: pd.DataFrame,
        filters: list[tuple[str, Any]],
        min_rows: int,
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        active = [(col, val) for col, val in filters if val is not None]
        if not active:
            return history, {}
        total = len(active)
        min_required = min(min_rows, len(history))
        best_subset = None
        best_applied: Dict[str, Any] | None = None
        for keep in range(total, -1, -1):
            subset = history
            applied: Dict[str, Any] = {}
            for col, val in active[:keep]:
                if col not in subset.columns:
                    continue
                if isinstance(val, str):
                    target = val.strip().upper()
                    series = subset[col].astype(str).str.upper()
                    subset = subset[series == target]
                    applied[col] = target
                else:
                    subset = subset[subset[col] == val]
                    applied[col] = val
                if subset.empty:
                    break
            if len(subset) >= min_required:
                if keep == 0 and best_applied is not None:
                    break
                return subset, applied
            if len(subset) > 0 and (best_applied is None or len(applied) > len(best_applied)):
                best_subset = subset
                best_applied = dict(applied)
        if best_subset is not None and best_applied is not None:
            return best_subset, best_applied
        return history, {}

    def _wddrb_model_probabilities(
        self,
        subset: pd.DataFrame,
        ref_row: Optional[pd.Series] = None,
        out: Optional[Dict[str, Any]] = None,
        phase: str = "inactive",
        state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        if subset is None or subset.empty or "model_code" not in subset.columns:
            uniform = 1.0 / len(WDDRB_MODEL_CODES)
            base = {code: uniform for code in WDDRB_MODEL_CODES}
            return self._constrain_wddrb_model_probs(base, ref_row, out, phase, state)
        series = subset["model_code"].astype(str).str.upper()
        probs = self._categorical_probs(series, WDDRB_MODEL_CODES)
        for code in WDDRB_MODEL_CODES:
            probs.setdefault(code, 1.0 / len(WDDRB_MODEL_CODES))
        return self._constrain_wddrb_model_probs(probs, ref_row, out, phase, state)

    def _constrain_wddrb_model_probs(
        self,
        probs: Dict[str, float],
        ref_row: Optional[pd.Series],
        out: Optional[Dict[str, Any]],
        phase: str,
        state: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        if not probs or ref_row is None or out is None:
            return probs
        if phase != "intraday_model":
            return probs
        adjusted = dict(probs)
        zero_keys: set[str] = set()
        touched_hi = bool(state and state.get("touched_prev_hi"))
        touched_lo = bool(state and state.get("touched_prev_lo"))
        if touched_hi:
            zero_keys.add("UG")
        if touched_lo:
            zero_keys.add("DG")
        if not zero_keys:
            bucket = self._current_open_bucket(out, ref_row)
            if isinstance(bucket, str):
                label = bucket.strip().upper()
                if label == "ABOVE_HIGH":
                    zero_keys.add("UG")
                if label == "BELOW_LOW":
                    zero_keys.add("DG")
        if not zero_keys:
            return adjusted
        changed = False
        for key in zero_keys:
            if key in adjusted and adjusted[key] != 0.0:
                adjusted[key] = 0.0
                changed = True
        if not changed:
            return adjusted
        total = sum(v for v in adjusted.values() if isinstance(v, (int, float)) and v > 0)
        if total <= 0:
            return adjusted
        for key, value in list(adjusted.items()):
            if isinstance(value, (int, float)) and value > 0:
                adjusted[key] = value / total
            else:
                adjusted[key] = 0.0
        return adjusted

    def _adr_filter_window_active(self, bar_dt: datetime) -> bool:
        t = bar_dt.time()
        start = time(20, 25)
        end = time(9, 25)
        return t >= start or t <= end

    def _resolve_wddrb_rdr_targets(
        self,
        out: Dict[str, Any],
        modeldb_row: Optional[pd.Series],
    ) -> tuple[Optional[str], Optional[str], Optional[bool]]:
        def _from_modeldb(col: str) -> Optional[Any]:
            if modeldb_row is None:
                return None
            val = modeldb_row.get(col)
            if isinstance(val, str):
                txt = val.strip()
                return txt if txt else None
            if val is None:
                return None
            if isinstance(val, (float, np.floating)) and not np.isfinite(val):
                return None
            if pd.isna(val):
                return None
            return val

        def _coerce_bool(value: Any) -> Optional[bool]:
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, (int, np.integer)):
                return bool(int(value))
            if isinstance(value, (float, np.floating)):
                if not np.isfinite(value):
                    return None
                return bool(int(value))
            if isinstance(value, str):
                txt = value.strip().lower()
                if txt in {"true", "t", "1", "yes"}:
                    return True
                if txt in {"false", "f", "0", "no"}:
                    return False
            return None

        trade = self._normalize_wddrb_value(
            out.get("model_rdr_trade_direction") or _from_modeldb("RDR_Trade_Direction")
        )
        box = self._normalize_wddrb_value(
            out.get("model_rdr_box_direction") or _from_modeldb("RDR_Box_Direction")
        )
        session_true = _coerce_bool(out.get("model_rdr_session_true"))
        if session_true is None:
            session_true = _coerce_bool(_from_modeldb("RDR_Session_True"))
        return trade, box, session_true

    def _build_wddrb_filters(
        self,
        phase: str,
        adr_model: Optional[str],
        breakout_side: Optional[str],
        rdr_trade: Optional[str],
        rdr_box: Optional[str],
        rdr_true: Optional[bool],
    ) -> tuple[dict, list[str], dict, str]:
        filters: dict = {}
        relax_order: list[str] = []
        applied: dict = {}

        def attach(col: str, value: Any) -> bool:
            if value is None:
                return False
            filters[col] = value
            if col not in relax_order:
                relax_order.append(col)
            applied[col] = value
            return True

        breakout_attached = attach("first_break_side", breakout_side) if breakout_side else False
        adr_attached = False
        rdr_attached = False
        next_attached = False

        if phase == "hold":
            if rdr_trade:
                rdr_attached = attach("model_rdr_trade_direction", rdr_trade) or rdr_attached
            if rdr_box:
                rdr_attached = attach("model_rdr_box_direction", rdr_box) or rdr_attached
            if rdr_true is not None:
                rdr_attached = attach("model_rdr_session_true", bool(rdr_true)) or rdr_attached
            if adr_model:
                adr_attached = attach("asia_range_model_hilo", adr_model)
        elif phase in {"overnight", "preopen"}:
            if adr_model:
                adr_attached = attach("asia_range_model_hilo", adr_model)
        elif phase == "next":
            if rdr_trade:
                next_attached = attach("next_model_rdr_trade_direction", rdr_trade) or next_attached
            if rdr_box:
                next_attached = attach("next_model_rdr_box_direction", rdr_box) or next_attached
            if rdr_true is not None:
                next_attached = attach("next_model_rdr_session_true", bool(rdr_true)) or next_attached

        stage_parts: list[str] = []
        if breakout_attached:
            stage_parts.append("breakout")
        if adr_attached:
            stage_parts.append("adr")
        if rdr_attached:
            stage_parts.append("rdr")
        if next_attached:
            stage_parts.append("next_rdr")

        if not stage_parts:
            stage_parts.append(phase or "unfiltered")

        stage = "_".join(stage_parts) + "_filter"
        return filters, relax_order, applied, stage

    def _wddrb_hold_probability(
        self,
        hist: pd.DataFrame,
        idx_col: str,
        sd_col: str,
        idx_now: Optional[float],
        sd_now: Optional[float],
        side: str,
    ) -> float:
        idx = self._safe_float(idx_now)
        sd_val = self._safe_float(sd_now)
        if hist.empty or idx is None or sd_val is None:
            return 0.5
        rename_map = {
            sd_col: f"combined_{side}_pct",
            idx_col: f"combined_{side}_idx",
        }
        valid_cols = [c for c in rename_map if c in hist.columns]
        if len(valid_cols) < 2:
            return 0.5
        base = hist[valid_cols].rename(columns=rename_map).copy()
        p_bucket = self._hold_prob_bucketed(base, idx, sd_val, side)
        p_reg = self._hold_prob_resid(base, idx, sd_val, side)
        values: list[float] = []
        for candidate in (p_bucket, p_reg):
            try:
                f = float(candidate)
                if np.isfinite(f):
                    values.append(f)
            except (TypeError, ValueError):
                continue
        if not values:
            blend = 0.5
        elif len(values) == 1:
            blend = values[0]
        else:
            blend = 0.5 * (values[0] + values[1])
        weight = self._wddrb_progress_weight(idx)
        adjusted = 0.5 + (blend - 0.5) * weight
        return float(max(0.0, min(1.0, adjusted)))

    @staticmethod
    def _wddrb_open_bucket_static(
        open_price: Any,
        rdr_lo: Any,
        rdr_hi: Any,
        rdr_q: Any,
    ) -> Optional[str]:
        try:
            op = float(open_price)
            lo = float(rdr_lo)
            hi = float(rdr_hi)
        except Exception:
            return None
        if not (np.isfinite(op) and np.isfinite(lo) and np.isfinite(hi)):
            return None
        if hi <= lo:
            return None
        mid = 0.5 * (hi + lo)
        sigma = hi - lo
        if not np.isfinite(sigma) or sigma <= 0:
            return None
        sd = (op - mid) / sigma
        for lower, upper, label in WDDRB_OPEN_SD_BUCKETS:
            if lower <= sd < upper:
                return label
        return None

    def _compute_wddrb_open_bucket(self, open_price: Any, rdr_lo: Any, rdr_hi: Any) -> Optional[str]:
        return self._wddrb_open_bucket_static(open_price, rdr_lo, rdr_hi, None)

    @staticmethod
    def _categorical_probs(series: pd.Series, categories: tuple[str, ...]) -> Dict[str, float]:
        clean = series.dropna().astype(str).str.upper()
        if clean.empty:
            return {}
        base_categories = list(categories)
        extras = sorted(set(clean.unique()) - set(base_categories))
        cats = base_categories + extras
        alpha = 1.0
        total = float(len(clean))
        denom = total + alpha * len(cats)
        probs: Dict[str, float] = {}
        counts = clean.value_counts()
        for code in cats:
            cnt = float(counts.get(code, 0.0))
            probs[code] = (cnt + alpha) / denom
        return probs

    def _wddrb_progress_weight(self, idx: Optional[float]) -> float:
        try:
            if idx is None or not np.isfinite(idx):
                return 0.0
        except Exception:
            return 0.0
        target = getattr(self, "wddrb_progress_buckets", 6.0)
        try:
            target_val = float(target)
        except Exception:
            target_val = 6.0
        if target_val <= 0:
            return 1.0
        weight = float(idx) / target_val
        return float(max(0.0, min(1.0, weight)))

    def _load_box_daily_table(self, symbol: str, kind: str) -> pd.DataFrame:
        """Load box-daily historical table (`day` or `session`) for a symbol with fallbacks."""
        cache = self._box_day_cache if kind == "day" else self._box_session_cache
        if symbol in cache:
            return cache[symbol]

        from pathlib import Path

        base = Path(self.data_dir)
        suffix = f"box_daily_{kind}"
        fname = f"{symbol}_{suffix}"

        candidates = [
            base / "processed" / symbol / f"{fname}.parquet",
            base / "processed" / symbol / f"{fname}.csv",
            base / "csv" / symbol / f"{fname}.parquet",
            base / "csv" / symbol / f"{fname}.csv",
            base / "processed" / f"{fname}.parquet",
            base / "processed" / f"{fname}.csv",
            base / "csv" / f"{fname}.parquet",
            base / "csv" / f"{fname}.csv",
        ]

        df = pd.DataFrame()
        for path in candidates:
            try:
                if path.exists():
                    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
                    break
            except Exception:
                continue

        if not df.empty:
            df = df.copy()
        if "day_id" in df.columns and "day_date" not in df.columns:
            day_ts = pd.to_datetime(df["day_id"], errors="coerce")
            df["day_date"] = day_ts.dt.date
        if "day_date" not in df.columns:
            df["day_date"] = pd.NaT

            if kind == "day" and symbol not in self._box_range_quantiles:
                series = df.get("range_pct_change")
                if series is not None:
                    s = pd.to_numeric(series, errors="coerce").dropna()
                    if len(s):
                        self._box_range_quantiles[symbol] = {
                            "p20": float(s.quantile(0.20)),
                            "p50": float(s.quantile(0.50)),
                            "p80": float(s.quantile(0.80)),
                        }
                    else:
                        self._box_range_quantiles[symbol] = {}
                else:
                    self._box_range_quantiles[symbol] = {}

        cache[symbol] = df
        return df

    def _get_box_day_history(self, symbol: str) -> pd.DataFrame:
        return self._load_box_daily_table(symbol, "day")

    def _get_box_session_history(self, symbol: str) -> pd.DataFrame:
        return self._load_box_daily_table(symbol, "session")

    def _session_tracker(self, anchor_day: date, symbol: str) -> Dict[str, Dict[str, Any]]:
        key = (anchor_day, symbol)
        tracker = self._box_session_tracker.get(key)
        if tracker is None:
            tracker = {sess: {"high": None, "low": None, "last_ts": None} for sess in TRACKED_SESSIONS}
            self._box_session_tracker[key] = tracker
        return tracker

    @staticmethod
    def _size_bucket(range_pct: Optional[float], p20: Optional[float], p50: Optional[float], p80: Optional[float]) -> str:
        try:
            if range_pct is None or not np.isfinite(range_pct):
                return "unknown"
        except Exception:
            return "unknown"
        thresholds = []
        for val in (p20, p50, p80):
            try:
                thresholds.append(float(val) if np.isfinite(val) else np.nan)
            except Exception:
                thresholds.append(np.nan)
        t20, t50, t80 = thresholds
        if not np.isfinite(t20) or not np.isfinite(t50) or not np.isfinite(t80):
            return "unknown"
        if range_pct <= t20:
            return "0-20%"
        if range_pct <= t50:
            return "20-50%"
        if range_pct <= t80:
            return "50-80%"
        return "80%+"

    def _range_size_bucket(self, symbol: str, anchor_day: Optional[date], range_pct: Optional[float]) -> str:
        if range_pct is None or not np.isfinite(range_pct):
            return "unknown"
        df = self._get_box_day_history(symbol)
        if df.empty:
            return "unknown"
        hist = df.copy()
        if anchor_day is not None:
            day_col = None
            if "day_date" in hist.columns:
                day_col = pd.to_datetime(hist["day_date"], errors="coerce")
            elif "day_id" in hist.columns:
                day_col = pd.to_datetime(hist["day_id"], errors="coerce")
            if day_col is not None:
                hist = hist[day_col.dt.date < anchor_day]
        series = pd.to_numeric(hist.get("range_pct_change"), errors="coerce").dropna()
        p20 = p50 = p80 = np.nan
        if len(series) >= 10:
            p20 = float(series.quantile(0.20))
            p50 = float(series.quantile(0.50))
            p80 = float(series.quantile(0.80))
        else:
            fallback = self._box_range_quantiles.get(symbol) or {}
            p20 = fallback.get("p20", np.nan)
            p50 = fallback.get("p50", np.nan)
            p80 = fallback.get("p80", np.nan)
        return self._size_bucket(range_pct, p20, p50, p80)

    def _box_history_filtered(self, df: pd.DataFrame, anchor_day: Optional[date], filters: List[Tuple[str, Any]], min_rows: int = 25) -> pd.DataFrame:
        if df.empty:
            return df
        hist = df.copy()
        if anchor_day is not None:
            day_series = None
            if "day_date" in hist.columns:
                day_series = pd.to_datetime(hist["day_date"], errors="coerce")
            elif "day_id" in hist.columns:
                day_series = pd.to_datetime(hist["day_id"], errors="coerce")
            if day_series is not None:
                hist = hist[day_series.dt.date < anchor_day]
        if hist.empty:
            hist = df.copy()

        filters = [f for f in filters if f[1] is not None]
        if not filters:
            return hist

        def apply(fs: List[Tuple[str, Any]]) -> pd.DataFrame:
            tmp = hist
            for col, val in fs:
                if col not in tmp.columns:
                    continue
                if isinstance(val, (list, tuple, set)):
                    tmp = tmp[tmp[col].isin(list(val))]
                else:
                    tmp = tmp[tmp[col] == val]
            return tmp

        for drop in range(0, len(filters) + 1):
            sub = apply(filters[drop:])
            if len(sub) >= min_rows:
                return sub
        return hist

    @staticmethod
    def _percentile_remaining(series: Optional[pd.Series], value: Optional[float], use_abs: bool = False) -> Optional[float]:
        try:
            if value is None or not np.isfinite(value):
                return None
        except Exception:
            return None
        if series is None:
            return None
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return None
        if use_abs:
            s = s.abs()
            val = abs(float(value))
        else:
            val = float(value)
        remaining = (s >= val).mean()
        remaining = max(0.0, min(1.0, remaining))
        return float(round(float(remaining), 3))

    # ──────────────────────────────────────────────────────────────
    # ModelDB-driven estimators (v3 compact forms)
    # ──────────────────────────────────────────────────────────────
    def _target_and_prior_prefix(self, session: str) -> tuple[str, str]:
        # Map any session to (target range, prior prefix used for context)
        if session in ("AR", "TA", "AS"):
            return "AR", "RDR"     # AR depends on prior RDR
        if session in ("OR", "TO", "OS"):
            return "OR", "ADR"     # OR depends on ADR
        # RR and its neighbors depend on ODR
        return "RR", "ODR"

    def _filter_modeldb_context(self, bar: Bar, out: Dict, target: str, prior_prefix: str) -> pd.DataFrame:
        hist = self._get_modeldb_for_symbol(out.get("symbol") or "")
        if hist.empty:
            return hist
        df = hist[hist.get("symbol") == out.get("symbol")].copy()
        # Day filter
        dn = out.get("day_number")
        if dn is not None and "day_num" in df.columns:
            df = df[df["day_num"] == int(dn)]
        # Prior context from consolidated output
        prior_model = out.get("prior_session_model")
        prior_box   = out.get("prior_session_box_dir")
        prior_true  = out.get("prior_session_true")
        prior_dir   = out.get("prior_session_break_out_direction")
        col_model = f"{prior_prefix}_Model"; col_box = f"{prior_prefix}_Box_Direction"
        col_true = f"{prior_prefix}_Session_True"; col_trade = f"{prior_prefix}_Trade_Direction"
        # Least→most important: box, model, session_true, trade_dir (per latest request trade_dir most important)
        filters: list[tuple[str, object]] = []
        if col_box in df.columns and isinstance(prior_box, str):
            filters.append((col_box, prior_box))
        if col_model in df.columns and isinstance(prior_model, str) and prior_model in ALL_MODELS:
            filters.append((col_model, prior_model))
        if col_true in df.columns and isinstance(prior_true, (bool, type(None))):
            filters.append((col_true, prior_true))
        if col_trade in df.columns and isinstance(prior_dir, str):
            filters.append((col_trade, "Long" if prior_dir == "long" else "Short"))

        def _apply(fs):
            tmp = df.copy()
            for k, v in fs:
                if k in tmp.columns and v is not None:
                    tmp = tmp[tmp[k] == v]
            return tmp

        sub = None
        if filters:
            for drop in range(0, len(filters)+1):
                fs = filters[drop:]
                cand = _apply(fs)
                if len(cand) >= 20:
                    sub = cand; break
            if sub is None:
                sub = _apply([])
        else:
            sub = df

        # If current range (target) is complete, refine by its model+box
        col_target_model = {"AR":"ADR_Model","OR":"ODR_Model","RR":"RDR_Model"}[target]
        col_target_box   = {"AR":"ADR_Box_Direction","OR":"ODR_Box_Direction","RR":"RDR_Box_Direction"}[target]
        cur_model = out.get("current_session_model")
        cur_box = out.get("current_session_box_dir")
        if isinstance(cur_model, str) and (col_target_model in sub.columns):
            sub = sub[sub[col_target_model] == cur_model]
        if isinstance(cur_box, str) and (col_target_box in sub.columns):
            sub = sub[sub[col_target_box] == cur_box]
        return sub

    def _estimate_session_true_from_modeldb(self, bar: Bar, out: Dict) -> float:
        target, prior_prefix = self._target_and_prior_prefix(bar.session)
        sub = self._filter_modeldb_context(bar, out, target, prior_prefix)
        col_true = {"AR":"ADR_Session_True","OR":"ODR_Session_True","RR":"RDR_Session_True"}[target]
        # Direction-aware: if we know the breakout direction, filter by the target trade direction
        col_trade = {"AR":"ADR_Trade_Direction","OR":"ODR_Trade_Direction","RR":"RDR_Trade_Direction"}[target]
        conf_dir = out.get("current_session_break_out_direction") or out.get("prior_session_break_out_direction")
        if isinstance(conf_dir, str) and (col_trade in sub.columns):
            want = "Long" if conf_dir == "long" else "Short" if conf_dir == "short" else None
            if want is not None:
                sub = sub[sub[col_trade] == want]
        if col_true in sub.columns and len(sub) > 0:
            vals = pd.to_numeric(sub[col_true], errors="coerce").dropna().astype(int)
            n = len(vals)
            if n > 0:
                p = (vals == 1).sum()
                # Laplace smoothing
                return round(float((p + 1.0) / (n + 2.0)), 3)
        return 0.5

    def _estimate_confirm_split_from_modeldb(self, bar: Bar, out: Dict) -> tuple[float, float]:
        target, prior_prefix = self._target_and_prior_prefix(bar.session)
        sub = self._filter_modeldb_context(bar, out, target, prior_prefix)
        col_trade = {"AR":"ADR_Trade_Direction","OR":"ODR_Trade_Direction","RR":"RDR_Trade_Direction"}[target]
        if col_trade in sub.columns and len(sub) > 0:
            vals = sub[col_trade].dropna().astype(str).str.title()
            n = len(vals)
            if n > 0:
                pL = (vals == "Long").sum()
                # Laplace smoothing
                p_long = (pL + 1.0) / (n + 2.0)
                return round(float(p_long), 3), round(float(1.0 - p_long), 3)
        return 0.5, 0.5

    # ──────────────────────────────────────────────────────────────
    # Intraday sequencing helpers
    # ──────────────────────────────────────────────────────────────
    def _within_intraday_window(self, ts_epoch: int) -> bool:
        dt = datetime.fromtimestamp(ts_epoch, tz=ZoneInfo(NY_TZ))
        start = dt.replace(hour=4, minute=0, second=0, microsecond=0)
        end = dt.replace(hour=15, minute=55, second=59, microsecond=0)
        if dt.hour < 4:
            start = start - timedelta(days=0)  # same day 04:00
            end = end
        return start <= dt <= end

    def _since_0400_bucket(self, ts_epoch: int, bucket_minutes: int = 30) -> int:
        dt = datetime.fromtimestamp(ts_epoch, tz=ZoneInfo(NY_TZ))
        start = dt.replace(hour=4, minute=0, second=0, microsecond=0)
        if dt < start:
            start = start - timedelta(days=0)
        elapsed = max(0, int((dt - start).total_seconds()))
        bucket = elapsed // (bucket_minutes * 60)
        return min(23, max(0, bucket))

    def _seq_filters_for_now(self, bar: Bar, ny_day: date, t_bucket_5m: Optional[int]) -> dict:
        st = self.engine._get_state(ny_day, bar.symbol)
        f: dict[str, object] = {}

        or_state = st.get("OR")
        if or_state and getattr(or_state, "final_model", None):
            model = str(or_state.final_model)
            if model and not model.upper().startswith("UNDEFINED"):
                f["ODR_Model"] = model
        if or_state and getattr(or_state, "box_std", None) is not None:
            bs = float(or_state.box_std)
            f["ODR_Box_Direction"] = "up" if bs > 0 else "down" if bs < 0 else "flat"

        os_state = st.get("OS")
        if os_state and getattr(os_state, "confirmed", False):
            side = os_state.side
            if side in (1, -1):
                dir_txt = "Long" if side == 1 else "Short"
                f["os_breakout_dir"] = dir_txt
        if os_state:
            false_flag = getattr(os_state, "false_detected", None)
            f["OS_Session_True"] = False if false_flag is True else True

        rr_state = st.get("RR")
        include_rdr = t_bucket_5m is not None and t_bucket_5m >= 77
        if include_rdr and rr_state and getattr(rr_state, "final_model", None):
            model = str(rr_state.final_model)
            if model and not model.upper().startswith("UNDEFINED"):
                f["RDR_Model"] = model

        return f

    def _seq_relax_order(self) -> list[str]:
        return [
            "OS_Session_True",
            "os_breakout_dir",
            "ODR_Box_Direction",
            "ODR_Model",
            "RDR_Model",
        ]

    def _session_windows(self, ny_day: date) -> dict:
        tz = ZoneInfo(NY_TZ)
        base = datetime.combine(ny_day, time(4, 0), tzinfo=tz)
        return {
            "OS": (base, base + timedelta(hours=4, minutes=25)),  # 04:00–08:25
            "TR": (base + timedelta(hours=4, minutes=30), base + timedelta(hours=5, minutes=25)),  # 08:30–09:25
            "RR": (base + timedelta(hours=5, minutes=30), base + timedelta(hours=6, minutes=25)),  # 09:30–10:25
            "RS": (base + timedelta(hours=6, minutes=30), base + timedelta(hours=11, minutes=55)), #10:30–15:55
        }

    def _session_from_5m_idx(self, ny_day: date, idx: int) -> Optional[str]:
        if idx is None:
            return None
        wins = self._session_windows(ny_day)
        ts = datetime.combine(ny_day, time(4, 0), tzinfo=ZoneInfo(NY_TZ)) + timedelta(minutes=5*idx)
        for name, (start, end) in wins.items():
            if start <= ts <= end:
                return name
        return "RS" if ts > wins["RS"][1] else None

    def _allowed_sessions(self, ny_day: date, current_bucket: Optional[int], extreme_idx: Optional[int]) -> set[str]:
        session_order = ["OS", "TR", "RR", "RS"]
        wins = self._session_windows(ny_day)
        base = datetime.combine(ny_day, time(4, 0), tzinfo=ZoneInfo(NY_TZ))

        extreme_session = self._session_from_5m_idx(ny_day, extreme_idx)
        allowed: set[str] = set()

        for sess in session_order:
            start_dt, end_dt = wins[sess]
            end_bucket = int(max(0, (end_dt - base).total_seconds() // 300))
            if extreme_session == sess:
                allowed.add(sess)
                continue
            if current_bucket is not None and current_bucket >= end_bucket:
                # session window has ended and did not produce the extreme → drop it
                continue
            allowed.add(sess)

        if not allowed and extreme_session:
            allowed.add(extreme_session)
        if not allowed:
            allowed.update(session_order)
        return allowed

    def _map_seq_filter_keys(self, filters: dict, df: pd.DataFrame) -> tuple[dict, list[str]]:
        mapping = {
            "ODR_Model": ["ODR_Model"],
            "ODR_Box_Direction": ["OR_Box_Direction"],
            "os_breakout_dir": ["OS_confirm_dir"],
            "OS_Session_True": ["OS_stays_true", "OS_Session_True"],
            "RDR_Model": ["RDR_Model", "NextRDR_Model"],
        }

        mapped: dict[str, object] = {}
        for key, val in filters.items():
            candidates = mapping.get(key, [key])
            for col in candidates:
                if col in df.columns:
                    mapped[col] = val
                    break

        relax_cols: list[str] = []
        for key in self._seq_relax_order():
            candidates = mapping.get(key, [key])
            for col in candidates:
                if col in mapped and col not in relax_cols:
                    relax_cols.append(col)
                    break

        return mapped, relax_cols

    def _apply_filters_with_backoff(self, df: pd.DataFrame, filters: dict, relax_order: list[str], min_rows: int) -> pd.DataFrame:
        if df.empty or not filters:
            return df

        def apply(keys):
            sub = df
            for k in keys:
                if k in sub.columns:
                    sub = sub[sub[k] == filters[k]]
            return sub

        active = [k for k in filters.keys() if k in df.columns]
        sub = apply(active)
        if len(sub) >= min_rows:
            return sub

        keep = active[:]
        for col in relax_order:
            if col not in keep:
                continue
            keep = [k for k in keep if k != col]
            sub = apply(keep)
            if len(sub) >= min_rows:
                return sub

        return sub if len(sub) else df

    def _seq_cols(self, H: pd.DataFrame, side: str) -> tuple[Optional[str], Optional[str]]:
        def pick(columns: list[str]) -> Optional[str]:
            for c in columns:
                if c in H.columns:
                    return c
            return None

        if side == "high":
            pct = pick(["combined_high_pct_change", "high_pct", "combined_high_pct"])
            tme = pick(["combined_high_idx", "combined_high_time_bucket", "combined_high_time", "high_time_bucket"])
        else:
            pct = pick(["combined_low_pct_change", "low_pct", "combined_low_pct"])
            tme = pick(["combined_low_idx", "combined_low_time_bucket", "combined_low_time", "low_time_bucket"])
        return pct, tme

    def _quantiles(self, s, qs):
        if s is None:
            return [None for _ in qs]
        try:
            series = pd.Series(s).dropna()
            if series.empty:
                return [None for _ in qs]
            return list(series.quantile(qs).values)
        except Exception:
            return [None for _ in qs]

    def _seq_location_probs(self, H: pd.DataFrame, which: str, allowed: Optional[set[str]] = None) -> tuple[float, float, float, float]:
        if H.empty:
            return (0.25, 0.25, 0.25, 0.25)
        col = "combined_high_where" if which == "high" else "combined_low_where"
        if col not in H.columns:
            return (0.25, 0.25, 0.25, 0.25)
        vals = H[col].astype(str).str.upper().str.strip()
        keys = ["OS", "TR", "RR", "RS"]
        counts = {k: int((vals == k).sum()) for k in keys}
        if allowed is not None:
            alpha = 1.0  # Dirichlet prior to keep allowed sessions in play
            total = sum(counts[k] + (alpha if k in allowed else 0.0) for k in keys)
            if total <= 0:
                uniform = 1.0 / len(allowed) if len(allowed) > 0 else 0.25
                return tuple(uniform if k in allowed else 0.0 for k in keys)
            return tuple(
                (counts[k] + (alpha if k in allowed else 0.0)) / total if k in allowed else 0.0
                for k in keys
            )
        total = sum(counts.values())
        if total <= 0:
            return (0.25, 0.25, 0.25, 0.25)
        return tuple(counts[k] / total for k in keys)

    def _seq_where_cols(self, H: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
        def pick(options: list[str]) -> Optional[str]:
            for col in options:
                if col in H.columns:
                    return col
            return None
        hi = pick(["combined_high_where", "day_high_where", "high_where"])
        lo = pick(["combined_low_where", "day_low_where", "low_where"])
        return hi, lo

    def _zero_impossible_sessions(self, probs: tuple[float, float, float, float], idx: Optional[int], ny_day: date, side: str, current_bucket: Optional[int]) -> tuple[float, float, float, float]:
        session_order = ["OS", "TR", "RR", "RS"]
        p_list = list(probs)

        extreme_session = None
        if idx is not None:
            extreme_session = self._session_from_5m_idx(ny_day, idx)
            if extreme_session in session_order:
                cutoff = session_order.index(extreme_session)
                for i in range(cutoff):
                    p_list[i] = 0.0

        if current_bucket is not None:
            windows = self._session_windows(ny_day)
            base = datetime.combine(ny_day, time(4, 0), tzinfo=ZoneInfo(NY_TZ))
            for sess in session_order:
                start_dt, end_dt = windows[sess]
                end_bucket = int(max(0, (end_dt - base).total_seconds() // 300))
                if current_bucket >= end_bucket and sess != extreme_session:
                    idx_sess = session_order.index(sess)
                    p_list[idx_sess] = 0.0

        total = sum(p_list)
        if total <= 0:
            return tuple(0.0 for _ in session_order)
        return tuple(p / total for p in p_list)

    def _hold_prob_bucketed(self, H: pd.DataFrame, t_bucket_now: Optional[int], price_pos_now: Optional[float], side: str) -> float:
        if H.empty or t_bucket_now is None or price_pos_now is None:
            return 0.5
        pct_col, t_col = self._seq_cols(H, side)
        if not pct_col or not t_col:
            return 0.5
        base = H.copy()
        q = self._quantiles(base[pct_col], [0.20, 0.50, 0.80, 0.85])
        q20, q50, q80, q85 = q

        def price_bin(v):
            if v is None or q20 is None:
                return "nan"
            if v < q20:
                return "<20"
            if v < q50:
                return "20-50"
            if v < q80:
                return "50-80"
            if q85 is not None and v < q85:
                return "80-85"
            return "≥85"

        base["price_bin"] = base[pct_col].apply(price_bin)
        time_vals = pd.to_numeric(base[t_col], errors="coerce").fillna(0).astype(int)
        base["time_bin"] = time_vals

        tb = int(t_bucket_now)
        pb = price_bin(price_pos_now)
        sub = self._apply_filters_with_backoff(base, {"time_bin": tb, "price_bin": pb}, ["price_bin", "time_bin"], self.MIN_SEQ_ROWS)
        if sub.empty:
            return 0.5
        time_vals_sub = pd.to_numeric(sub[t_col], errors="coerce").fillna(0)
        y = (time_vals_sub <= tb).astype(int)
        n = len(y)
        if n == 0:
            return 0.5
        raw = float((y.sum() + 1.0) / (n + 2.0))  # Laplace smoothing
        shrink = min(1.0, n / 40.0)  # require larger sample for confident extremes
        return 0.5 + shrink * (raw - 0.5)

    def _hold_prob_resid(self, H: pd.DataFrame, t_bucket_now: Optional[int], price_pos_now: Optional[float], side: str) -> float:
        if H.empty or t_bucket_now is None or price_pos_now is None:
            return 0.5
        pct_col, t_col = self._seq_cols(H, side)
        if not pct_col or not t_col:
            return 0.5
        df = H[[pct_col, t_col]].copy()
        df[pct_col] = pd.to_numeric(df[pct_col], errors="coerce")
        df[t_col] = pd.to_numeric(df[t_col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if df.empty:
            return self._hold_prob_bucketed(H, t_bucket_now, price_pos_now, side)

        time_vals = df[t_col]
        if time_vals.max() <= 0:
            return 0.5
        if len(df) < 2:
            return self._hold_prob_bucketed(H, t_bucket_now, price_pos_now, side)
        x2 = (time_vals / float(time_vals.max())).astype(float)
        x1 = df[pct_col].astype(float)
        X = np.c_[np.ones(len(x2)), x2]
        try:
            beta = np.linalg.lstsq(X, x1, rcond=None)[0]
            resid = x1 - (X @ beta)
        except Exception:
            return self._hold_prob_bucketed(H, t_bucket_now, price_pos_now, side)

    def _laplace_mean(self, labels: pd.Series) -> float:
        """Laplace-smoothed proportion for binary labels."""
        if labels is None or len(labels) == 0:
            return 0.5
        y = pd.to_numeric(labels, errors="coerce").dropna()
        if y.empty:
            return 0.5
        total = float(len(y))
        return float(((y.sum() if total else 0.0) + 1.0) / (total + 2.0))

    def _percentile_reached(
        self,
        series: Optional[pd.Series],
        current_value: Optional[float],
        use_abs: bool = False,
        head: bool = False,
    ) -> Optional[float]:
        """Percentile helper (tail by default, head when head=True)."""
        if series is None or current_value is None:
            return None
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return None
        try:
            if use_abs:
                values = values.abs()
                val = abs(float(current_value))
            else:
                val = float(current_value)
        except Exception:
            return None
        frac = (values <= val).mean() if head else (values >= val).mean()
        if not np.isfinite(frac):
            return None
        return float(max(0.0, min(1.0, frac)))

    def _safe_float(self, value: Any) -> Optional[float]:
        try:
            f = float(value)
            if np.isfinite(f):
                return f
        except Exception:
            pass
        return None

    def _weekly_residual_probability(
        self,
        df: pd.DataFrame,
        price_col: str,
        day_col: str,
        eval_day: Optional[int],
        current_price: Optional[float],
        label_builder: Callable[[pd.Series], pd.Series],
        side: str,
        min_rows: int = 10,
    ) -> float:
        """Residual-based probability for weekly high/low holding logic."""
        if (
            df.empty
            or price_col not in df.columns
            or day_col not in df.columns
            or eval_day is None
            or current_price is None
        ):
            return 0.5

        price = pd.to_numeric(df[price_col], errors="coerce")
        day_vals = pd.to_numeric(df[day_col], errors="coerce")
        mask = price.notna() & day_vals.notna()
        if not mask.any():
            return 0.5

        data = pd.DataFrame({"price": price[mask], "day": day_vals[mask]})
        if data.empty:
            return 0.5

        labels = label_builder(data["day"])
        labels = pd.to_numeric(labels, errors="coerce")
        data = data.assign(label=labels).dropna()
        if data.empty:
            return 0.5

        if len(data) < min_rows or data["label"].nunique() <= 1:
            return self._laplace_mean(data["label"])

        prices = data["price"].astype(float).to_numpy()
        if side == "low":
            prices = np.abs(prices)
            current_val = abs(float(current_price))
        else:
            current_val = float(current_price)
        if not np.isfinite(current_val):
            return 0.5

        days = data["day"].astype(float).to_numpy()
        max_day = 5.0
        X = np.column_stack([np.ones(len(days)), days / max_day])

        try:
            beta = np.linalg.lstsq(X, prices, rcond=None)[0]
        except Exception:
            return self._laplace_mean(data["label"])

        residuals = prices - (X @ beta)
        y = data["label"].astype(float).to_numpy()
        if y.size == 0:
            return 0.5

        R = np.column_stack([np.ones(len(residuals)), residuals])
        w = np.zeros(2)
        lam = 1e-3
        for _ in range(15):
            z = R @ w
            p = 1.0 / (1.0 + np.exp(-z))
            W = p * (1 - p)
            if np.allclose(W, 0):
                break
            H = R.T @ (W[:, None] * R) + lam * np.eye(2)
            g = R.T @ (y - p) - lam * w
            try:
                step = np.linalg.lstsq(H, g, rcond=None)[0]
            except Exception:
                return self._laplace_mean(data["label"])
            w += step
            if np.linalg.norm(step) < 1e-6:
                break

        eval_scaled = float(eval_day) / max_day
        resid_now = current_val - (beta[0] + beta[1] * eval_scaled)
        z_now = w[0] + w[1] * resid_now
        prob = 1.0 / (1.0 + np.exp(-z_now))
        if not np.isfinite(prob):
            return self._laplace_mean(data["label"])
        return float(np.clip(prob, 0.001, 0.999))

        y = (time_vals <= t_bucket_now).astype(int)
        if len(y) < self.MIN_SEQ_ROWS or y.nunique() < 2:
            return self._hold_prob_bucketed(H, t_bucket_now, price_pos_now, side)
        R = np.c_[np.ones(len(resid)), resid.values]
        w = np.zeros(2)
        try:
            for _ in range(12):
                z = R @ w
                p = 1.0 / (1.0 + np.exp(-z))
                W = p * (1 - p)
                lam = 1e-3
                Hn = R.T @ (W[:, None] * R) + lam * np.eye(2)
                gn = R.T @ (y - p) - lam * w
                step = np.linalg.lstsq(Hn, gn, rcond=None)[0]
                w = w + step
                if np.linalg.norm(step) < 1e-6:
                    break
            # Predict now
            t_norm = float(t_bucket_now) / max(1, time_vals.max())
            r_now = float(price_pos_now) - (beta[0] + beta[1] * t_norm)
            z_now = w[0] + w[1] * r_now
            return float(1.0 / (1.0 + np.exp(-z_now)))
        except Exception:
            return self._hold_prob_bucketed(H, t_bucket_now, price_pos_now, side)

    def _estimate_first_mid_break_from_modeldb(self, bar: Bar, out: Dict) -> tuple[float, float]:
        """Estimate P(first mid break is 'asia') vs 'overnight' using prior-session context filters.

        Filtering strategy:
          - Choose target range (AR/OR/RR) and its prior session code.
          - Filter model_db rows for this symbol by:
            day_num (if available), and prior session context: model, box direction, trade direction, session_true.
          - If sample too small, relax in order: session_true, trade_dir, model, box_dir.
          - Compute share of 'asia'/'overnight' in First_Mid_Break.
        """
        hist = self._get_modeldb_for_symbol(bar.symbol)
        if hist.empty or "First_Mid_Break" not in hist.columns:
            return 0.5, 0.5

        # Determine target range and map to prior session prefix used in model_db columns
        sess = bar.session
        if sess in ("AR", "TA", "AS"):
            target = "AR"; prior_prefix = "RDR"  # AR uses previous RDR context
        elif sess in ("OR", "TO", "OS"):
            target = "OR"; prior_prefix = "ADR"  # OR uses ADR context
        else:
            target = "RR"; prior_prefix = "ODR"  # RR uses ODR context

        df = hist[hist.get("symbol") == bar.symbol].copy()
        # Day filter (exact day_num when present)
        day_num = out.get("day_number")
        if day_num is not None and "day_num" in df.columns:
            df = df[df["day_num"] == int(day_num)]

        # Build prior context from consolidated output
        prior_model = out.get("prior_session_model")  # e.g., 'UX','RC',...
        prior_box = out.get("prior_session_box_dir")  # 'up','down','flat'
        prior_true = out.get("prior_session_true")    # True/False/None
        prior_dir = out.get("prior_session_break_out_direction")  # 'long'/'short'/None
        # Map to model_db column names
        col_model = f"{prior_prefix}_Model"
        col_box   = f"{prior_prefix}_Box_Direction"
        col_true  = f"{prior_prefix}_Session_True"
        col_trade = f"{prior_prefix}_Trade_Direction"

        # Apply filters with graceful backoff (least → most important):
        # box_dir, model, session_true, trade_dir
        filters = []
        if col_box in df.columns and isinstance(prior_box, str):
            filters.append((col_box, prior_box))
        if col_model in df.columns and isinstance(prior_model, str) and prior_model in ALL_MODELS:
            filters.append((col_model, prior_model))
        if col_true in df.columns and isinstance(prior_true, (bool, type(None))):
            filters.append((col_true, prior_true))
        if col_trade in df.columns and isinstance(prior_dir, str):
            filters.append((col_trade, "Long" if prior_dir == "long" else "Short"))

        # Try progressively relaxed filtering
        def _apply(fs):
            tmp = df.copy()
            for k, v in fs:
                if k in tmp.columns and v is not None:
                    tmp = tmp[tmp[k] == v]
            return tmp

        # Relax filters progressively in the order they were added (box → model → true → trade).
        # Start with all filters; if too small, drop the earliest (least important) first, etc.
        if filters:
            for drop_count in range(0, len(filters) + 1):
                fs = filters[drop_count:]
                sub = _apply(fs)
                if len(sub) >= 20:  # minimum sample size
                    break
            else:
                sub = _apply([])
        else:
            sub = df.copy()

        if sub.empty or "First_Mid_Break" not in sub.columns:
            return 0.5, 0.5

        s = sub["First_Mid_Break"].dropna().astype(str)
        n = len(s)
        if n <= 0:
            return 0.5, 0.5
        asia = (s == "asia").sum()
        overnight = (s == "overnight").sum()
        # Laplace smoothing
        p_asia = (asia + 1.0) / (n + 2.0)
        p_ovr  = (overnight + 1.0) / (n + 2.0)
        return float(p_asia), float(p_ovr)

    # ──────────────────────────────────────────────────────────────
    # Core ingestion
    # ──────────────────────────────────────────────────────────────
    def ingest_bar(self, bar: Bar) -> Dict:
        """Ingest a bar and return the consolidated output snapshot."""
        out = self.engine.ingest_bar(bar)
        ny_day = out.get("ny_day")
        if isinstance(ny_day, pd.Timestamp):
            ny_day = ny_day.date()
        self._last[(ny_day, out["symbol"])] = _Last(ny_day=ny_day, symbol=out["symbol"], out=out)
        return out

    def _last_out(self, bar: Bar) -> Optional[Dict]:
        ny_day = self.engine._ny_date_from_epoch(bar.ts)
        key = (ny_day, bar.symbol)
        rec = self._last.get(key)
        return rec.out if rec else None

    def _ensure_latest_out(self, bar: Bar) -> Dict:
        """Return cached consolidated snapshot if it already covers this bar."""
        last = self._last_out(bar)
        if last is not None and last.get("ts") == bar.ts:
            return last
        return self.ingest_bar(bar)

    # ──────────────────────────────────────────────────────────────
    # Range session metrics
    # ──────────────────────────────────────────────────────────────
    def get_range_session_metrics_v3(self, bar: Bar) -> RangeSessionOut:
        """Map consolidated fields to RangeSessionOut.

        Notes:
        - box_std: from current session when available
        - pre-confirm probabilities default to 0.5 until historical layer added
        - post-confirm metrics for follow sessions use tracked max_ext/max_ret (SD)
        """
        # Ensure state is up-to-date
        out = self._ensure_latest_out(bar)

        rs = RangeSessionOut()

        # Set day number
        if out.get("day_number") is not None:
            rs.day_number = int(out["day_number"])

        # Session label probabilities (pre/post confirm are set below)
        box_std = out.get("current_session_box_std")
        rs.box_std = float(round(box_std, 2)) if box_std is not None else 0.0
        rs.p_confirm_long = 0.5
        rs.p_confirm_short = 0.5

        # Helper: box_std bucket (1-decimal) for robust matching
        def _box_bucket(val):
            try:
                f = float(val)
                if np.isfinite(f):
                    return float(np.round(f, 1))
            except Exception:
                pass
            return None

        # Candle index (follow or range sessions)
        rs.candle_index = out.get("current_time_idx") if bar.session in ("OS","TR","RR","RS") else None

        # Pre-confirm p_long/p_short using Box_STD bucketing (0.1) with optional day filter.
        try:
            # Local bucket helper matching v1 semantics
            def _bucket(val: Optional[float], step: float = 0.1) -> Optional[float]:
                try:
                    if val is None:
                        return None
                    v = float(val)
                    if not np.isfinite(v):
                        return None
                    b = (v // step) * step
                    return round(float(b), 2)
                except Exception:
                    return None

            df_all = self._get_metrics_for_symbol(bar.symbol)
            if not df_all.empty:
                # Resolve target range
                target_range = bar.session if bar.session in ("AR","OR","RR") else {"AS":"AR","OS":"OR","RS":"RR"}.get(bar.session)
                session_col = "range_session" if "range_session" in df_all.columns else ("Range_Session" if "Range_Session" in df_all.columns else None)
                base = df_all[(df_all.get("symbol") == bar.symbol) & (df_all[session_col] == target_range)].copy() if (target_range and session_col) else pd.DataFrame()

                # Day filter (no backoff to match v1; remove if not present)
                if (out.get("day_number") is not None) and ("day_num" in base.columns):
                    base = base[base["day_num"] == int(out["day_number"]) ]

                # Bucket current values
                b_box = _bucket(box_std, 0.1)

                sub = base.copy()
                # Apply bucket filters
                if (b_box is not None) and ("Box_STD" in sub.columns):
                    sub = sub[np.isfinite(sub["Box_STD"])].copy()
                    sub["Box_STD_bucketed"] = sub["Box_STD"].apply(lambda x: _bucket(x, 0.1))
                    sub = sub[sub["Box_STD_bucketed"] == b_box].drop(columns=["Box_STD_bucketed"])

                if ("Confirmation_Side" in sub.columns) and (len(sub) > 0):
                    cs = pd.to_numeric(sub["Confirmation_Side"], errors="coerce").dropna().astype(float)
                    total = float(len(cs))
                    if total > 0:
                        rs.p_confirm_long = round(float((cs == 1.0).sum()) / total, 2)
                        rs.p_confirm_short = round(float((cs == 0.0).sum()) / total, 2)
        except Exception:
            pass

        # Post-confirmation mapping for follow sessions
        if bar.session in ("AS", "OS", "RS"):
            rs.is_post_confirm = bool(out.get("current_session_break_out", False))

            # Direction of confirmation: prefer current session's breakout direction, fallback to prior session
            cur_dir = out.get("current_session_break_out_direction")
            prior_dir = out.get("prior_session_break_out_direction")
            dir_val = cur_dir or prior_dir
            if isinstance(dir_val, str):
                rs.confirm_side = "long" if dir_val == "long" else "short" if dir_val == "short" else "none"

            # Track progress metrics (already SD-scaled in consolidated state)
            max_ext_sd = out.get("current_session_max_ext")
            max_ret_sd = out.get("current_session_max_ret")
            rs.current_ext_sd = float(round(max_ext_sd, 2)) if max_ext_sd is not None else None
            rs.current_ret_sd = float(round(max_ret_sd, 2)) if max_ret_sd is not None else None
            rs.max_ext_candle_index = out.get("current_session_time_idx_extension")
            rs.max_ret_candle_index = out.get("current_session_time_idx_retracement")
            rs.confirm_candle_index = out.get("current_session_break_out_idx")

            # Resolve anchor day for sessions that cross midnight (AS)
            ny_day_val = out.get("ny_day")
            if isinstance(ny_day_val, pd.Timestamp):
                ny_day_val = ny_day_val.date()
            if bar.session == "AS":
                anchor_day = self.engine._as_anchor_day(ny_day_val, bar.ts, bar.symbol)
            else:
                anchor_day = ny_day_val

            # Compute live session close location (SD) relative to prior range IDR extremes
            try:
                # Map follow session to its prior range session code
                follow_to_prior = {"AS": "AR", "OS": "OR", "RS": "RR"}
                prior_code = follow_to_prior.get(bar.session)
                # Use the resolved direction (current or prior) for sign
                if prior_code and isinstance(dir_val, str) and dir_val in ("long", "short"):
                    # Pull prior range state for current trading day
                    st = self.engine._get_state(anchor_day, bar.symbol)
                    prior_state = st.get(prior_code)
                    if prior_state and prior_state.idr_hi_close is not None and prior_state.idr_lo_close is not None:
                        idr_hi = float(prior_state.idr_hi_close)
                        idr_lo = float(prior_state.idr_lo_close)
                        idr_std = idr_hi - idr_lo
                        if idr_std and idr_std > 0:
                            if dir_val == "long":
                                rs.session_close_std = round(float((bar.c - idr_hi) / idr_std), 2)
                            else:
                                rs.session_close_std = round(float((idr_lo - bar.c) / idr_std), 2)
            except Exception:
                pass

            # Historical percentiles filtered by day/box/confirmation (+ time when available)
            try:
                df = self._get_metrics_for_symbol(bar.symbol)
                if not df.empty:
                    # Map follow session to its prior range session code
                    follow_to_prior = {"AS": "AR", "OS": "OR", "RS": "RR"}
                    prior_code = follow_to_prior.get(bar.session)
                    if prior_code:
                        sub = df[(df.get("symbol") == bar.symbol) & (df.get("range_session") == prior_code)].copy()
                    else:
                        sub = pd.DataFrame()

                    # Day number filter
                    if (rs.day_number is not None) and ("day_num" in sub.columns):
                        sub = sub[sub["day_num"] == rs.day_number]
                    # Note: do NOT filter by Box_STD here; false-day/time distributions
                    # should be conditioned by day + direction (+ confirmation idx bucket),
                    # not by box bucket to keep sufficient sample size.

                    # Confirmation side filter (-1 -> 0.0, 1 -> 1.0)
                    if isinstance(dir_val, str) and "Confirmation_Side" in sub.columns:
                        hist_side = 1.0 if dir_val == "long" else 0.0 if dir_val == "short" else None
                        if hist_side is not None:
                            sub = sub[np.isfinite(sub["Confirmation_Side"]) & (sub["Confirmation_Side"].astype(float) == hist_side)]

                    # For Session_Close_STD distribution, do NOT restrict by confirmation idx;
                    # use the side/day filtered subset (more stable sample).
                    sub_close = sub.copy()

                    # Confirmation idx bucket (int bins of 1) for time-sensitive metrics only
                    if rs.confirm_candle_index is not None and "Confirmation_Idx" in sub.columns:
                        idx_series = pd.to_numeric(sub["Confirmation_Idx"], errors="coerce")
                        sub = sub[idx_series.notna()]
                        sub["_idx_bucket"] = idx_series.astype(int)
                        sub = sub[sub["_idx_bucket"] == int(rs.confirm_candle_index)]

                    if len(sub) > 0:
                        # False session probability with backoff and smoothing to avoid 0.0/1.0 spikes
                        if "False_Session" in sub.columns:
                            valid = pd.to_numeric(sub["False_Session"], errors="coerce").dropna().astype(float)
                            p_false = None
                            n = int(len(valid))
                            if n >= 20:
                                false_count = int((valid == 1.0).sum())
                                # Laplace smoothing
                                p_false = (false_count + 1.0) / (n + 2.0)
                            else:
                                # Backoff to day+direction slice (no idx bucket) for more samples
                                base_vals = pd.to_numeric(sub_close.get("False_Session", pd.Series(dtype=float)), errors="coerce").dropna().astype(float)
                                n2 = int(len(base_vals))
                                if n2 > 0:
                                    false_count2 = int((base_vals == 1.0).sum())
                                    p_false = (false_count2 + 1.0) / (n2 + 2.0)
                            if p_false is not None:
                                rs.p_false_session = round(float(p_false), 2)

                        # Directional confirmation probabilities post-confirm
                        if rs.is_post_confirm and isinstance(dir_val, str) and "False_Session" in sub.columns:
                            vals = pd.to_numeric(sub["False_Session"], errors="coerce").dropna().astype(float)
                            if len(vals) > 0:
                                p_true = float((vals == 0.0).mean())
                                p_false = float((vals == 1.0).mean())
                                if dir_val == "long":
                                    rs.p_confirm_long = round(p_true, 2)
                                    rs.p_confirm_short = round(p_false, 2)
                                else:
                                    rs.p_confirm_short = round(p_true, 2)
                                    rs.p_confirm_long = round(p_false, 2)

                        # Percentiles for progress metrics
                        max_ext = rs.current_ext_sd if rs.current_ext_sd is not None else 0.0
                        max_ret = rs.current_ret_sd if rs.current_ret_sd is not None else None

                        if "M7_Retracement_STD" in sub.columns and max_ret is not None:
                            hist = pd.to_numeric(sub["M7_Retracement_STD"], errors="coerce").dropna()
                            if len(hist) > 0:
                                rs.p_m7_ret_sd = round(float((hist < max_ret).mean()), 2)

                        if "Max_Extension_STD" in sub.columns:
                            hist = pd.to_numeric(sub["Max_Extension_STD"], errors="coerce").dropna()
                            if len(hist) > 0:
                                rs.p_max_ext_sd = round(float((hist < max_ext).mean()), 2)

                        if "Max_Retracement_STD" in sub.columns and max_ret is not None:
                            hist = pd.to_numeric(sub["Max_Retracement_STD"], errors="coerce").dropna()
                            if len(hist) > 0:
                                rs.p_max_ret_sd = round(float((hist < max_ret).mean()), 2)

                        # Percentile for session close location vs historical Session_Close_STD
                        if (rs.session_close_std is not None) and ("Session_Close_STD" in sub_close.columns):
                            hist = pd.to_numeric(sub_close["Session_Close_STD"], errors="coerce").dropna()
                            if len(hist) > 0:
                                rs.p_session_close_std = round(float((hist < rs.session_close_std).mean()), 2)

                        # Time percentiles using scaled time within session
                        windows = self.engine._get_session_windows(anchor_day) if hasattr(self.engine, "_get_session_windows") else None
                        if isinstance(windows, dict) and bar.session in windows:
                            sess_start, sess_end = windows[bar.session]
                            # Use timezone-aware to match window tz
                            try:
                                from zoneinfo import ZoneInfo
                                from ..constants import NY_TZ
                                bar_dt = datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ))
                            except Exception:
                                bar_dt = datetime.fromtimestamp(bar.ts)
                            total_secs = max(1.0, (sess_end - sess_start).total_seconds())
                            current_scaled = (bar_dt - sess_start).total_seconds() / total_secs
                        else:
                            current_scaled = 0.5

                        if "M7_Scaled" in sub.columns:
                            m7 = pd.to_numeric(sub["M7_Scaled"], errors="coerce").dropna()
                            if len(m7) > 0:
                                rs.p_m7_time = round(float((m7 < current_scaled).mean()), 2)

                        if "Max_Ext_Scaled" in sub.columns:
                            ext = pd.to_numeric(sub["Max_Ext_Scaled"], errors="coerce").dropna()
                            if len(ext) > 0:
                                rs.p_max_ext_time = round(float((ext < current_scaled).mean()), 2)

                        if "Max_Ret_Scaled" in sub.columns:
                            ret = pd.to_numeric(sub["Max_Ret_Scaled"], errors="coerce").dropna()
                            if len(ret) > 0:
                                rs.p_max_ret_time = round(float((ret < current_scaled).mean()), 2)
            except Exception:
                pass

            # False session derived flag from consolidated (do not override probabilities)
            rs.is_false_session = bool(out.get("current_session_break_out") and (out.get("current_session_true") is False))

        return rs

    # ──────────────────────────────────────────────────────────────
    # Daily box context (Asia-normalized)
    # ──────────────────────────────────────────────────────────────
    def get_box_daily_context_v3(self, bar: Bar) -> BoxDailyOut:
        """Compute Asia-normalized daily box context with volatility sizing and percentiles."""
        out = self._ensure_latest_out(bar)

        ny_day = self.engine._ny_date_from_epoch(bar.ts)
        try:
            bar_dt = datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ))
        except Exception:
            bar_dt = datetime.fromtimestamp(bar.ts)
        anchor_day = self.engine._asia_day_anchor(bar_dt) if hasattr(self.engine, "_asia_day_anchor") else ny_day

        state_today = self.engine._get_state(ny_day, bar.symbol)
        state_anchor = state_today if anchor_day == ny_day else self.engine._get_state(anchor_day, bar.symbol)
        ar_state = state_anchor.get("AR") if state_anchor else None
        as_state = state_anchor.get("AS") if state_anchor else None
        or_state = state_today.get("OR") if state_today else None
        if (not or_state or getattr(or_state, "final_model", None) is None) and state_anchor:
            or_state = state_anchor.get("OR")
        rr_state = state_today.get("RR") if state_today else None
        if (not rr_state or getattr(rr_state, "final_model", None) is None) and state_anchor:
            rr_state = state_anchor.get("RR")

        def _clean(val: Any) -> Optional[float]:
            try:
                f = float(val)
                if np.isfinite(f):
                    return f
            except Exception:
                pass
            return None

        def _clean_model(val: Any) -> Optional[str]:
            if val is None:
                return None
            txt = str(val).strip()
            if not txt:
                return None
            txt_low = txt.lower()
            if txt_low.startswith("undefined") or txt_low in {"n/a", "na"}:
                return None
            return txt.upper() if len(txt) <= 4 else txt

        idr_hi_close = _clean(getattr(ar_state, "idr_hi_close", None))
        idr_lo_close = _clean(getattr(ar_state, "idr_lo_close", None))
        asia_mid = None
        asia_range = None
        range_pct_change = None

        if idr_hi_close is not None and idr_lo_close is not None and idr_hi_close > idr_lo_close:
            asia_range = idr_hi_close - idr_lo_close
            asia_mid = 0.5 * (idr_hi_close + idr_lo_close)
            if idr_lo_close != 0:
                try:
                    range_pct_change = (asia_range / idr_lo_close) * 100.0
                except Exception:
                    range_pct_change = None
        if range_pct_change is not None:
            range_pct_change = float(round(range_pct_change, 4))

        high_ar_sd = _clean(out.get("high_ar_sd"))
        low_ar_sd = _clean(out.get("low_ar_sd"))

        # Update live session tracker with current bar extremes
        tracker = self._session_tracker(anchor_day, bar.symbol)
        current_session = out.get("current_session")
        if current_session in tracker:
            rec = tracker[current_session]
            rec_high = rec.get("high")
            rec_low = rec.get("low")
            if rec_high is None or bar.h > rec_high:
                rec["high"] = bar.h
            if rec_low is None or bar.l < rec_low:
                rec["low"] = bar.l
            rec["last_ts"] = bar.ts

        # Determine Asia confirmation side using live consolidated output
        confirm_dir = None
        if out.get("current_session") == "AS" and out.get("current_session_break_out"):
            confirm_dir = out.get("current_session_break_out_direction")
        elif out.get("prior_session") == "AS" and out.get("prior_session_break_out"):
            confirm_dir = out.get("prior_session_break_out_direction")
        elif as_state and getattr(as_state, "confirmed", False):
            side = getattr(as_state, "side", None)
            if side == 1:
                confirm_dir = "long"
            elif side == -1:
                confirm_dir = "short"

        asia_confirm_side = "none"
        if isinstance(confirm_dir, str):
            if confirm_dir.lower() == "long":
                asia_confirm_side = "up"
            elif confirm_dir.lower() == "short":
                asia_confirm_side = "down"

        adr_model = _clean_model(getattr(ar_state, "final_model", None)) if ar_state else None
        odr_model = _clean_model(getattr(or_state, "final_model", None)) if or_state else None
        rdr_model = _clean_model(getattr(rr_state, "final_model", None)) if rr_state else None

        bucket = self._range_size_bucket(bar.symbol, anchor_day, range_pct_change) if range_pct_change is not None else "unknown"
        bucket_filter = bucket if bucket in {"0-20%", "20-50%", "50-80%", "80%+"} else None
        confirm_filter = asia_confirm_side if asia_confirm_side in ("up", "down") else None
        odr_filter = odr_model if odr_model else None
        rdr_filter = rdr_model if rdr_model else None

        # Historical day-level percentiles
        day_hist = self._box_history_filtered(
            self._get_box_day_history(bar.symbol),
            anchor_day,
            [
                ("range_size_category", bucket_filter),
                ("asia_confirm_side", confirm_filter),
                ("odr_model", odr_filter),
                ("rdr_model", rdr_filter),
            ],
        )
        day_high_pct = self._percentile_remaining(day_hist.get("day_max_ext_sd") if not day_hist.empty else None, high_ar_sd)
        day_low_pct = self._percentile_remaining(day_hist.get("day_min_ret_sd") if not day_hist.empty else None, low_ar_sd, use_abs=True)

        hist_counts: Dict[str, int] = {"day": int(len(day_hist))}

        # Session-level percentiles
        session_hist_all = self._get_box_session_history(bar.symbol)
        available_session_codes: set[str] = set()
        if not session_hist_all.empty and "session_code" in session_hist_all.columns:
            available_session_codes = set(session_hist_all["session_code"].unique())
        session_high_sd: Dict[str, Optional[float]] = {}
        session_low_sd: Dict[str, Optional[float]] = {}
        session_high_pct: Dict[str, Optional[float]] = {}
        session_low_pct: Dict[str, Optional[float]] = {}

        for sess in TRACKED_SESSIONS:
            if sess == "AS":
                sess_hist = day_hist.copy()
            else:
                if not session_hist_all.empty and "session_code" in session_hist_all.columns:
                    sess_hist = session_hist_all[session_hist_all["session_code"] == sess].copy()
                else:
                    sess_hist = pd.DataFrame()
                sess_hist = self._box_history_filtered(
                    sess_hist,
                    anchor_day,
                    [
                        ("range_size_category", bucket_filter),
                        ("asia_confirm_side", confirm_filter),
                        ("odr_model", odr_filter),
                        ("rdr_model", rdr_filter),
                    ],
                    min_rows=15,
                )
            hist_counts[f"session_{sess}"] = int(len(sess_hist))

            track = tracker.get(sess, {})
            sess_high_val = track.get("high")
            sess_low_val = track.get("low")

            ext_sd_val = None
            ret_sd_val = None
            if asia_range and asia_range > 0 and asia_mid is not None:
                if sess_high_val is not None:
                    ext_sd_val = (float(sess_high_val) - asia_mid) / asia_range
                if sess_low_val is not None:
                    ret_sd_val = (float(sess_low_val) - asia_mid) / asia_range

            session_high_sd[sess] = ext_sd_val
            session_low_sd[sess] = ret_sd_val
            pct_high = self._percentile_remaining(sess_hist.get("ext_sd") if not sess_hist.empty else None, ext_sd_val)
            pct_low = self._percentile_remaining(sess_hist.get("ret_sd") if not sess_hist.empty else None, ret_sd_val, use_abs=True)
            if sess == "AS":
                if pct_high is None and day_high_pct is not None:
                    pct_high = day_high_pct
                if pct_low is None and day_low_pct is not None:
                    pct_low = day_low_pct
            session_high_pct[sess] = pct_high
            session_low_pct[sess] = pct_low

        sd_level_map: Dict[str, float] = {}
        sd_levels: List[float] = []
        if asia_mid is not None and asia_range and asia_range > 0:
            for mult in (1.5, 3.0, 5.5, 8.5):
                up = asia_mid + mult * asia_range
                dn = asia_mid - mult * asia_range
                sd_level_map[f"+{mult}"] = up
                sd_level_map[f"-{mult}"] = dn
                sd_levels.extend([up, dn])

        fallback_mid = asia_mid
        if fallback_mid is None:
            tmp_mid = _clean(bar.c)
            fallback_mid = tmp_mid if tmp_mid is not None else 0.0

        return BoxDailyOut(
            asia_mid=fallback_mid,
            sd_levels=sd_levels,
            volatility_bucket=bucket,
            range_pct_change=range_pct_change,
            range_size_category=bucket,
            asia_confirm_side=asia_confirm_side,
            asia_range=asia_range,
            day_high_sd=high_ar_sd,
            day_low_sd=low_ar_sd,
            day_high_percentile=day_high_pct,
            day_low_percentile=day_low_pct,
            session_high_sd=session_high_sd,
            session_low_sd=session_low_sd,
            session_high_percentiles=session_high_pct,
            session_low_percentiles=session_low_pct,
            hist_sample_sizes=hist_counts,
            sd_level_map=sd_level_map,
            adr_model=adr_model,
            odr_model=odr_model,
            rdr_model=rdr_model,
            filters_applied={
                "range_size_category": bucket_filter or "",
                "asia_confirm_side": confirm_filter or "",
                "odr_model": odr_filter or "",
                "rdr_model": rdr_filter or "",
            },
        )

    # ──────────────────────────────────────────────────────────────
    # ModelDB live (compact v3 mapping)
    # ──────────────────────────────────────────────────────────────
    def get_modeldb_live_v3(self, bar: Bar) -> Dict[str, float]:
        """Produce a compact live probability dict using consolidated fields.

        Keys mirror v2 for drop-in compatibility where possible.
        """
        out = self._ensure_latest_out(bar)

        # Mid-break probabilities → 1.0 if broken, else neutral 0.5
        p_adr_mid_broken = 1.0 if out.get("adr_mid_break") else 0.5
        p_odr_mid_broken = 1.0 if out.get("odr_mid_break") else 0.5

        # Optional first mid-break split (can be omitted if not needed)
        try:
            p_first_asia, p_first_overnight = self._estimate_first_mid_break_from_modeldb(bar, out)
        except Exception:
            p_first_asia, p_first_overnight = 0.5, 0.5

        # Session true probability from ModelDB (with context/backoff). Do not clamp to live truth.
        sess_true = self._estimate_session_true_from_modeldb(bar, out)

        # Box direction one-hot when determined for current session
        box_dir = out.get("current_session_box_dir")
        if box_dir == "up":
            p_box_up, p_box_down = 1.0, 0.0
        elif box_dir == "down":
            p_box_up, p_box_down = 0.0, 1.0
        else:
            p_box_up = p_box_down = 0.5 if box_dir is None else 0.0

        # Model probabilities: one-hot if current session model is a known label
        model = out.get("current_session_model") or "Undefined"
        model_probs = {m: 0.0 for m in ALL_MODELS}
        if model in ALL_MODELS:
            model_probs[model] = 1.0
        else:
            # neutral uniform over models
            u = 1.0 / len(ALL_MODELS)
            for m in ALL_MODELS:
                model_probs[m] = u

        # Remaining-model probabilities (for in-progress range):
        # If current range session is not complete, estimate model distribution from model_db using prior-session context.
        def _estimate_remaining_models() -> Dict[str, float]:
            # Determine target range for models
            if bar.session in ("AR","OR","RR"):
                target = bar.session
            else:
                target = {"AS":"AR","OS":"OR","RS":"RR"}.get(bar.session)
            col = {"AR":"ADR_Model","OR":"ODR_Model","RR":"RDR_Model"}.get(target)
            hist = self._get_modeldb_for_symbol(out.get("symbol") or "")
            if not hist.empty and col in hist.columns:
                # Start with symbol slice
                df = hist[hist.get("symbol") == out.get("symbol")].copy()

                # Day filter
                day_num = out.get("day_number")
                if day_num is not None and "day_num" in df.columns:
                    df = df[df["day_num"] == int(day_num)]

                # Prior prefix selection
                prior_prefix = "RDR" if target == "AR" else ("ADR" if target == "OR" else "ODR")
                # Pull prior context from consolidated
                prior_model = out.get("prior_session_model")
                prior_box   = out.get("prior_session_box_dir")
                prior_true  = out.get("prior_session_true")
                prior_dir   = out.get("prior_session_break_out_direction")

                # Column map
                col_model = f"{prior_prefix}_Model"; col_box = f"{prior_prefix}_Box_Direction"
                col_true = f"{prior_prefix}_Session_True"; col_trade = f"{prior_prefix}_Trade_Direction"

                filters: list[tuple[str, object]] = []
                # Least → most important: box_dir, model, session_true, trade_dir
                if col_box in df.columns and isinstance(prior_box, str):
                    filters.append((col_box, prior_box))
                if col_model in df.columns and isinstance(prior_model, str) and prior_model in ALL_MODELS:
                    filters.append((col_model, prior_model))
                if col_true in df.columns and isinstance(prior_true, (bool, type(None))):
                    filters.append((col_true, prior_true))
                if col_trade in df.columns and isinstance(prior_dir, str):
                    filters.append((col_trade, "Long" if prior_dir == "long" else "Short"))

                def _apply(fs):
                    tmp = df.copy()
                    for k, v in fs:
                        if k in tmp.columns and v is not None:
                            tmp = tmp[tmp[k] == v]
                    return tmp

                sub = None
                if filters:
                    for drop in range(0, len(filters)+1):
                        fs = filters[drop:]
                        cand = _apply(fs)
                        if len(cand) >= 20:
                            sub = cand; break
                    if sub is None:
                        sub = _apply([])
                else:
                    sub = df

                series = sub[col].dropna().astype(str)
                # Restrict to models that are still possible from the live engine's allowed set when provided;
                # otherwise fall back to ALLOWED_NEXT(current model).
                allowed_from_engine = out.get("current_session_allowed_models")
                if isinstance(allowed_from_engine, (list, tuple, set)) and len(allowed_from_engine) > 0:
                    allowed = set(str(m) for m in allowed_from_engine)
                else:
                    current_live_model = out.get("current_session_model")
                    allowed = set(ALLOWED_NEXT.get(current_live_model, ALL_MODELS)) if isinstance(current_live_model, str) else set(ALL_MODELS)
                if not series.empty:
                    series = series[series.isin(allowed)]
                if not series.empty:
                    # Laplace smoothing across allowed set only
                    counts = {m: 1 for m in allowed}
                    for m in series:
                        if m in counts:
                            counts[m] += 1
                    total = float(sum(counts.values()))
                    return {m: counts[m]/total for m in allowed}
                else:
                    # No historical rows after restriction → uniform over allowed
                    if allowed:
                        u_allowed = 1.0 / float(len(allowed))
                        return {m: u_allowed for m in allowed}
            # Fallback uniform — restrict to engine-provided allowed set when present; else ALLOWED_NEXT
            allowed_from_engine = out.get("current_session_allowed_models")
            if isinstance(allowed_from_engine, (list, tuple, set)) and len(allowed_from_engine) > 0:
                allowed = set(str(m) for m in allowed_from_engine)
            else:
                current_live_model = out.get("current_session_model")
                allowed = set(ALLOWED_NEXT.get(current_live_model, ALL_MODELS)) if isinstance(current_live_model, str) else set(ALL_MODELS)
            if allowed:
                u = 1.0/float(len(allowed))
                return {m: u for m in allowed}
            u = 1.0/len(ALL_MODELS)
            return {m: u for m in ALL_MODELS}

        # If in a range session and not complete → use remaining distribution; else keep one-hot
        in_range = bar.session in ("AR","OR","RR")
        # For range sessions, consolidated sets current_session_break_out == True when range completes
        is_complete = bool(out.get("current_session_break_out", False)) if in_range else False
        # Better completeness check: consolidated provides current_session_model and prior_session*; but we can check attrs
        # Use box_std presence plus follow-session to infer; fallback to model_probs as set above
        if in_range and not is_complete:
            # We don't have an explicit 'complete' flag in consolidated output for range; rely on session_range_attrs via engine state when available.
            # If not determinable, assume not complete and use remaining distribution.
            # Replace current one-hot with estimated distribution.
            model_probs = _estimate_remaining_models()

        # Confirmation direction probabilities from ModelDB with the same filtering/backoff.
        p_confirm_long_forward, p_confirm_short_forward = self._estimate_confirm_split_from_modeldb(bar, out)
        # Clamp to 1/0 once the follow session confirms in a direction.
        if bar.session in ("AS", "OS", "RS") and bool(out.get("current_session_break_out")):
            dir_now = out.get("current_session_break_out_direction") or out.get("prior_session_break_out_direction")
            if dir_now == "long":
                p_confirm_long_forward, p_confirm_short_forward = 1.0, 0.0
            elif dir_now == "short":
                p_confirm_long_forward, p_confirm_short_forward = 0.0, 1.0

        return {
            "p_adr_mid_broken": round(p_adr_mid_broken, 3),
            "p_odr_mid_broken": round(p_odr_mid_broken, 3),
            "p_session_true": round(sess_true, 3),
            "p_confirm_long_forward": round(p_confirm_long_forward, 3),
            "p_confirm_short_forward": round(p_confirm_short_forward, 3),
            # First mid-break split can be included or omitted; keeping for now
            "p_first_mid_break_asia": round(p_first_asia, 3),
            "p_first_mid_break_overnight": round(p_first_overnight, 3),
            "p_box_up": round(p_box_up, 3),
            "p_box_down": round(p_box_down, 3),
            "p_model_dx": round(model_probs.get("DX", 0.0), 3),
            "p_model_rc": round(model_probs.get("RC", 0.0), 3),
            "p_model_uxp": round(model_probs.get("UXP", 0.0), 3),
            "p_model_rx": round(model_probs.get("RX", 0.0), 3),
            "p_model_ux": round(model_probs.get("UX", 0.0), 3),
            "p_model_u": round(model_probs.get("U", 0.0), 3),
            "p_model_d": round(model_probs.get("D", 0.0), 3),
            "p_model_dxp": round(model_probs.get("DXP", 0.0), 3),
        }

    # ──────────────────────────────────────────────────────────────
    # Seq intraday (minimal v3 mapping)
    # ──────────────────────────────────────────────────────────────
    def get_seq_intraday_context_v3(self, bar: Bar) -> SeqIntradayOut:
        """Map consolidated intraday tracking to SeqIntradayOut (fast filtered version)."""
        out = self._ensure_latest_out(bar)

        # Outside 04:00→15:55 → neutral output
        if not self._within_intraday_window(bar.ts):
            return SeqIntradayOut()

        seq_hist = self._get_seq_intraday_for_symbol(bar.symbol)
        if seq_hist.empty:
            return SeqIntradayOut()

        if "symbol" in seq_hist.columns:
            seq_hist_symbol = seq_hist[seq_hist["symbol"] == bar.symbol].copy()
        else:
            seq_hist_symbol = seq_hist.copy()

        ny_day = self.engine._ny_date_from_epoch(bar.ts)

        # Current state values from consolidated output
        t_bucket_5m = out.get("current_time_idx")
        t_bucket_30m = self._since_0400_bucket(bar.ts)
        hi_now = out.get("high_pct")
        lo_now = out.get("low_pct")
        # hi/lo time indices from intraday state (5-minute buckets)
        state = self.engine._get_state(ny_day, bar.symbol)
        intraday_state = state.get("intraday")
        hi_idx = getattr(intraday_state, "hi_time_idx", None) if intraday_state else None
        lo_idx = getattr(intraday_state, "lo_time_idx", None) if intraday_state else None

        filters = self._seq_filters_for_now(bar, ny_day, t_bucket_5m)
        mapped_filters, relax_cols = self._map_seq_filter_keys(filters, seq_hist)
        sub = self._apply_filters_with_backoff(seq_hist, mapped_filters, relax_cols, self.MIN_SEQ_ROWS)
        if len(sub) < self.MIN_SEQ_ROWS and "OR_Box_Direction" in mapped_filters:
            # Relax ODR box direction explicitly when we don't have enough history
            mf2 = dict(mapped_filters)
            mf2.pop("OR_Box_Direction", None)
            relax2 = [c for c in relax_cols if c != "OR_Box_Direction"]
            sub = self._apply_filters_with_backoff(seq_hist, mf2, relax2, self.MIN_SEQ_ROWS)

        allowed_high = self._allowed_sessions(ny_day, t_bucket_5m, hi_idx)
        allowed_low = self._allowed_sessions(ny_day, t_bucket_5m, lo_idx)

        if self.include_rdr_filter:
            quant_filters = dict(mapped_filters)
            quant_relax = list(relax_cols)
        else:
            quant_filters = {k: v for k, v in mapped_filters.items() if not k.startswith("RDR_") and k != "NextRDR_Model"}
            quant_relax = [c for c in relax_cols if not c.startswith("RDR_") and c != "NextRDR_Model"]
        base_quant = self._apply_filters_with_backoff(seq_hist, quant_filters, quant_relax, self.MIN_SEQ_ROWS)
        if not base_quant.empty and "symbol" in base_quant.columns:
            base_quant = base_quant[base_quant["symbol"] == bar.symbol].copy()
        if base_quant.empty:
            base_quant = seq_hist_symbol.copy() if not seq_hist_symbol.empty else seq_hist.copy()

        def _apply_session_gate(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            gated = df
            hi_col, lo_col = self._seq_where_cols(df)
            if hi_col and allowed_high:
                gated = gated[gated[hi_col].astype(str).str.upper().isin(allowed_high)]
            if lo_col and allowed_low:
                gated = gated[gated[lo_col].astype(str).str.upper().isin(allowed_low)]
            return gated

        sub_gated = _apply_session_gate(base_quant)
        if sub_gated.empty:
            sub_gated = _apply_session_gate(seq_hist_symbol)
        if sub_gated.empty:
            sub_gated = seq_hist_symbol.copy()

        pH = self._seq_location_probs(sub_gated, "high", allowed_high)
        pL = self._seq_location_probs(sub_gated, "low", allowed_low)
        pH = self._zero_impossible_sessions(pH, hi_idx, ny_day, "high", t_bucket_5m)
        pL = self._zero_impossible_sessions(pL, lo_idx, ny_day, "low", t_bucket_5m)

        quant_source = sub_gated if not sub_gated.empty else base_quant
        if quant_source.empty:
            quant_source = seq_hist_symbol
        pct_hi_col, _ = self._seq_cols(quant_source, "high")
        pct_lo_col, _ = self._seq_cols(quant_source, "low")
        hi_series = pd.to_numeric(quant_source[pct_hi_col], errors="coerce") if pct_hi_col else None
        lo_series_raw = pd.to_numeric(quant_source[pct_lo_col], errors="coerce") if pct_lo_col else None
        q_hi = self._quantiles(hi_series, [0.20, 0.50, 0.80, 0.85]) if hi_series is not None else (None, None, None, None)
        q_lo = self._quantiles(lo_series_raw, [0.20, 0.50, 0.80, 0.85]) if lo_series_raw is not None else (None, None, None, None)

        trend_guard_hi = bool(hi_now is not None and q_hi[0] is not None and hi_now >= q_hi[0])
        trend_guard_lo = bool(lo_now is not None and q_lo[2] is not None and lo_now <= q_lo[2])
        failed_20_hi = bool(hi_now is not None and q_hi[0] is not None and hi_now < q_hi[0])
        failed_20_lo = bool(lo_now is not None and q_lo[0] is not None and lo_now > q_lo[0])

        def _percentile(series, value, side: str):
            if series is None or value is None:
                return None
            s = pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                return None
            if side == "high":
                total = float(len(s))
                rank = float((s >= value).sum())
                return rank / total if total > 0 else None
            total = float(len(s))
            rank = float((s <= value).sum())
            return rank / total if total > 0 else None

        hi_pct_percentile = _percentile(hi_series, hi_now, "high")
        lo_pct_percentile = _percentile(lo_series_raw, lo_now, "low")

        p_high_hold_b = self._hold_prob_bucketed(sub_gated, t_bucket_30m, hi_now, "high") if hi_now is not None else 0.5
        p_low_hold_b = self._hold_prob_bucketed(sub_gated, t_bucket_30m, lo_now, "low") if lo_now is not None else 0.5
        p_high_hold_r = self._hold_prob_resid(sub_gated, t_bucket_30m, hi_now, "high") if hi_now is not None else p_high_hold_b
        p_low_hold_r = self._hold_prob_resid(sub_gated, t_bucket_30m, lo_now, "low") if lo_now is not None else p_low_hold_b

        def _clean_prob(val: Optional[float], fallback: float) -> float:
            if isinstance(val, (int, float)) and np.isfinite(val):
                return float(val)
            return float(fallback)

        p_high_hold_b = _clean_prob(p_high_hold_b, 0.5)
        p_low_hold_b = _clean_prob(p_low_hold_b, 0.5)
        p_high_hold_r = _clean_prob(p_high_hold_r, p_high_hold_b)
        p_low_hold_r = _clean_prob(p_low_hold_r, p_low_hold_b)

        p_high_hold_b = 0.5 * (p_high_hold_b + p_high_hold_r)
        p_low_hold_b = 0.5 * (p_low_hold_b + p_low_hold_r)

        # At the final 30-minute bucket (15:55), the outcome is already known → holds become 1.0
        if t_bucket_30m is not None:
            max_bucket = 23  # 04:00→15:55 in 30m buckets
            if t_bucket_30m >= max_bucket:
                p_high_hold_b = 1.0
                p_low_hold_b = 1.0
                p_high_hold_r = 1.0
                p_low_hold_r = 1.0

        # Session context helpers
        or_state = state.get("OR")
        os_state = state.get("OS")
        rr_state = state.get("RR")

        odr_model = getattr(or_state, "final_model", None) if (or_state and or_state.final_model) else None
        odr_box = None
        if or_state and getattr(or_state, "box_std", None) is not None:
            bs = float(or_state.box_std)
            odr_box = "up" if bs > 0 else "down" if bs < 0 else "flat"
        if odr_box == "flat":  # treat flat as not informative
            odr_box = None

        odr_trade = None
        os_break = None
        os_true = None
        if os_state:
            if getattr(os_state, "confirmed", False) and os_state.side in (1, -1):
                odr_trade = "long" if os_state.side == 1 else "short"
                os_break = odr_trade
                os_true = not bool(os_state.false_detected)
            elif getattr(os_state, "false_detected", None) is True:
                os_true = False

        rdr_model = getattr(rr_state, "final_model", None) if (rr_state and rr_state.final_model) else None
        rdr_box = None
        if rr_state and getattr(rr_state, "box_std", None) is not None:
            bs = float(rr_state.box_std)
            rdr_box = "up" if bs > 0 else "down" if bs < 0 else "flat"
        if rdr_box == "flat":
            rdr_box = None

        return SeqIntradayOut(
            p_cycle_pair=0.5,
            p_high_holds=p_high_hold_r,
            p_low_holds=p_low_hold_r,
            p_day_high_in_OS=round(pH[0], 3),
            p_day_high_in_TR=round(pH[1], 3),
            p_day_high_in_RR=round(pH[2], 3),
            p_day_high_in_RS=round(pH[3], 3),
            p_day_low_in_OS=round(pL[0], 3),
            p_day_low_in_TR=round(pL[1], 3),
            p_day_low_in_RR=round(pL[2], 3),
            p_day_low_in_RS=round(pL[3], 3),
            p_high_holds_bucketed=round(p_high_hold_b, 3),
            p_low_holds_bucketed=round(p_low_hold_b, 3),
            p_high_holds_reg=round(p_high_hold_r, 3),
            p_low_holds_reg=round(p_low_hold_r, 3),
            trend_guard_hi=trend_guard_hi,
            trend_guard_lo=trend_guard_lo,
            failed_20_hi=failed_20_hi,
            failed_20_lo=failed_20_lo,
            t_bucket_5m=int(t_bucket_5m) if t_bucket_5m is not None else None,
            hi_pct_now=hi_now,
            lo_pct_now=lo_now,
            hi_pct_percentile=hi_pct_percentile,
            lo_pct_percentile=lo_pct_percentile,
            os_breakout_dir=os_break,
            os_true=os_true,
            odr_model=odr_model,
            odr_box=odr_box,
            odr_trade=odr_trade,
            rdr_model=rdr_model,
            rdr_box=rdr_box,
        )

    # ──────────────────────────────────────────────────────────────
    # WDDRB adapter
    # ──────────────────────────────────────────────────────────────
    def get_wddrb_context(self, bar: Bar) -> WddrbOut:
        out = self._ensure_latest_out(bar)
        history = self._get_wddrb_for_symbol(bar.symbol)
        if history.empty:
            return WddrbOut(notes=["no WDDRB history for symbol"])

        symbol = out.get("symbol") or bar.symbol
        bar_dt = datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ))
        ny_day_val = out.get("ny_day")
        if isinstance(ny_day_val, pd.Timestamp):
            ny_day_date = ny_day_val.date()
        elif isinstance(ny_day_val, date):
            ny_day_date = ny_day_val
        else:
            ny_day_date = self.engine._ny_date_from_epoch(bar.ts)
        active_day = self._wddrb_active_day(bar.ts, ny_day_date)
        cache_day = active_day or ny_day_date
        modeldb_row = self._modeldb_row_for_day(symbol, cache_day)
        current_row = self._wddrb_row_for_day(symbol, cache_day)
        prev_day = prev_business_day(cache_day) if cache_day else None
        prev_row = self._prior_wddrb_row(symbol, prev_day) if prev_day else None
        cycle_row = current_row if current_row is not None else prev_row
        prev_rdr_lo = self._safe_float(cycle_row.get("rdr_lo")) if cycle_row is not None else None
        prev_rdr_hi = self._safe_float(cycle_row.get("rdr_hi")) if cycle_row is not None else None
        last_day = self._wddrb_last_day.get(symbol)
        if last_day != cache_day:
            self._wddrb_hold_cache.pop(symbol, None)
            self._wddrb_last_day[symbol] = cache_day

        hist_rows = int(len(history))
        idx_now = self._safe_float(out.get("wddrb_current_post_index_30m"))
        session_label = (out.get("session") or bar.session or "").upper()
        fallback_adr = cycle_row.get("asia_range_model_hilo") if cycle_row is not None else None
        requested_adr = self._normalize_wddrb_value(out.get("asia_range_model_hilo") or fallback_adr)
        adr_model: Optional[str] = None
        last_adr_entry = self._wddrb_last_adr.get(symbol)
        if isinstance(last_adr_entry, dict):
            last_adr_value = last_adr_entry.get("value")
            last_adr_day = last_adr_entry.get("day")
        else:
            last_adr_value = last_adr_entry
            last_adr_day = None
        if self._adr_filter_window_active(bar_dt):
            candidate_adr = requested_adr
            allow_refresh = (
                last_adr_value is None
                or last_adr_day != cache_day
                or bar_dt.time() >= time(9, 30)
            )
            if candidate_adr and allow_refresh:
                adr_model = candidate_adr
                self._wddrb_last_adr[symbol] = {"value": adr_model, "day": cache_day}
            else:
                adr_model = last_adr_value or candidate_adr
        else:
            self._wddrb_last_adr.pop(symbol, None)
            adr_model = None
        breakout_side = self._normalize_wddrb_value(out.get("wddrb_break_side"))
        seed_breakout = breakout_side
        rdr_trade, rdr_box, rdr_true = self._resolve_wddrb_rdr_targets(out, modeldb_row)
        if cycle_row is not None and (
            bar_dt.time() < time(9, 30) or bar_dt.time() >= time(16, 0)
        ):
            rdr_trade = self._normalize_wddrb_value(cycle_row.get("model_rdr_trade_direction")) or rdr_trade
            rdr_box = self._normalize_wddrb_value(cycle_row.get("model_rdr_box_direction")) or rdr_box
            if cycle_row.get("model_rdr_session_true") is not None:
                rdr_true = bool(cycle_row.get("model_rdr_session_true"))

        hold_hist = history
        hold_stage = "pre_window"
        hold_filters_applied: dict[str, Any] = {}
        hold_filters: dict[str, Any] = {}
        hold_relax: list[str] = []
        hold_applied: dict[str, Any] = {}
        cache_entry = None
        filter_signature = (adr_model, seed_breakout, rdr_trade, rdr_box, bool(rdr_true) if rdr_true is not None else None)
        if idx_now is not None:
            cache_entry = self._wddrb_hold_cache.get(symbol)
            need_refresh = cache_entry is None
            if not need_refresh:
                last_idx = cache_entry.get("idx")
                if last_idx is None or idx_now < last_idx:
                    need_refresh = True
                elif cache_entry.get("signature") != filter_signature:
                    need_refresh = True
            if need_refresh:
                hold_filters, hold_relax, hold_applied, hold_stage = self._build_wddrb_filters(
                    "hold", adr_model, seed_breakout, rdr_trade, rdr_box, rdr_true
                )
                cache_entry = {
                    "filters": dict(hold_filters),
                    "relax": list(hold_relax),
                    "applied": dict(hold_applied),
                    "stage": hold_stage,
                    "idx": idx_now,
                    "signature": filter_signature,
                }
                self._wddrb_hold_cache[symbol] = cache_entry
            else:
                hold_stage = cache_entry.get("stage", "hold_filter")
                hold_filters = cache_entry.get("filters", {})
                hold_relax = cache_entry.get("relax", [])
                hold_applied = cache_entry.get("applied", {})
            hold_hist = (
                self._apply_filters_with_backoff(history, hold_filters, hold_relax, self.MIN_SEQ_ROWS)
                if cache_entry and cache_entry.get("filters")
                else history
            )
            if hold_hist.empty:
                hold_hist = history
            hold_filters_applied = dict(cache_entry.get("applied", {}))
        else:
            self._wddrb_hold_cache.pop(symbol, None)
            apply_hold_filters = bar_dt.time() >= time(16, 0) or bar_dt.time() <= time(9, 25)
            if apply_hold_filters:
                hold_filters, hold_relax, hold_applied, hold_stage = self._build_wddrb_filters(
                    "hold", adr_model, seed_breakout, rdr_trade, rdr_box, rdr_true
                )
                if hold_filters:
                    hold_hist = self._apply_filters_with_backoff(history, hold_filters, hold_relax, self.MIN_SEQ_ROWS)
                    if hold_hist.empty:
                        hold_hist = history
                hold_filters_applied = dict(hold_applied)
            else:
                hold_stage = "pre_window"
                hold_filters_applied = {}

        applied_filters = dict(hold_filters_applied)
        stage_out = hold_stage
        if breakout_side is None and stage_out.startswith("breakout_"):
            stage_out = stage_out.replace("breakout_", "", 1)
        applied_filters["stage"] = stage_out
        hist_filtered_rows = int(len(hold_hist))

        full_high_sd_now = self._safe_float(out.get("wddrb_full_high_sd"))
        full_low_sd_now = self._safe_float(out.get("wddrb_full_low_sd"))

        full_high_hist = hold_hist.dropna(subset=["full_high_index_30m", "high_sd_full"])
        full_low_hist = hold_hist.dropna(subset=["full_low_index_30m", "low_sd_full"])

        p_full_high = self._wddrb_hold_probability(
            full_high_hist, "full_high_index_30m", "high_sd_full", idx_now, full_high_sd_now, "high"
        )
        p_full_low = self._wddrb_hold_probability(
            full_low_hist, "full_low_index_30m", "low_sd_full", idx_now, full_low_sd_now, "low"
        )

        full_high_percentile = self._percentile_reached(
            pd.to_numeric(full_high_hist["high_sd_full"], errors="coerce"), full_high_sd_now
        )
        full_low_percentile = self._percentile_reached(
            pd.to_numeric(full_low_hist["low_sd_full"], errors="coerce"), full_low_sd_now, use_abs=True, head=True
        )

        breakout_side = self._normalize_wddrb_value(out.get("wddrb_break_side"))
        post_high_sd_now = self._safe_float(out.get("wddrb_post_high_sd"))
        post_low_sd_now = self._safe_float(out.get("wddrb_post_low_sd"))

        post_hist = hold_hist.dropna(subset=["post_high_index_30m", "post_low_index_30m"])
        if breakout_side:
            post_hist = post_hist.dropna(subset=["post_high_index_30m", "post_low_index_30m"])

        if breakout_side and not post_hist.empty:
            post_hist = post_hist.dropna(subset=["post_total_index_30m"])

        sample_size_post = len(post_hist)
        p_post_high = None
        p_post_low = None
        post_high_percentile = None
        post_low_percentile = None

        if breakout_side and post_hist is not None and not post_hist.empty:
            p_post_high = self._wddrb_hold_probability(
                post_hist, "post_high_index_30m", "high_sd_post_break", idx_now, post_high_sd_now, "high"
            )
            p_post_low = self._wddrb_hold_probability(
                post_hist, "post_low_index_30m", "low_sd_post_break", idx_now, post_low_sd_now, "low"
            )
            post_high_percentile = self._percentile_reached(
                pd.to_numeric(post_hist["high_sd_post_break"], errors="coerce"), post_high_sd_now
            )
            post_low_percentile = self._percentile_reached(
                pd.to_numeric(post_hist["low_sd_post_break"], errors="coerce"), post_low_sd_now, use_abs=True, head=True
            )

        sample_size_model = len(hold_hist)
        # Breakout probabilities (before breakout use history, after breakout clamp)
        p_break_long = 0.5
        p_break_short = 0.5
        if breakout_side:
            if breakout_side == "UP":
                p_break_long, p_break_short = 1.0, 0.0
            elif breakout_side == "DOWN":
                p_break_long, p_break_short = 0.0, 1.0
        else:
            hist_break = hold_hist.dropna(subset=["first_break_side"])
            if not hist_break.empty:
                fb = hist_break["first_break_side"].astype(str).str.upper()
                p_break_long = self._laplace_mean((fb == "UP").astype(int))
                p_break_short = self._laplace_mean((fb == "DOWN").astype(int))

        notes: list[str] = []
        if hist_filtered_rows < self.MIN_SEQ_ROWS:
            notes.append("limited_sample")
        if breakout_side and post_hist.empty:
            notes.append("no_post_history_for_breakout")

        return WddrbOut(
            stage=stage_out,
            filters_applied=applied_filters,
            hist_rows=hist_rows,
            hist_filtered_rows=hist_filtered_rows,
            sample_size_full=len(hold_hist),
            sample_size_post=sample_size_post,
            sample_size_model=sample_size_model,
            p_full_high_holds=round(p_full_high, 3),
            p_full_low_holds=round(p_full_low, 3),
            p_post_high_holds=round(p_post_high, 3) if p_post_high is not None else None,
            p_post_low_holds=round(p_post_low, 3) if p_post_low is not None else None,
            full_high_percentile=None if full_high_percentile is None else round(full_high_percentile, 3),
            full_low_percentile=None if full_low_percentile is None else round(full_low_percentile, 3),
            post_high_percentile=None if post_high_percentile is None else round(post_high_percentile, 3),
            post_low_percentile=None if post_low_percentile is None else round(post_low_percentile, 3),
            p_break_long=round(p_break_long, 3),
            p_break_short=round(p_break_short, 3),
            notes=notes,
        )

    def get_wddrb_model_context(self, bar: Bar) -> WddrbModelOut:
        out = self._ensure_latest_out(bar)
        history = self._get_wddrb_for_symbol(bar.symbol)
        if history.empty:
            return WddrbModelOut(notes=["no WDDRB history for symbol"])

        symbol = out.get("symbol") or bar.symbol
        ny_day_val = out.get("ny_day")
        if isinstance(ny_day_val, pd.Timestamp):
            ny_day_date = ny_day_val.date()
        elif isinstance(ny_day_val, date):
            ny_day_date = ny_day_val
        else:
            ny_day_date = self.engine._ny_date_from_epoch(bar.ts)
        if ny_day_date is None:
            return WddrbModelOut(notes=["ny_day unavailable"])

        bar_dt = datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ))
        phase = self._wddrb_model_phase(bar_dt)
        if phase == "inactive":
            return WddrbModelOut(stage="inactive")

        reference_day = self._wddrb_model_reference_day(ny_day_date)
        ref_row = self._wddrb_row_for_day(symbol, reference_day)
        if ref_row is None:
            return WddrbModelOut(stage="waiting_reference", notes=["no reference history"])

        filters: list[tuple[str, Any]] = []
        notes: list[str] = []
        state = self._update_wddrb_model_state(symbol, ny_day_date, out)

        prev_break = self._normalize_wddrb_value(ref_row.get("first_break_side"))
        if prev_break:
            filters.append(("first_break_side", prev_break))

        def _bucket_from_price(price: Any) -> Optional[str]:
            val = self._safe_float(price)
            if val is None or not np.isfinite(val):
                return None
            return self._wddrb_open_bucket_static(
                val,
                ref_row.get("rdr_lo"),
                ref_row.get("rdr_hi"),
                ref_row.get("rdr_q"),
            )

        stage_label = "preopen_model"
        if phase == "preopen":
            prev_adr = self._normalize_wddrb_value(ref_row.get("asia_range_model_hilo"))
            if prev_adr:
                filters.append(("asia_range_model_hilo", prev_adr))
            bucket = _bucket_from_price(out.get("wddrb_preopen_price"))
            if bucket is None:
                return WddrbModelOut(stage="preopen_pending", notes=["preopen_price_unavailable"])
            filters.append(("open_bucket_quarters", bucket))
        else:
            stage_label = "intraday_model"
            open_finalized = bool(out.get("wddrb_open_finalized"))
            bucket = _bucket_from_price(out.get("wddrb_open_price") if open_finalized else out.get("wddrb_preopen_price"))
            if bucket is None:
                return WddrbModelOut(stage="intraday_wait_open", notes=["open_price_unavailable"])
            filters.append(("open_bucket_quarters", bucket))
            live_trade = self._normalize_wddrb_value(out.get("current_session_break_out_direction"))
            trade = live_trade or self._normalize_wddrb_value(state.get("next_model_rdr_trade_direction"))
            if trade:
                filters.append(("next_model_rdr_trade_direction", trade))
            live_box = self._normalize_wddrb_value(out.get("current_session_box_dir"))
            box_dir = live_box or self._normalize_wddrb_value(state.get("next_model_rdr_box_direction"))
            if box_dir:
                filters.append(("next_model_rdr_box_direction", box_dir))
            sess_true = out.get("current_session_true")
            if sess_true is None:
                sess_true = state.get("next_model_rdr_session_true")
            if sess_true is not None:
                filters.append(("next_model_rdr_session_true", bool(sess_true)))
            self._update_intraday_touch_flags(state, out, ref_row)

        subset, applied = self._apply_wddrb_model_filters(history, filters, self.MIN_WDDRB_MODEL_ROWS)
        if len(subset) < self.MIN_WDDRB_MODEL_ROWS:
            notes.append("limited_sample")
        probs = self._wddrb_model_probabilities(subset, ref_row, out, stage_label, state)

        return WddrbModelOut(
            stage=stage_label,
            filters_applied=applied,
            sample_size=int(len(subset)),
            p_model_ug=round(probs.get("UG", 0.0), 3),
            p_model_u=round(probs.get("U", 0.0), 3),
            p_model_ux=round(probs.get("UX", 0.0), 3),
            p_model_o=round(probs.get("O", 0.0), 3),
            p_model_i=round(probs.get("I", 0.0), 3),
            p_model_dx=round(probs.get("DX", 0.0), 3),
            p_model_d=round(probs.get("D", 0.0), 3),
            p_model_dg=round(probs.get("DG", 0.0), 3),
            notes=notes,
        )

    # ──────────────────────────────────────────────────────────────
    # Asia sequencing helpers (20:30 → 08:25: AS → TO → OR → OS)
    # ──────────────────────────────────────────────────────────────
    def _within_asia_window(self, ts_epoch: int) -> bool:
        """Check if bar is within Asia sequencing window (20:30→08:25)."""
        dt = datetime.fromtimestamp(ts_epoch, tz=ZoneInfo(NY_TZ))
        t = dt.time()
        start = time(20, 30)
        end = time(8, 25)

        # Window spans midnight: 20:30 day n → 08:25 day n+1
        return t >= start or t <= end
    
    def _since_asia_start_bucket(self, ts_epoch: int, bucket_minutes: int = 30) -> int:
        """Calculate bucket index since Asia window start (20:30).
        
        Window: 20:30 day n → 08:25 day n+1 = 11h55m = 715 minutes = ~24 buckets (30min)
        """
        dt = datetime.fromtimestamp(ts_epoch, tz=ZoneInfo(NY_TZ))
        bar_date = dt.date()
        bar_time = dt.time()
        if bar_time >= time(20, 30):
            anchor_day = bar_date
        else:
            anchor_day = prev_business_day(bar_date)
        start = datetime.combine(anchor_day, time(20, 30), tzinfo=ZoneInfo(NY_TZ))
        elapsed = max(0, int((dt - start).total_seconds()))
        bucket = elapsed // (bucket_minutes * 60)
        return min(23, max(0, bucket))
    
    def _asia_session_windows(self, anchor_day: date) -> dict:
        """Get session windows for Asia sequencing (20:30 anchor_day → 08:25 next day).
        
        Sessions in order:
        - AS (20:30-01:55 anchor_day/next_day)
        - TO (02:00-02:55 next_day)
        - OR (03:00-03:55 next_day)
        - OS (04:00-08:25 next_day)
        """
        tz = ZoneInfo(NY_TZ)
        next_day = anchor_day + timedelta(days=1)
        
        # AS spans midnight
        as_start = datetime.combine(anchor_day, time(20, 30), tzinfo=tz)
        as_end = datetime.combine(next_day, time(1, 55), tzinfo=tz)
        
        # Day n+1 sessions
        to_start = datetime.combine(next_day, time(2, 0), tzinfo=tz)
        to_end = datetime.combine(next_day, time(2, 55), tzinfo=tz)
        or_start = datetime.combine(next_day, time(3, 0), tzinfo=tz)
        or_end = datetime.combine(next_day, time(3, 55), tzinfo=tz)
        os_start = datetime.combine(next_day, time(4, 0), tzinfo=tz)
        os_end = datetime.combine(next_day, time(8, 25), tzinfo=tz)
        
        return {
            "AS": (as_start, as_end),
            "TO": (to_start, to_end),
            "OR": (or_start, or_end),
            "OS": (os_start, os_end),
        }
    
    def _asia_session_from_5m_idx(self, anchor_day: date, idx: int) -> Optional[str]:
        """Map Asia 5-minute index to session name.
        
        Asia index starts at 0 at 20:30 anchor_day (start of AS session).
        Each 5-minute bar increments the index.
        """
        if idx is None:
            return None
        wins = self._asia_session_windows(anchor_day)
        asia_start = datetime.combine(anchor_day, time(20, 30), tzinfo=ZoneInfo(NY_TZ))
        ts = asia_start + timedelta(minutes=5 * idx)
        
        for name, (start, end) in wins.items():
            if start <= ts <= end:
                return name
        return "OS" if ts > wins["OS"][1] else None
    
    def _asia_allowed_sessions(self, anchor_day: date, current_bucket: Optional[int], 
                               extreme_idx: Optional[int]) -> set[str]:
        """Determine which sessions are still possible for Asia extremes."""
        session_order = ["AS", "TO", "OR", "OS"]
        wins = self._asia_session_windows(anchor_day)
        asia_start = datetime.combine(anchor_day, time(20, 30), tzinfo=ZoneInfo(NY_TZ))
        
        extreme_session = self._asia_session_from_5m_idx(anchor_day, extreme_idx)
        allowed: set[str] = set()
        
        for sess in session_order:
            start_dt, end_dt = wins[sess]
            # Calculate Asia index for session end
            end_asia_idx = int(max(0, (end_dt - asia_start).total_seconds() // 300))
            
            if extreme_session == sess:
                allowed.add(sess)
                continue
            
            if current_bucket is not None and current_bucket >= end_asia_idx:
                # Session has ended and didn't produce the extreme → exclude it
                continue
            
            allowed.add(sess)
        
        if not allowed and extreme_session:
            allowed.add(extreme_session)
        if not allowed:
            allowed.update(session_order)
        
        return allowed
    
    def _asia_filters_for_now(self, bar: Bar, ny_day: date, anchor_day: date, t_bucket_5m: Optional[int]) -> dict:
        """Build filters for Asia sequencing history lookup."""
        symbol = bar.symbol
        st_today = self.engine._get_state(ny_day, symbol)
        st_anchor = self.engine._get_state(anchor_day, symbol) if anchor_day != ny_day else st_today
        f: dict[str, object] = {}

        # AR context (prior range for Asia)
        ar_state = st_anchor.get("AR") if st_anchor else None
        if ar_state and getattr(ar_state, "final_model", None):
            model = str(ar_state.final_model)
            if model and not model.upper().startswith("UNDEFINED"):
                f["ADR_Model"] = model
        if ar_state and getattr(ar_state, "box_std", None) is not None:
            bs = float(ar_state.box_std)
            f["ADR_Box_Direction"] = "up" if bs > 0 else "down" if bs < 0 else "flat"

        # AS follow session (spans midnight, so use anchor state)
        as_state = st_anchor.get("AS") if st_anchor else None
        if as_state and getattr(as_state, "confirmed", False):
            side = as_state.side
            if side in (1, -1):
                f["as_breakout_dir"] = "Long" if side == 1 else "Short"
        if as_state:
            false_flag = getattr(as_state, "false_detected", None)
            f["AS_Session_True"] = False if false_flag is True else True

        # OR range (occurs on the following calendar day)
        or_state = st_today.get("OR") if st_today else None
        if (not or_state or getattr(or_state, "final_model", None) is None) and st_anchor:
            or_state = st_anchor.get("OR")
        if or_state and getattr(or_state, "final_model", None):
            model = str(or_state.final_model)
            if model and not model.upper().startswith("UNDEFINED"):
                f["ODR_Model"] = model

        return f
    
    def _asia_seq_relax_order(self) -> list[str]:
        """Return filter relaxation order for Asia sequencing (least → most important)."""
        return [
            "AS_Session_True",
            "as_breakout_dir",
            "ADR_Box_Direction",
            "ADR_Model",
            "ODR_Model",
        ]
    
    def _map_asia_seq_filter_keys(self, filters: dict, df: pd.DataFrame) -> tuple[dict, list[str]]:
        """Map filter keys to database column names for Asia sequencing."""
        mapping = {
            "ADR_Model": ["ADR_Model"],
            "ADR_Box_Direction": ["AR_Box_Direction"],
            "as_breakout_dir": ["AS_confirm_dir"],
            "AS_Session_True": ["AS_stays_true", "AS_Session_True"],
            "ODR_Model": ["ODR_Model"],
        }
        
        mapped: dict[str, object] = {}
        for key, val in filters.items():
            candidates = mapping.get(key, [key])
            for col in candidates:
                if col in df.columns:
                    mapped[col] = val
                    break
        
        relax_cols: list[str] = []
        for key in self._asia_seq_relax_order():
            candidates = mapping.get(key, [key])
            for col in candidates:
                if col in mapped and col not in relax_cols:
                    relax_cols.append(col)
                    break
        
        return mapped, relax_cols
    
    def _asia_seq_location_probs(self, H: pd.DataFrame, which: str, allowed: Optional[set[str]] = None) -> tuple[float, float, float, float]:
        """Calculate location probabilities for Asia sessions (AS, TO, OR, OS)."""
        if H.empty:
            return (0.25, 0.25, 0.25, 0.25)
        col = "combined_high_where" if which == "high" else "combined_low_where"
        if col not in H.columns:
            return (0.25, 0.25, 0.25, 0.25)
        vals = H[col].astype(str).str.upper().str.strip()
        keys = ["AS", "TO", "OR", "OS"]
        counts = {k: int((vals == k).sum()) for k in keys}
        if allowed is not None:
            alpha = 1.0  # Dirichlet prior to keep allowed sessions in play
            total = sum(counts[k] + (alpha if k in allowed else 0.0) for k in keys)
            if total <= 0:
                uniform = 1.0 / len(allowed) if len(allowed) > 0 else 0.25
                return tuple(uniform if k in allowed else 0.0 for k in keys)
            return tuple(
                (counts[k] + (alpha if k in allowed else 0.0)) / total if k in allowed else 0.0
                for k in keys
            )
        total = sum(counts.values())
        if total <= 0:
            return (0.25, 0.25, 0.25, 0.25)
        return tuple(counts[k] / total for k in keys)
    
    def _asia_zero_impossible_sessions(self, probs: tuple[float, float, float, float], 
                                       idx: Optional[int], anchor_day: date, side: str, 
                                       current_bucket: Optional[int]) -> tuple[float, float, float, float]:
        """Zero out probabilities for sessions that are no longer possible."""
        session_order = ["AS", "TO", "OR", "OS"]
        p_list = list(probs)
        
        extreme_session = None
        if idx is not None:
            extreme_session = self._asia_session_from_5m_idx(anchor_day, idx)
            if extreme_session in session_order:
                cutoff = session_order.index(extreme_session)
                for i in range(cutoff):
                    p_list[i] = 0.0
        
        if current_bucket is not None:
            windows = self._asia_session_windows(anchor_day)
            asia_start = datetime.combine(anchor_day, time(20, 30), tzinfo=ZoneInfo(NY_TZ))
            for sess in session_order:
                start_dt, end_dt = windows[sess]
                end_bucket = int(max(0, (end_dt - asia_start).total_seconds() // 300))
                if current_bucket >= end_bucket and sess != extreme_session:
                    idx_sess = session_order.index(sess)
                    p_list[idx_sess] = 0.0
        
        total = sum(p_list)
        if total <= 0:
            return tuple(0.0 for _ in session_order)
        return tuple(p / total for p in p_list)

    def get_seq_asia_context_v3(self, bar: Bar) -> SeqAsiaOut:
        """Map consolidated Asia tracking to SeqAsiaOut (Asia sequencing).
        
        Window: 20:30 day n → 08:25 day n+1 (AS → TO → OR → OS)
        """
        out = self._ensure_latest_out(bar)
        
        # Check if within Asia window (20:30 → 08:25)
        if not self._within_asia_window(bar.ts):
            return SeqAsiaOut()
        
        seq_hist_raw = self._get_seq_asia_for_symbol(bar.symbol)
        if seq_hist_raw.empty:
            return SeqAsiaOut()
        
        seq_hist = seq_hist_raw.copy()
        
        ny_day = self.engine._ny_date_from_epoch(bar.ts)
        
        # Determine anchor day (day when AS session starts)
        bar_dt = datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ))
        
        bar_date = bar_dt.date()
        bar_time = bar_dt.time()
        if bar_time >= time(20, 30):
            anchor_day = bar_date
        else:
            anchor_day = prev_business_day(bar_date)

        # Calculate current Asia index (5-minute buckets from 20:30)
        start_time = time(20, 30)
        asia_start = datetime.combine(anchor_day, start_time, tzinfo=ZoneInfo(NY_TZ))

        # Calculate 5-minute bucket index and clamp to the window length (≈12 hours)
        delta_seconds = max(0, (bar_dt - asia_start).total_seconds())
        t_bucket_5m = int(delta_seconds // 300)
        t_bucket_5m = min(t_bucket_5m, 143)  # 715 minutes / 5
        t_bucket_30m = self._since_asia_start_bucket(bar.ts)
        asia_start_ts = int(asia_start.timestamp())

        state_today = self.engine._get_state(ny_day, bar.symbol)
        state_anchor = state_today if anchor_day == ny_day else self.engine._get_state(anchor_day, bar.symbol)

        # Restrict history to completed days up to the anchor day
        seq_hist = seq_hist.copy()
        if "day_id" in seq_hist.columns:
            day_col = pd.to_datetime(seq_hist["day_id"], errors="coerce").dt.date
            seq_hist = seq_hist[day_col <= anchor_day]
        if seq_hist.empty:
            seq_hist = seq_hist_raw.copy()

        if "symbol" in seq_hist.columns:
            seq_hist_symbol = seq_hist[seq_hist["symbol"] == bar.symbol].copy()
        else:
            seq_hist_symbol = seq_hist.copy()

        extrema_key = (anchor_day, bar.symbol)
        ext = self._asia_extrema.get(extrema_key)
        if not ext or ext.get("anchor_ts") != asia_start_ts:
            ext = {
                "anchor_ts": asia_start_ts,
                "hi_price": bar.h,
                "hi_ts": bar.ts,
                "lo_price": bar.l,
                "lo_ts": bar.ts,
            }
        else:
            if bar.h > ext.get("hi_price", bar.h):
                ext["hi_price"] = bar.h
                ext["hi_ts"] = bar.ts
            if bar.l < ext.get("lo_price", bar.l):
                ext["lo_price"] = bar.l
                ext["lo_ts"] = bar.ts
        self._asia_extrema[extrema_key] = ext

        asia_high = ext.get("hi_price")
        asia_low = ext.get("lo_price")
        hi_ts = ext.get("hi_ts")
        lo_ts = ext.get("lo_ts")
        hi_idx = int(max(0, (hi_ts - asia_start_ts) // 300)) if hi_ts is not None else None
        lo_idx = int(max(0, (lo_ts - asia_start_ts) // 300)) if lo_ts is not None else None
        
        # Session state (for contextual flags/models)
        ar_state = state_anchor.get("AR") if state_anchor else None

        ar_mid = None
        if ar_state and ar_state.idr_hi_close is not None and ar_state.idr_lo_close is not None:
            ar_mid = (ar_state.idr_hi_close + ar_state.idr_lo_close) / 2.0

        hi_now = None
        lo_now = None
        if ar_mid and ar_mid > 0:
            if asia_high is not None:
                hi_now = (asia_high - ar_mid) / ar_mid * 100
            if asia_low is not None:
                lo_now = (asia_low - ar_mid) / ar_mid * 100
        
        filters = self._asia_filters_for_now(bar, ny_day, anchor_day, t_bucket_5m)
        mapped_filters, relax_cols = self._map_asia_seq_filter_keys(filters, seq_hist)
        sub = self._apply_filters_with_backoff(seq_hist, mapped_filters, relax_cols, self.MIN_SEQ_ROWS)
        if len(sub) < self.MIN_SEQ_ROWS and "AR_Box_Direction" in mapped_filters:
            # Relax AR box direction explicitly when we don't have enough history
            mf2 = dict(mapped_filters)
            mf2.pop("AR_Box_Direction", None)
            relax2 = [c for c in relax_cols if c != "AR_Box_Direction"]
            sub = self._apply_filters_with_backoff(seq_hist, mf2, relax2, self.MIN_SEQ_ROWS)
        
        allowed_high = self._asia_allowed_sessions(anchor_day, t_bucket_5m, hi_idx)
        allowed_low = self._asia_allowed_sessions(anchor_day, t_bucket_5m, lo_idx)
        
        base_quant = self._apply_filters_with_backoff(seq_hist, mapped_filters, relax_cols, self.MIN_SEQ_ROWS)
        if not base_quant.empty and "symbol" in base_quant.columns:
            base_quant = base_quant[base_quant["symbol"] == bar.symbol].copy()
        if base_quant.empty:
            base_quant = seq_hist_symbol.copy() if not seq_hist_symbol.empty else seq_hist.copy()
        
        def _apply_session_gate(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            gated = df
            hi_col, lo_col = self._seq_where_cols(df)
            if hi_col and allowed_high:
                gated = gated[gated[hi_col].astype(str).str.upper().isin(allowed_high)]
            if lo_col and allowed_low:
                gated = gated[gated[lo_col].astype(str).str.upper().isin(allowed_low)]
            return gated
        
        sub_gated = _apply_session_gate(base_quant)
        if sub_gated.empty:
            sub_gated = _apply_session_gate(seq_hist_symbol)
        if sub_gated.empty:
            sub_gated = seq_hist_symbol.copy()
        
        pH = self._asia_seq_location_probs(sub_gated, "high", allowed_high)
        pL = self._asia_seq_location_probs(sub_gated, "low", allowed_low)
        pH = self._asia_zero_impossible_sessions(pH, hi_idx, anchor_day, "high", t_bucket_5m)
        pL = self._asia_zero_impossible_sessions(pL, lo_idx, anchor_day, "low", t_bucket_5m)
        
        quant_source = sub_gated if not sub_gated.empty else base_quant
        if quant_source.empty:
            quant_source = seq_hist_symbol
        pct_hi_col, _ = self._seq_cols(quant_source, "high")
        pct_lo_col, _ = self._seq_cols(quant_source, "low")
        hi_series = pd.to_numeric(quant_source[pct_hi_col], errors="coerce") if pct_hi_col else None
        lo_series_raw = pd.to_numeric(quant_source[pct_lo_col], errors="coerce") if pct_lo_col else None
        q_hi = self._quantiles(hi_series, [0.20, 0.50, 0.80, 0.85]) if hi_series is not None else (None, None, None, None)
        q_lo = self._quantiles(lo_series_raw, [0.20, 0.50, 0.80, 0.85]) if lo_series_raw is not None else (None, None, None, None)
        
        trend_guard_hi = bool(hi_now is not None and q_hi[0] is not None and hi_now >= q_hi[0])
        trend_guard_lo = bool(lo_now is not None and q_lo[2] is not None and lo_now <= q_lo[2])
        failed_20_hi = bool(hi_now is not None and q_hi[0] is not None and hi_now < q_hi[0])
        failed_20_lo = bool(lo_now is not None and q_lo[0] is not None and lo_now > q_lo[0])
        
        def _percentile(series, value, side: str):
            if series is None or value is None:
                return None
            s = pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                return None
            if side == "high":
                total = float(len(s))
                rank = float((s >= value).sum())
                return rank / total if total > 0 else None
            total = float(len(s))
            rank = float((s <= value).sum())
            return rank / total if total > 0 else None
        
        hi_pct_percentile = _percentile(hi_series, hi_now, "high")
        lo_pct_percentile = _percentile(lo_series_raw, lo_now, "low")
        
        p_high_hold_b = self._hold_prob_bucketed(sub_gated, t_bucket_30m, hi_now, "high") if hi_now is not None else 0.5
        p_low_hold_b = self._hold_prob_bucketed(sub_gated, t_bucket_30m, lo_now, "low") if lo_now is not None else 0.5
        p_high_hold_r = self._hold_prob_resid(sub_gated, t_bucket_30m, hi_now, "high") if hi_now is not None else p_high_hold_b
        p_low_hold_r = self._hold_prob_resid(sub_gated, t_bucket_30m, lo_now, "low") if lo_now is not None else p_low_hold_b

        def _clean_prob(val: Optional[float], fallback: float) -> float:
            if isinstance(val, (int, float)) and np.isfinite(val):
                return float(val)
            return float(fallback)

        p_high_hold_b = _clean_prob(p_high_hold_b, 0.5)
        p_low_hold_b = _clean_prob(p_low_hold_b, 0.5)
        p_high_hold_r = _clean_prob(p_high_hold_r, p_high_hold_b)
        p_low_hold_r = _clean_prob(p_low_hold_r, p_low_hold_b)

        p_high_hold_b = 0.5 * (p_high_hold_b + p_high_hold_r)
        p_low_hold_b = 0.5 * (p_low_hold_b + p_low_hold_r)
        
        # At the final 30-minute bucket (08:25), the outcome is already known → holds become 1.0
        if t_bucket_30m is not None:
            max_bucket = 23  # 20:30→08:25 in 30m buckets
            if t_bucket_30m >= max_bucket:
                p_high_hold_b = 1.0
                p_low_hold_b = 1.0
                p_high_hold_r = 1.0
                p_low_hold_r = 1.0
        
        # Session context helpers
        ar_state = state_anchor.get("AR") if state_anchor else None
        as_state = state_anchor.get("AS") if state_anchor else None
        or_state = state_today.get("OR") if state_today else None
        if (not or_state or getattr(or_state, "final_model", None) is None) and state_anchor:
            or_state = state_anchor.get("OR")
        
        ar_model = getattr(ar_state, "final_model", None) if (ar_state and ar_state.final_model) else None
        ar_box = None
        if ar_state and getattr(ar_state, "box_std", None) is not None:
            bs = float(ar_state.box_std)
            ar_box = "up" if bs > 0 else "down" if bs < 0 else "flat"
        if ar_box == "flat":  # treat flat as not informative
            ar_box = None
        
        ar_trade = None
        as_break = None
        as_true = None
        if as_state:
            if getattr(as_state, "confirmed", False) and as_state.side in (1, -1):
                ar_trade = "long" if as_state.side == 1 else "short"
                as_break = ar_trade
                as_true = not bool(as_state.false_detected)
            elif getattr(as_state, "false_detected", None) is True:
                as_true = False
        
        or_model = getattr(or_state, "final_model", None) if (or_state and or_state.final_model) else None
        or_box = None
        if or_state and getattr(or_state, "box_std", None) is not None:
            bs = float(or_state.box_std)
            or_box = "up" if bs > 0 else "down" if bs < 0 else "flat"
        if or_box == "flat":
            or_box = None
        
        return SeqAsiaOut(
            p_cycle_pair=0.5,
            p_high_holds=p_high_hold_r,
            p_low_holds=p_low_hold_r,
            p_asia_high_in_AS=round(pH[0], 3),
            p_asia_high_in_TO=round(pH[1], 3),
            p_asia_high_in_OR=round(pH[2], 3),
            p_asia_high_in_OS=round(pH[3], 3),
            p_asia_low_in_AS=round(pL[0], 3),
            p_asia_low_in_TO=round(pL[1], 3),
            p_asia_low_in_OR=round(pL[2], 3),
            p_asia_low_in_OS=round(pL[3], 3),
            p_high_holds_bucketed=round(p_high_hold_b, 3),
            p_low_holds_bucketed=round(p_low_hold_b, 3),
            p_high_holds_reg=round(p_high_hold_r, 3),
            p_low_holds_reg=round(p_low_hold_r, 3),
            trend_guard_hi=trend_guard_hi,
            trend_guard_lo=trend_guard_lo,
            failed_20_hi=failed_20_hi,
            failed_20_lo=failed_20_lo,
            t_bucket_5m=int(t_bucket_5m) if t_bucket_5m is not None else None,
            hi_pct_now=hi_now,
            lo_pct_now=lo_now,
            hi_pct_percentile=hi_pct_percentile,
            lo_pct_percentile=lo_pct_percentile,
            as_breakout_dir=as_break,
            as_true=as_true,
            ar_model=ar_model,
            ar_box=ar_box,
            ar_trade=ar_trade,
            or_model=or_model,
            or_box=or_box,
        )

    # ──────────────────────────────────────────────────────────────
    # Daycycle sequencing helpers (WDR: 09:30 day n → 09:25 day n+1)
    # ──────────────────────────────────────────────────────────────
    def _daycycle_relax_order(self) -> list[str]:
        """Return filter relaxation order for daycycle filtering (least → most important)."""
        return [
            "regular_box_dir",
            "regular_stays_true",
            "regular_confirm_dir",
            "adr_model",
            "odr_model",
        ]

    def _wdr_session_windows(self, anchor_day: date) -> Dict[str, Tuple[datetime, datetime]]:
        """Get session windows for WDR cycle (09:30 anchor_day → 09:25 next day).
        
        Sessions in order:
        - RR (09:30-10:25 anchor_day)
        - RS (10:30-15:55 anchor_day)
        - TOUT (16:00-16:55 anchor_day)
        - TA (18:00-19:25 anchor_day)
        - AR (19:30-20:25 anchor_day)
        - AS (20:30-01:55 next_day)
        - TO (02:00-02:55 next_day)
        - OR (03:00-03:55 next_day)
        - OS (04:00-08:25 next_day)
        - TR (08:30-09:25 next_day)
        """
        tz = ZoneInfo(NY_TZ)
        next_day = anchor_day + timedelta(days=1)
        
        # Day n sessions
        rr_start = datetime.combine(anchor_day, time(9, 30), tzinfo=tz)
        rr_end = datetime.combine(anchor_day, time(10, 25), tzinfo=tz)
        rs_start = datetime.combine(anchor_day, time(10, 30), tzinfo=tz)
        rs_end = datetime.combine(anchor_day, time(15, 55), tzinfo=tz)
        tout_start = datetime.combine(anchor_day, time(16, 0), tzinfo=tz)
        tout_end = datetime.combine(anchor_day, time(16, 55), tzinfo=tz)
        ta_start = datetime.combine(anchor_day, time(18, 0), tzinfo=tz)
        ta_end = datetime.combine(anchor_day, time(19, 25), tzinfo=tz)
        ar_start = datetime.combine(anchor_day, time(19, 30), tzinfo=tz)
        ar_end = datetime.combine(anchor_day, time(20, 25), tzinfo=tz)
        
        # Day n+1 sessions
        as_start = datetime.combine(anchor_day, time(20, 30), tzinfo=tz)
        as_end = datetime.combine(next_day, time(1, 55), tzinfo=tz)
        to_start = datetime.combine(next_day, time(2, 0), tzinfo=tz)
        to_end = datetime.combine(next_day, time(2, 55), tzinfo=tz)
        or_start = datetime.combine(next_day, time(3, 0), tzinfo=tz)
        or_end = datetime.combine(next_day, time(3, 55), tzinfo=tz)
        os_start = datetime.combine(next_day, time(4, 0), tzinfo=tz)
        os_end = datetime.combine(next_day, time(8, 25), tzinfo=tz)
        tr_start = datetime.combine(next_day, time(8, 30), tzinfo=tz)
        tr_end = datetime.combine(next_day, time(9, 25), tzinfo=tz)
        
        return {
            "RR": (rr_start, rr_end),
            "RS": (rs_start, rs_end),
            "TOUT": (tout_start, tout_end),
            "TA": (ta_start, ta_end),
            "AR": (ar_start, ar_end),
            "AS": (as_start, as_end),
            "TO": (to_start, to_end),
            "OR": (or_start, or_end),
            "OS": (os_start, os_end),
            "TR": (tr_start, tr_end),
        }

    def _session_from_wdr_idx(self, anchor_day: date, wdr_idx: Optional[int]) -> Optional[str]:
        """Map WDR index to session name.
        
        WDR index starts at 0 at 09:30 anchor_day (start of RR session).
        Each 5-minute bar increments the index.
        """
        if wdr_idx is None:
            return None
        
        windows = self._wdr_session_windows(anchor_day)
        wdr_start = datetime.combine(anchor_day, time(9, 30), tzinfo=ZoneInfo(NY_TZ))
        ts = wdr_start + timedelta(minutes=5 * wdr_idx)
        
        # Check each session window
        for name, (start, end) in windows.items():
            if start <= ts <= end:
                return name
        
        # If past all windows, return last session (TR)
        return "TR"

    def _allowed_sessions_wdr(self, anchor_day: date, current_wdr_idx: Optional[int], 
                               extreme_wdr_idx: Optional[int]) -> set[str]:
        """Determine which sessions are still possible for WDR extremes.
        
        Similar to intraday logic: if a session has ended and didn't produce
        the extreme, exclude it. If we know the extreme location, only allow
        that session and later ones.
        """
        session_order = ["RR", "RS", "TOUT", "TA", "AR", "AS", "TO", "OR", "OS", "TR"]
        windows = self._wdr_session_windows(anchor_day)
        wdr_start = datetime.combine(anchor_day, time(9, 30), tzinfo=ZoneInfo(NY_TZ))
        
        extreme_session = self._session_from_wdr_idx(anchor_day, extreme_wdr_idx)
        allowed: set[str] = set()
        
        for sess in session_order:
            start_dt, end_dt = windows[sess]
            # Calculate WDR index for session end
            end_wdr_idx = int(max(0, (end_dt - wdr_start).total_seconds() // 300))
            
            if extreme_session == sess:
                # Extreme is in this session, allow it
                allowed.add(sess)
                continue
            
            if current_wdr_idx is not None and current_wdr_idx >= end_wdr_idx:
                # Session has ended and didn't produce the extreme → exclude it
                continue
            
            allowed.add(sess)
        
        if not allowed and extreme_session:
            allowed.add(extreme_session)
        if not allowed:
            allowed.update(session_order)
        
        return allowed

    def _parse_cycle_pair_ordered(self, pair_str: str) -> Optional[Tuple[str, str]]:
        """Parse cycle_pair_ordered string (e.g., "RR-RS") into (high_session, low_session).
        
        Format is always high then low, separated by a delimiter (hyphen or similar).
        """
        if not pair_str or pd.isna(pair_str):
            return None
        
        pair_str = str(pair_str).strip().upper()
        # Try common delimiters
        for sep in ["-", "_", " "]:
            if sep in pair_str:
                parts = pair_str.split(sep, 1)
                if len(parts) == 2:
                    return (parts[0].strip(), parts[1].strip())
        
        # If no delimiter found, assume single session (shouldn't happen but handle gracefully)
        return None

    def _gate_cycle_pairs(self, df: pd.DataFrame, allowed_high: set[str], 
                          allowed_low: set[str]) -> pd.DataFrame:
        """Filter cycle pairs based on allowed sessions.
        
        Filters rows where cycle_pair_ordered indicates both high and low
        sessions are still possible.
        """
        if df.empty or "cycle_pair_ordered" not in df.columns:
            return df
        
        gated = df.copy()
        
        def is_allowed(pair_str):
            parsed = self._parse_cycle_pair_ordered(pair_str)
            if parsed is None:
                return True  # Keep rows with invalid pairs (let other filters handle)
            high_sess, low_sess = parsed
            return high_sess in allowed_high and low_sess in allowed_low
        
        mask = gated["cycle_pair_ordered"].apply(is_allowed)
        return gated[mask]

    def _daycycle_filters_for_now(self, bar: Bar, ny_day: date, seq_hist: pd.DataFrame, out: Dict) -> tuple[dict, date, Optional[str], Optional[str]]:
        """Build filters for daycycle history lookup from consolidated output.

        Returns:
            (filters_dict, anchor_day, ar_model_live, or_model_live)
        """
        filters: dict[str, object] = {}
        anchor_day = ny_day
        
        # Determine anchor day: WDR cycle starts at 09:30 day n, ends at 09:25 day n+1
        # For bars before 09:30, use previous day as anchor
        bar_dt = datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ))
        if bar_dt.hour < 9 or (bar_dt.hour == 9 and bar_dt.minute < 30):
            anchor_day = ny_day - timedelta(days=1)
        
        cache_key = (anchor_day, bar.symbol)
        
        # Get state from both anchor_day and ny_day for proper midnight crossover handling
        st_anchor = self.engine._get_state(anchor_day, bar.symbol) if anchor_day != ny_day else None
        st_current = self.engine._get_state(ny_day, bar.symbol)
        st = st_current  # Use current day state as primary for backward compatibility
        
        # Get AR model from session_range_attrs (persists when AR completes)
        # AR spans midnight, so check anchor_day first (if different), then current day
        ar_model_live = None
        if st_anchor:
            ar_attrs = st_anchor.get("session_range_attrs", {}).get("AR", {})
            if ar_attrs and ar_attrs.get("complete") and ar_attrs.get("model"):
                model = str(ar_attrs["model"])
                if model and not model.upper().startswith("UNDEFINED"):
                    ar_model_live = model
                    filters["adr_model"] = model
        
        # Check current day for AR if not found on anchor_day
        if ar_model_live is None:
            ar_attrs = st.get("session_range_attrs", {}).get("AR", {})
            if ar_attrs and ar_attrs.get("complete") and ar_attrs.get("model"):
                model = str(ar_attrs["model"])
                if model and not model.upper().startswith("UNDEFINED"):
                    ar_model_live = model
                    filters["adr_model"] = model
        
        # Check cache for AR model if not available from current state
        if ar_model_live is None:
            cached_ar = self._daycycle_model_cache.get(cache_key, {}).get("AR_Model")
            if cached_ar:
                ar_model_live = cached_ar
                filters["adr_model"] = cached_ar
        
        # Get OR model from session_range_attrs (persists when OR completes)
        # OR completes before midnight, so check anchor_day first
        or_model_live = None
        or_attrs = None
        if st_anchor:
            or_attrs = st_anchor.get("session_range_attrs", {}).get("OR", {})
        if (not or_attrs or not or_attrs.get("complete")) and st_current:
            or_attrs = st_current.get("session_range_attrs", {}).get("OR", {})
        
        if or_attrs and or_attrs.get("complete") and or_attrs.get("model"):
            model = str(or_attrs["model"])
            if model and not model.upper().startswith("UNDEFINED"):
                or_model_live = model
                filters["odr_model"] = model
        
        # Check cache for OR model if not available from current state
        if or_model_live is None:
            cached_or = self._daycycle_model_cache.get(cache_key, {}).get("OR_Model")
            if cached_or:
                or_model_live = cached_or
                filters["odr_model"] = cached_or
        
        # Get RR box direction from session_range_attrs (captured when RR completes at 10:25)
        # RR completes before midnight, so check anchor_day first
        rr_box_dir = None
        rr_attrs = None
        if st_anchor:
            rr_attrs = st_anchor.get("session_range_attrs", {}).get("RR", {})
        if (not rr_attrs or not rr_attrs.get("complete")) and st_current:
            rr_attrs = st_current.get("session_range_attrs", {}).get("RR", {})
        
        if rr_attrs and rr_attrs.get("complete") and rr_attrs.get("box_dir"):
            box_dir = str(rr_attrs["box_dir"])
            if box_dir != "flat":
                rr_box_dir = box_dir
                filters["regular_box_dir"] = box_dir
                # Cache for later use
                if cache_key not in self._daycycle_model_cache:
                    self._daycycle_model_cache[cache_key] = {}
                self._daycycle_model_cache[cache_key]["RR_Box_Direction"] = box_dir
        
        # Check cache for RR box direction if not available from current state
        if "regular_box_dir" not in filters:
            cached_box = self._daycycle_model_cache.get(cache_key, {}).get("RR_Box_Direction")
            if cached_box and cached_box != "flat":
                filters["regular_box_dir"] = cached_box
        
        # Get RS confirmation direction and truth (from RS session)
        # RS completes before midnight, so check anchor_day first
        rs_state = st_anchor.get("RS") if st_anchor else None
        if (not rs_state or not getattr(rs_state, "confirmed", False)) and anchor_day != ny_day:
            # Fallback to current day if not found on anchor_day
            if st_current:
                rs_state = st_current.get("RS")
        
        if rs_state and getattr(rs_state, "confirmed", False):
            side = rs_state.side
            if side in (1, -1):
                confirm_dir = "Long" if side == 1 else "Short"
                filters["regular_confirm_dir"] = confirm_dir
                # Session true: confirmed and not false
                stays_true = bool(rs_state.confirmed and not rs_state.false_detected)
                filters["regular_stays_true"] = stays_true
        
        # Update cache with AR/OR models when they become available
        if cache_key not in self._daycycle_model_cache:
            self._daycycle_model_cache[cache_key] = {}
        if ar_model_live:
            self._daycycle_model_cache[cache_key]["AR_Model"] = ar_model_live
        if or_model_live:
            self._daycycle_model_cache[cache_key]["OR_Model"] = or_model_live
        
        return filters, anchor_day, ar_model_live, or_model_live

    def get_seq_daycycle_context_v3(self, bar: Bar) -> SeqDaycycleOut:
        """Map consolidated WDR tracking to SeqDaycycleOut (daycycle sequencing).

        Window: 09:30 day n → 09:25 day n+1 (WDR cycle)
        """
        out = self._ensure_latest_out(bar)
        
        # Check if within WDR window (09:30 → 09:25 next day)
        bar_dt = datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ))
        hour = bar_dt.hour
        minute = bar_dt.minute
        bar_time = bar_dt.time()
        is_wdr_window = (bar_time >= time(9, 30)) or (bar_time <= time(9, 25))
        
        if not is_wdr_window:
            return SeqDaycycleOut()
        
        seq_hist = self._get_seq_daycycle_for_symbol(bar.symbol)
        if seq_hist.empty:
            return SeqDaycycleOut()
        
        if "symbol" in seq_hist.columns:
            seq_hist_symbol = seq_hist[seq_hist["symbol"] == bar.symbol].copy()
        else:
            seq_hist_symbol = seq_hist.copy()
        
        ny_day = self.engine._ny_date_from_epoch(bar.ts)
        
        # Build filters from consolidated output
        filters, anchor_day, ar_model_live, or_model_live = self._daycycle_filters_for_now(
            bar, ny_day, seq_hist_symbol, out
        )
        
        # Filter history with backoff
        relax_order = self._daycycle_relax_order()
        
        # Filter to dates before anchor day (WDR cycle start day)
        # Use anchor_day instead of ny_day to handle midnight crossover correctly
        hist_slice = seq_hist_symbol.copy()
        if not hist_slice.empty and "day_id" in hist_slice.columns:
            hist_slice["day_id"] = pd.to_datetime(hist_slice["day_id"], errors="coerce")
            prior = hist_slice[hist_slice["day_id"] < pd.Timestamp(anchor_day)]
            hist_slice = prior if not prior.empty else hist_slice
        
        # Apply filters with backoff
        sub = self._apply_filters_with_backoff(hist_slice, filters, relax_order, self.MIN_SEQ_ROWS)
        
        # If sample too small, try relaxing regular_box_dir explicitly
        if len(sub) < self.MIN_SEQ_ROWS and "regular_box_dir" in filters:
            filters_no_box = {k: v for k, v in filters.items() if k != "regular_box_dir"}
            relax_no_box = [k for k in relax_order if k != "regular_box_dir"]
            sub = self._apply_filters_with_backoff(hist_slice, filters_no_box, relax_no_box, self.MIN_SEQ_ROWS)
        
        if sub.empty:
            sub = hist_slice.copy()
        
        # Get WDR values from consolidated output
        wdr_hi_pct = out.get("wdr_high_pct")
        wdr_lo_pct = out.get("wdr_low_pct")
        wdr_idx = out.get("current_wdr_time_idx")
        wdr_hi_idx = out.get("wdr_high_time_idx")
        wdr_lo_idx = out.get("wdr_low_time_idx")
        
        # Calculate allowed sessions for high and low based on WDR indices
        allowed_high = self._allowed_sessions_wdr(anchor_day, wdr_idx, wdr_hi_idx)
        allowed_low = self._allowed_sessions_wdr(anchor_day, wdr_idx, wdr_lo_idx)
        
        # Gate cycle pairs based on allowed sessions
        sub_gated = self._gate_cycle_pairs(sub, allowed_high, allowed_low)
        if sub_gated.empty:
            # Fallback: try gating on hist_slice if sub_gated is empty
            sub_gated = self._gate_cycle_pairs(hist_slice, allowed_high, allowed_low)
        if sub_gated.empty:
            # Final fallback: use ungated sub
            sub_gated = sub.copy()
        
        # Find percentile columns
        pct_hi_col = "combined_high_pct_change" if "combined_high_pct_change" in sub_gated.columns else None
        pct_lo_col = "combined_low_pct_change" if "combined_low_pct_change" in sub_gated.columns else None
        idx_hi_col = "combined_high_idx" if "combined_high_idx" in sub_gated.columns else None
        idx_lo_col = "combined_low_idx" if "combined_low_idx" in sub_gated.columns else None
        
        # Calculate percentiles from gated data
        def percentile(series: Optional[pd.Series], value, side: str) -> Optional[float]:
            if series is None or value is None:
                return None
            s = pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                return None
            if side == "high":
                total = float(len(s))
                rank = float((s >= value).sum())
                return rank / total if total > 0 else None
            total = float(len(s))
            rank = float((s <= value).sum())
            return rank / total if total > 0 else None
        
        hi_series = pd.to_numeric(sub_gated[pct_hi_col], errors="coerce") if pct_hi_col else None
        lo_series = pd.to_numeric(sub_gated[pct_lo_col], errors="coerce") if pct_lo_col else None
        
        hi_pct_percentile = percentile(hi_series, wdr_hi_pct, "high")
        lo_pct_percentile = percentile(lo_series, wdr_lo_pct, "low")
        
        # Calculate hold probabilities from gated data
        def hold_prob(df: pd.DataFrame, col: Optional[str], current_idx: Optional[int]) -> float:
            if col is None or current_idx is None or col not in df.columns:
                return 0.5
            idx_vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if idx_vals.empty:
                return 0.5
            y = (idx_vals <= current_idx).astype(int)
            n = len(y)
            if n == 0:
                return 0.5
            # Laplace smoothing
            return float((y.sum() + 1.0) / (n + 2.0))
        
        p_hi_hold = hold_prob(sub_gated, idx_hi_col, wdr_idx)
        p_lo_hold = hold_prob(sub_gated, idx_lo_col, wdr_idx)
        
        # Calculate cycle pair hold probabilities from gated data
        # These probabilities are based on the cycle pairs that remain possible after gating
        p_cycle_pair_high_hold = 0.5
        p_cycle_pair_low_hold = 0.5
        
        if "cycle_pair_ordered" in sub_gated.columns and not sub_gated.empty and idx_hi_col and idx_lo_col:
            # Calculate hold probabilities specifically from cycle pairs
            # For high: probability that high index <= current index, based on cycle pairs
            hi_idx_vals = pd.to_numeric(sub_gated[idx_hi_col], errors="coerce").dropna()
            if not hi_idx_vals.empty and wdr_idx is not None:
                y_high = (hi_idx_vals <= wdr_idx).astype(int)
                n_high = len(y_high)
                if n_high > 0:
                    # Laplace smoothing
                    p_cycle_pair_high_hold = float((y_high.sum() + 1.0) / (n_high + 2.0))
            
            # For low: probability that low index <= current index, based on cycle pairs
            lo_idx_vals = pd.to_numeric(sub_gated[idx_lo_col], errors="coerce").dropna()
            if not lo_idx_vals.empty and wdr_idx is not None:
                y_low = (lo_idx_vals <= wdr_idx).astype(int)
                n_low = len(y_low)
                if n_low > 0:
                    # Laplace smoothing
                    p_cycle_pair_low_hold = float((y_low.sum() + 1.0) / (n_low + 2.0))
        else:
            # Fallback to regular hold probabilities if cycle pair data not available
            p_cycle_pair_high_hold = p_hi_hold
            p_cycle_pair_low_hold = p_lo_hold
        
        # Calculate cycle pair probabilities from gated data
        cycle_high_probs: Dict[str, float] = {}
        cycle_low_probs: Dict[str, float] = {}
        p_cycle_pair_24h = 0.5
        
        if "cycle_pair_ordered" in sub_gated.columns and not sub_gated.empty:
            # Count cycle pairs for high locations
            pair_counts_high: Dict[str, int] = {}
            pair_counts_low: Dict[str, int] = {}
            total_pairs = 0
            
            for _, row in sub_gated.iterrows():
                pair_str = row.get("cycle_pair_ordered")
                parsed = self._parse_cycle_pair_ordered(pair_str)
                if parsed:
                    high_sess, low_sess = parsed
                    pair_counts_high[high_sess] = pair_counts_high.get(high_sess, 0) + 1
                    pair_counts_low[low_sess] = pair_counts_low.get(low_sess, 0) + 1
                    total_pairs += 1
            
            # Normalize to probabilities (with Laplace smoothing)
            if total_pairs > 0:
                # High location probabilities
                all_sessions = set(pair_counts_high.keys()) | set(pair_counts_low.keys())
                # Default sessions list for consistent normalization
                default_sessions = ["RR", "RS", "AS", "AR", "OR", "OS", "TR"]
                all_sessions = all_sessions | set(default_sessions)
                
                # Normalize probabilities: (count + 1) / (total_pairs + num_sessions)
                # Each pair contributes to exactly one high location and one low location
                denominator = total_pairs + len(all_sessions)
                
                for sess in all_sessions:
                    count_high = pair_counts_high.get(sess, 0)
                    count_low = pair_counts_low.get(sess, 0)
                    # Laplace smoothing: (count + 1) / (total + num_sessions)
                    cycle_high_probs[sess] = (count_high + 1.0) / denominator
                    cycle_low_probs[sess] = (count_low + 1.0) / denominator
                
                # Calculate p_cycle_pair_24h: probability of the most likely cycle pair
                # This is a placeholder - in practice, you might want to match the current
                # high/low location pair and use its probability
                if wdr_hi_idx is not None and wdr_lo_idx is not None:
                    hi_sess = self._session_from_wdr_idx(anchor_day, wdr_hi_idx)
                    lo_sess = self._session_from_wdr_idx(anchor_day, wdr_lo_idx)
                    if hi_sess and lo_sess:
                        # Find matching pairs
                        matching = sub_gated[
                            sub_gated["cycle_pair_ordered"].apply(
                                lambda x: self._parse_cycle_pair_ordered(x) == (hi_sess, lo_sess)
                            )
                        ]
                        if len(matching) > 0:
                            p_cycle_pair_24h = len(matching) / len(sub_gated)
                        else:
                            # Use uniform over all pairs if no exact match
                            p_cycle_pair_24h = 1.0 / max(1, len(sub_gated["cycle_pair_ordered"].unique()))
            else:
                # Default uniform probabilities if no data
                default_sessions = ["RR", "RS", "TOUT", "TA", "AR", "AS", "TO", "OR", "OS", "TR"]
                uniform = 1.0 / len(default_sessions)
                cycle_high_probs = {s: uniform for s in default_sessions}
                cycle_low_probs = {s: uniform for s in default_sessions}
        
        # Get current session attributes for output
        # RS completes before midnight, so check anchor_day state first (WDR cycle start day)
        st = self.engine._get_state(anchor_day, bar.symbol)
        
        # If RS state not found on anchor_day, check current ny_day (for forward compatibility)
        rs_state = st.get("RS")
        if not rs_state or not getattr(rs_state, "confirmed", False):
            st_current = self.engine._get_state(ny_day, bar.symbol)
            rs_state = st_current.get("RS") if st_current else None
        
        regular_confirm_dir = None
        regular_stays_true = None
        if rs_state and getattr(rs_state, "confirmed", False):
            side = rs_state.side
            if side in (1, -1):
                regular_confirm_dir = "Long" if side == 1 else "Short"
                regular_stays_true = bool(rs_state.confirmed and not rs_state.false_detected)
        
        regular_box_dir = None
        # RR completes before midnight, so check anchor_day state first
        rr_attrs = st.get("session_range_attrs", {}).get("RR", {})
        if (not rr_attrs or not rr_attrs.get("complete")) and anchor_day != ny_day:
            # Fallback to current day if not found on anchor_day
            st_current = self.engine._get_state(ny_day, bar.symbol)
            if st_current:
                rr_attrs = st_current.get("session_range_attrs", {}).get("RR", {})
        if rr_attrs and rr_attrs.get("complete") and rr_attrs.get("box_dir"):
            box_dir = str(rr_attrs["box_dir"])
            if box_dir != "flat":
                regular_box_dir = box_dir
        
        # Check cache for regular_box_dir if not available
        cache_key = (anchor_day, bar.symbol)
        if regular_box_dir is None:
            cached_box = self._daycycle_model_cache.get(cache_key, {}).get("RR_Box_Direction")
            if cached_box and cached_box != "flat":
                regular_box_dir = cached_box
        
        return SeqDaycycleOut(
            p_cycle_pair_24h=round(p_cycle_pair_24h, 3),
            p_hi_hold_24h=round(p_hi_hold, 3),
            p_lo_hold_24h=round(p_lo_hold, 3),
            p_cycle_pair_high_hold=round(p_cycle_pair_high_hold, 3),
            p_cycle_pair_low_hold=round(p_cycle_pair_low_hold, 3),
            cycle_high_probs=cycle_high_probs,
            cycle_low_probs=cycle_low_probs,
            hi_pct_percentile=round(hi_pct_percentile, 3) if hi_pct_percentile is not None else None,
            lo_pct_percentile=round(lo_pct_percentile, 3) if lo_pct_percentile is not None else None,
            wdr_high_pct=wdr_hi_pct,
            wdr_low_pct=wdr_lo_pct,
            current_wdr_idx=int(wdr_idx) if wdr_idx is not None else None,
            wdr_high_idx=int(wdr_hi_idx) if wdr_hi_idx is not None else None,
            wdr_low_idx=int(wdr_lo_idx) if wdr_lo_idx is not None else None,
            hist_rows=len(hist_slice),
            hist_filtered_rows=len(sub_gated),
            regular_confirm_dir=regular_confirm_dir,
            regular_stays_true=regular_stays_true,
            regular_box_dir=regular_box_dir,
            filter_asia_model=ar_model_live,
            filter_overnight_model=or_model_live,
        )

    def get_first_range_candle_context_v3(self, bar: Bar) -> FirstRangeCandleOut:
        """Return probabilities for the first range candle context."""
        out = self._ensure_latest_out(bar)

        symbol = out.get("symbol")
        if not symbol:
            return FirstRangeCandleOut()

        current_session = out.get("current_session")
        session_map = {"AS": "AR", "OS": "OR", "RS": "RR"}
        range_session = None
        if isinstance(current_session, str):
            if current_session in ("AR", "OR", "RR"):
                range_session = current_session
            else:
                range_session = session_map.get(current_session)

        df = self._get_first_range_candle_df(symbol)
        if df.empty:
            return FirstRangeCandleOut()

        subset = df
        if range_session is not None and "range_session" in df.columns:
            scoped = df[df["range_session"] == range_session]
            if not scoped.empty:
                subset = scoped

        current_session_model = out.get("current_session_model")
        if isinstance(current_session_model, str) and current_session_model and current_session_model != "Undefined":
            column_model = "current_session_model"
            if column_model in subset.columns:
                model_filtered = subset[subset[column_model].astype(str) == current_session_model]
                if not model_filtered.empty:
                    subset = model_filtered

        break_dir = out.get("first_break_dir")
        wick_dir = out.get("first_break_dir_via_wick")
        if isinstance(break_dir, str) and break_dir:
            column = "first_break_dir"
            if column in subset.columns:
                filtered = subset[subset[column].astype(str) == break_dir]
            else:
                filtered = pd.DataFrame()
            if filtered.empty and isinstance(wick_dir, str) and wick_dir:
                column_wick = "first_break_dir_via_wick"
                if column_wick in subset.columns:
                    filtered = subset[subset[column_wick].astype(str) == wick_dir]
            if not filtered.empty:
                subset = filtered
        elif isinstance(wick_dir, str) and wick_dir:
            column_wick = "first_break_dir_via_wick"
            if column_wick in subset.columns:
                filtered = subset[subset[column_wick].astype(str) == wick_dir]
                if not filtered.empty:
                    subset = filtered

        sample_size = int(len(subset))

        box_probs = self._distribution(subset.get("session_box_dir"), ("up", "down"))
        touched_rbc_prob = self._bool_probability(subset.get("touched_rbc"))
        touched_m7_prob = self._bool_probability(subset.get("touched_m7"))
        first_break_true_range_only_prob = self._bool_probability(subset.get("first_break_true_range_only"))
        first_break_true_wick_prob = self._bool_probability(subset.get("first_break_true_via_wick_range_only"))

        return FirstRangeCandleOut(
            sample_size=sample_size,
            p_session_box_dir=box_probs,
            p_session_trade_direction={},
            p_session_true=0.5,
            p_current_session_model={},
            p_first_break_true_via_wick=first_break_true_wick_prob,
            p_first_break_true_range_only=first_break_true_range_only_prob,
            p_touched_rbc=touched_rbc_prob,
            p_touched_m7=touched_m7_prob,
        )

    def get_weekly_context_v3(self, bar: Bar) -> WeeklyOut:
        """Weekly adapter: filter by Wed-vs-Tue model and compute hold probabilities."""
        out = self._ensure_latest_out(bar)
        week_model = out.get("wed_overnight_vs_tue_regular_model")
        if not isinstance(week_model, str) or not week_model or week_model.startswith("Undefined"):
            return WeeklyOut()

        hist = self._get_weekly_model_for_symbol(bar.symbol)
        if hist.empty:
            return WeeklyOut()

        df = hist.copy()
        if "symbol" in df.columns:
            df = df[df["symbol"] == bar.symbol]
        sample_size = len(df)
        if sample_size == 0:
            return WeeklyOut()

        ny_day = out.get("ny_day")
        if isinstance(ny_day, pd.Timestamp):
            ny_day = ny_day.date()
        elif isinstance(ny_day, datetime):
            ny_day = ny_day.date()
        if not isinstance(ny_day, date):
            ny_day = self.engine._ny_date_from_epoch(bar.ts)
        week_start = trading_week_tuesday(ny_day)

        hist_slice = df
        hist_rows = len(hist_slice)
        if "week_id" in df.columns:
            week_ids = pd.to_datetime(df["week_id"], errors="coerce")
            mask = week_ids.dt.date < week_start
            filtered = df[mask]
            if not filtered.empty:
                hist_slice = filtered
                hist_rows = len(hist_slice)

        filtered = hist_slice
        model_col = "day1_tue_overnight_vs_regular_model"
        if model_col in filtered.columns:
            scoped = filtered[filtered[model_col].astype(str) == str(week_model)]
            if not scoped.empty:
                filtered = scoped
        filtered_rows = len(filtered)
        if filtered.empty:
            filtered = hist_slice
            filtered_rows = len(filtered)

        day_number = out.get("day_number")
        day_number = int(day_number) if day_number is not None else None
        week_hi_pct = self._safe_float(out.get("week_high_pct_vs_tue_mid"))
        week_lo_pct = self._safe_float(out.get("week_low_pct_vs_tue_mid"))
        day_hi_pct = self._safe_float(out.get("day_high_pct_vs_tue_mid"))
        day_lo_pct = self._safe_float(out.get("day_low_pct_vs_tue_mid"))

        if day_number is not None and week_hi_pct is not None:
            def label_week_hi(series: pd.Series) -> pd.Series:
                return (series <= day_number).astype(int)
            p_week_hi_hold = self._weekly_residual_probability(
                filtered,
                "weekly_high_pct_vs_tue_mid",
                "day_of_weekly_high",
                day_number,
                week_hi_pct,
                label_week_hi,
                side="high",
            )
        else:
            p_week_hi_hold = 0.5

        if day_number is not None and week_lo_pct is not None:
            def label_week_lo(series: pd.Series) -> pd.Series:
                return (series <= day_number).astype(int)
            p_week_lo_hold = self._weekly_residual_probability(
                filtered,
                "weekly_low_pct_vs_tue_mid",
                "day_of_weekly_low",
                day_number,
                week_lo_pct,
                label_week_lo,
                side="low",
            )
        else:
            p_week_lo_hold = 0.5

        day_name = DAY_NAME.get(day_number) if day_number is not None else None
        if day_name:
            day_hi_col = f"day{day_number}_{day_name}_high_pct_vs_tue_mid"
            if day_hi_col in filtered.columns and day_hi_pct is not None:
                def label_day_hi(series: pd.Series) -> pd.Series:
                    return (series == day_number).astype(int)
                p_day_hi_hold = self._weekly_residual_probability(
                    filtered,
                    day_hi_col,
                    "day_of_weekly_high",
                    day_number,
                    day_hi_pct,
                    label_day_hi,
                    side="high",
                )
            else:
                p_day_hi_hold = 0.5

            day_lo_col = f"day{day_number}_{day_name}_low_pct_vs_tue_mid"
            if day_lo_col in filtered.columns and day_lo_pct is not None:
                def label_day_lo(series: pd.Series) -> pd.Series:
                    return (series == day_number).astype(int)
                p_day_lo_hold = self._weekly_residual_probability(
                    filtered,
                    day_lo_col,
                    "day_of_weekly_low",
                    day_number,
                    day_lo_pct,
                    label_day_lo,
                    side="low",
                )
            else:
                p_day_lo_hold = 0.5
        else:
            p_day_hi_hold = 0.5
            p_day_lo_hold = 0.5

        week_hi_percentile = self._percentile_reached(filtered.get("weekly_high_pct_vs_tue_mid"), week_hi_pct)
        week_lo_percentile = self._percentile_reached(filtered.get("weekly_low_pct_vs_tue_mid"), week_lo_pct, use_abs=True)

        if day_number is not None and day_name:
            day_hi_col = f"day{day_number}_{day_name}_high_pct_vs_tue_mid"
            day_lo_col = f"day{day_number}_{day_name}_low_pct_vs_tue_mid"
            day_hi_percentile = self._percentile_reached(filtered.get(day_hi_col), day_hi_pct)
            day_lo_percentile = self._percentile_reached(filtered.get(day_lo_col), day_lo_pct, use_abs=True)
        else:
            day_hi_percentile = None
            day_lo_percentile = None

        return WeeklyOut(
            sample_size=sample_size,
            hist_rows=hist_rows,
            filtered_rows=filtered_rows,
            filter_model=str(week_model),
            day_number=day_number,
            week_high_pct_vs_tue_mid=week_hi_pct,
            week_low_pct_vs_tue_mid=week_lo_pct,
            day_high_pct_vs_tue_mid=day_hi_pct,
            day_low_pct_vs_tue_mid=day_lo_pct,
            p_week_high_holds=round(p_week_hi_hold, 3),
            p_week_low_holds=round(p_week_lo_hold, 3),
            week_high_percentile_reached=round(week_hi_percentile, 3) if week_hi_percentile is not None else None,
            week_low_percentile_reached=round(week_lo_percentile, 3) if week_lo_percentile is not None else None,
            current_day_high_percentile_reached=round(day_hi_percentile, 3) if day_hi_percentile is not None else None,
            current_day_low_percentile_reached=round(day_lo_percentile, 3) if day_lo_percentile is not None else None,
        )

    def get_consolidated_probability_output_v3(self, bar: Bar) -> Dict[str, float]:
        """Return a flat dict combining all probability/percentile metrics."""
        range_out = self.get_range_session_metrics_v3(bar)
        box_out = self.get_box_daily_context_v3(bar)
        modeldb_out = self.get_modeldb_live_v3(bar)
        seq_intraday_out = self.get_seq_intraday_context_v3(bar)
        seq_asia_out = self.get_seq_asia_context_v3(bar)
        seq_daycycle_out = self.get_seq_daycycle_context_v3(bar)
        wddrb_out = self.get_wddrb_context(bar)
        wddrb_model_out = self.get_wddrb_model_context(bar)
        first_range_out = self.get_first_range_candle_context_v3(bar)
        weekly_out = self.get_weekly_context_v3(bar)

        contexts: Dict[str, Dict[str, Any]] = {
            "range": asdict(range_out),
            "box_daily": asdict(box_out),
            "modeldb": modeldb_out,
            "seq_intraday": asdict(seq_intraday_out),
            "seq_asia": asdict(seq_asia_out),
            "seq_daycycle": asdict(seq_daycycle_out),
            "wddrb": asdict(wddrb_out),
            "wddrb_model": asdict(wddrb_model_out),
            "first_range_candle": asdict(first_range_out),
            "weekly": asdict(weekly_out),
        }

        aggregated: Dict[str, float] = {}
        for prefix, payload in contexts.items():
            aggregated.update(self._extract_probability_fields(payload, prefix))
        return aggregated

    def _extract_probability_fields(self, payload: Optional[Dict[str, Any]], prefix: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if not payload:
            return metrics
        for key, value in payload.items():
            if value is None or not self._is_probability_key(key):
                continue
            self._flatten_probability_value(metrics, prefix, key, value)
        return metrics

    def _flatten_probability_value(self, target: Dict[str, float], prefix: str, key: str, value: Any) -> None:
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                self._flatten_probability_value(target, prefix, f"{key}_{sub_key}", sub_val)
            return
        if isinstance(value, (list, tuple)):
            for idx, sub_val in enumerate(value):
                self._flatten_probability_value(target, prefix, f"{key}_{idx}", sub_val)
            return
        if self._is_valid_metric_value(value):
            target[f"{prefix}_{key}"] = float(value)

    def _is_probability_key(self, key: str) -> bool:
        lowered = key.lower()
        return lowered.startswith("p_") or "percentile" in lowered

    def _is_valid_metric_value(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return True
        if isinstance(value, Real):
            try:
                return math.isfinite(float(value))
            except (TypeError, ValueError):
                return False
        return False
