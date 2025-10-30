from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

from .contracts import (
    Bar,
    RangeSessionOut,
    SeqIntradayOut,
    ModelDBOut,
)
from .consolidated_live_engine import ConsolidatedLiveEngine
from ..constants import NY_TZ


ALL_MODELS: tuple[str, ...] = ("RC", "RX", "UX", "DX", "UXP", "DXP", "U", "D")

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
        self.MIN_SEQ_ROWS = 10

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
        """Load model_db history for a symbol, preferring processed/<SYM>/<SYM>_model_db.parquet."""
        if symbol in self._modeldb_cache:
            return self._modeldb_cache[symbol]
        from pathlib import Path
        base = Path(self.data_dir)
        # Parquet first
        p_main = base / "processed" / symbol / f"{symbol}_model_db.parquet"
        try:
            if p_main.exists():
                df = pd.read_parquet(p_main)
                self._modeldb_cache[symbol] = df
                return df
        except Exception:
            pass
        # Any parquet candidate
        try:
            for f in (base / "processed" / symbol).glob("*_model_db.parquet"):
                try:
                    df = pd.read_parquet(f)
                    self._modeldb_cache[symbol] = df
                    return df
                except Exception:
                    continue
        except Exception:
            pass
        # CSV fallback
        try:
            c_main = base / "csv" / symbol / f"{symbol}_model_db.csv"
            if c_main.exists():
                df = pd.read_csv(c_main)
                self._modeldb_cache[symbol] = df
                return df
        except Exception:
            pass
        self._modeldb_cache[symbol] = pd.DataFrame()
        return self._modeldb_cache[symbol]

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

    def _seq_filters_for_now(self, bar: Bar, ny_day: date) -> dict:
        st = self.engine._get_state(ny_day, bar.symbol)
        f: dict[str, object] = {}

        or_state = st.get("OR")
        if or_state and getattr(or_state, "final_model", None):
            f["ODR_Model"] = or_state.final_model
        if or_state and getattr(or_state, "box_std", None) is not None:
            bs = float(or_state.box_std)
            f["ODR_Box_Direction"] = "up" if bs > 0 else "down" if bs < 0 else "flat"

        os_state = st.get("OS")
        if os_state and getattr(os_state, "confirmed", False):
            side = os_state.side
            if side in (1, -1):
                dir_txt = "long" if side == 1 else "short"
                f["ODR_Trade_Direction"] = dir_txt
                f["os_breakout_dir"] = dir_txt
        if os_state and (os_state.confirmed or os_state.false_detected is not None):
            f["OS_Session_True"] = bool(os_state.confirmed and not os_state.false_detected)

        rr_state = st.get("RR")
        if rr_state and getattr(rr_state, "final_model", None):
            f["RDR_Model"] = rr_state.final_model
        if rr_state and getattr(rr_state, "box_std", None) is not None:
            bs = float(rr_state.box_std)
            f["RDR_Box_Direction"] = "up" if bs > 0 else "down" if bs < 0 else "flat"

        adr_state = st.get("ADR_mid")
        if adr_state and getattr(adr_state, "broken", False):
            f["ADR_Mid_Broken"] = True

        return f

    def _seq_relax_order(self) -> list[str]:
        return [
            "ADR_Mid_Broken",
            "OS_Session_True",
            "os_breakout_dir",
            "ODR_Trade_Direction",
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
            "ODR_Trade_Direction": ["OS_confirm_dir"],
            "os_breakout_dir": ["OS_confirm_dir"],
            "OS_Session_True": ["OS_stays_true", "OS_Session_True"],
            "RDR_Model": ["RDR_Model", "NextRDR_Model"],
            "ADR_Mid_Broken": ["asia_mid_broken"],
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
        if side == "low":
            base[pct_col] = base[pct_col].abs()
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
        if side == "low":
            df[pct_col] = df[pct_col].abs()

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
        out = self.ingest_bar(bar)

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
    # ModelDB live (compact v3 mapping)
    # ──────────────────────────────────────────────────────────────
    def get_modeldb_live_v3(self, bar: Bar) -> Dict[str, float]:
        """Produce a compact live probability dict using consolidated fields.

        Keys mirror v2 for drop-in compatibility where possible.
        """
        out = self.ingest_bar(bar)

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
        out = self.ingest_bar(bar)

        # Outside 04:00→15:55 → neutral output
        if not self._within_intraday_window(bar.ts):
            return SeqIntradayOut()

        seq_hist = self._get_seq_intraday_for_symbol(bar.symbol)
        if seq_hist.empty:
            return SeqIntradayOut()

        ny_day = self.engine._ny_date_from_epoch(bar.ts)

        filters = self._seq_filters_for_now(bar, ny_day)
        mapped_filters, relax_cols = self._map_seq_filter_keys(filters, seq_hist)
        sub = self._apply_filters_with_backoff(seq_hist, mapped_filters, relax_cols, self.MIN_SEQ_ROWS)
        if len(sub) < self.MIN_SEQ_ROWS and "OR_Box_Direction" in mapped_filters:
            # Relax ODR box direction explicitly when we don't have enough history
            mf2 = dict(mapped_filters)
            mf2.pop("OR_Box_Direction", None)
            relax2 = [c for c in relax_cols if c != "OR_Box_Direction"]
            sub = self._apply_filters_with_backoff(seq_hist, mf2, relax2, self.MIN_SEQ_ROWS)

        if sub.empty:
            sub = seq_hist[seq_hist.get("symbol") == bar.symbol]
            hi_col, lo_col = self._seq_where_cols(sub)
            if hi_col and allowed_high:
                sub = sub[sub[hi_col].astype(str).str.upper().isin(allowed_high)]
            if lo_col and allowed_low:
                sub = sub[sub[lo_col].astype(str).str.upper().isin(allowed_low)]

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

        allowed_high = self._allowed_sessions(ny_day, t_bucket_5m, hi_idx)
        allowed_low = self._allowed_sessions(ny_day, t_bucket_5m, lo_idx)

        hi_col, lo_col = self._seq_where_cols(sub)
        if hi_col and allowed_high:
            sub = sub[sub[hi_col].astype(str).str.upper().isin(allowed_high)]
        if lo_col and allowed_low:
            sub = sub[sub[lo_col].astype(str).str.upper().isin(allowed_low)]
        if sub.empty:
            sub = seq_hist[seq_hist.get("symbol") == bar.symbol]

        pH = self._seq_location_probs(sub, "high", allowed_high)
        pL = self._seq_location_probs(sub, "low", allowed_low)
        pH = self._zero_impossible_sessions(pH, hi_idx, ny_day, "high", t_bucket_5m)
        pL = self._zero_impossible_sessions(pL, lo_idx, ny_day, "low", t_bucket_5m)

        pct_hi_col, _ = self._seq_cols(sub, "high")
        pct_lo_col, _ = self._seq_cols(sub, "low")
        hi_series = pd.to_numeric(sub[pct_hi_col], errors="coerce") if pct_hi_col else None
        lo_series_raw = pd.to_numeric(sub[pct_lo_col], errors="coerce") if pct_lo_col else None
        q_hi = self._quantiles(hi_series, [0.20, 0.50, 0.80, 0.15])
        q_lo = self._quantiles(lo_series_raw, [0.20, 0.50, 0.80, 0.85])

        trend_guard_hi = bool(hi_now is not None and q_hi[0] is not None and hi_now >= q_hi[0])
        trend_guard_lo = bool(lo_now is not None and q_lo[2] is not None and lo_now <= q_lo[2])
        failed_20_hi = bool(hi_now is not None and q_hi[0] is not None and hi_now < q_hi[0])
        failed_20_lo = bool(lo_now is not None and q_lo[0] is not None and lo_now > q_lo[0])

        def _percentile(series, value):
            if series is None or value is None:
                return None
            s = pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                return None
            return float((s <= value).mean())

        hi_pct_percentile = _percentile(hi_series, hi_now)
        lo_pct_percentile = _percentile(lo_series_raw, lo_now)

        lo_abs_now = abs(lo_now) if lo_now is not None else None
        p_high_hold_b = self._hold_prob_bucketed(sub, t_bucket_30m, hi_now, "high") if hi_now is not None else 0.5
        p_low_hold_b = self._hold_prob_bucketed(sub, t_bucket_30m, lo_abs_now, "low") if lo_abs_now is not None else 0.5
        p_high_hold_r = self._hold_prob_resid(sub, t_bucket_30m, hi_now, "high") if hi_now is not None else p_high_hold_b
        p_low_hold_r = self._hold_prob_resid(sub, t_bucket_30m, lo_abs_now, "low") if lo_abs_now is not None else p_low_hold_b

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
            q_hi_20=q_hi[0], q_hi_50=q_hi[1], q_hi_80=q_hi[2], q_hi_15=q_hi[3],
            q_lo_20=q_lo[0], q_lo_50=q_lo[1], q_lo_80=q_lo[2], q_lo_85=q_lo[3],
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
