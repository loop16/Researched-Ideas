"""
Consolidated Live Feature Engine

This module provides a single output stream with one line per 5-minute bar
containing all the essential features needed for model adapters.

Key Features:
- Single consolidated output
- Resets at proper session boundaries
- One line per 5-minute bar
- All essential features in one place
"""

from __future__ import annotations
from datetime import datetime, timedelta, date, time
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, field

from .contracts import Bar
from ..constants import NY_TZ
from zoneinfo import ZoneInfo
from ..sessions import windows_for_day
from ..calendars import prev_business_day, next_business_day, trading_day_anchor, day_number_from_anchor

# Model labels (used for allowed set tracking during range formation)
ALL_MODELS: tuple[str, ...] = ("RC", "RX", "UX", "DX", "UXP", "DXP", "U", "D")


@dataclass
class SessionState:
    """State for any session (range or follow)."""
    session: str
    ny_day: date
    symbol: str
    
    # Range session features (AR/OR/RR)
    open0: Optional[float] = None
    closeN: Optional[float] = None
    dr_hi: Optional[float] = None
    dr_lo: Optional[float] = None
    idr_hi_close: Optional[float] = None
    idr_lo_close: Optional[float] = None
    box_std: Optional[float] = None
    # During range formation, track which models remain possible (monotone shrink)
    allowed_models: set = field(default_factory=lambda: set(ALL_MODELS))
    candle_idx: int = 0
    complete: bool = False
    final_model: Optional[str] = None
    
    # Follow session features (AS/OS/RS)
    confirmed: bool = False
    side: Optional[int] = None  # 1 for long, -1 for short
    confirm_time: Optional[datetime] = None
    confirm_candle_idx: Optional[int] = None  # Candle index when confirmation occurred
    false_detected: bool = False
    max_ext: float = 0.0
    max_ret: Optional[float] = None
    max_ext_time_idx: Optional[int] = None
    max_ret_time_idx: Optional[int] = None
    
    # Track highs and lows since confirmation (start after confirmation candle)
    highest_high: Optional[float] = None
    lowest_low: Optional[float] = None
    highest_high_time_idx: Optional[int] = None
    lowest_low_time_idx: Optional[int] = None
    
    # Session timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def is_range_session(self) -> bool:
        return self.session in ["AR", "OR", "RR"]
    
    def is_follow_session(self) -> bool:
        return self.session in ["AS", "OS", "RS"]
    
    def update_range_geometry(self, bar: Bar) -> None:
        """Update range geometry for range sessions."""
        if not self.is_range_session():
            return
        
        # First bar - capture open
        if self.open0 is None:
            self.open0 = bar.o
            self.start_time = datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ))
        
        # Update DR (true extremes from high/low)
        if self.dr_hi is None:
            self.dr_hi = bar.h
        else:
            self.dr_hi = max(self.dr_hi, bar.h)
            
        if self.dr_lo is None:
            self.dr_lo = bar.l
        else:
            self.dr_lo = min(self.dr_lo, bar.l)
        
        # Update IDR (close-to-close extremes)
        if self.idr_hi_close is None:
            self.idr_hi_close = bar.c
        else:
            self.idr_hi_close = max(self.idr_hi_close, bar.c)
            
        if self.idr_lo_close is None:
            self.idr_lo_close = bar.c
        else:
            self.idr_lo_close = min(self.idr_lo_close, bar.c)
        
        # Update final close
        self.closeN = bar.c
        
        # Calculate box_std
        if (self.idr_hi_close is not None and self.idr_lo_close is not None and 
            self.idr_hi_close > self.idr_lo_close and self.open0 is not None):
            
            idr_std = self.idr_hi_close - self.idr_lo_close
            if idr_std > 0:
                self.box_std = (self.closeN - self.open0) / idr_std
        
        # Update candle index
        self.candle_idx += 1

    
    
    
    def update_follow_session(self, bar: Bar, prior_range: 'SessionState') -> None:
        """Update follow session state."""
        if not self.is_follow_session() or not prior_range.is_range_session():
            return
        
        # Check for confirmation (first close-only break of prior DR, matching build_world_ultra_fast)
        # Allow confirmation once prior DR exists, not just when complete
        if not self.confirmed and (prior_range.dr_hi is not None and prior_range.dr_lo is not None):
            if bar.c > prior_range.dr_hi:
                self.confirmed = True
                self.side = 1  # Long
                self.confirm_time = datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ))
                self.confirm_candle_idx = self.candle_idx  # Store confirmation candle index
            elif bar.c < prior_range.dr_lo:
                self.confirmed = True
                self.side = -1  # Short
                self.confirm_time = datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ))
                self.confirm_candle_idx = self.candle_idx  # Store confirmation candle index
        
        # Check for false session (opposite close-only break after confirmation)
        if self.confirmed and not self.false_detected:
            if (self.side == 1 and prior_range.dr_lo is not None and bar.c < prior_range.dr_lo):
                self.false_detected = True
            elif (self.side == -1 and prior_range.dr_hi is not None and bar.c > prior_range.dr_hi):
                self.false_detected = True
        
        # Update extension/retracement tracking (start AFTER confirmation candle)
        if (self.confirmed and self.confirm_candle_idx is not None and 
            self.candle_idx > self.confirm_candle_idx and 
            prior_range.idr_hi_close is not None and prior_range.idr_lo_close is not None):
            
            idr_std = prior_range.idr_hi_close - prior_range.idr_lo_close
            if idr_std > 0:
                # Track highest high and lowest low since confirmation (after confirmation candle)
                if self.highest_high is None or bar.h > self.highest_high:
                    self.highest_high = bar.h
                    self.highest_high_time_idx = self.candle_idx
                
                if self.lowest_low is None or bar.l < self.lowest_low:
                    self.lowest_low = bar.l
                    self.lowest_low_time_idx = self.candle_idx
                
                # Calculate extension/retracement based on confirmation side
                if self.side == 1:  # Long confirmation
                    # Extension: highest high beyond IDR high
                    if self.highest_high is not None:
                        max_ext_std = (self.highest_high - prior_range.idr_hi_close) / idr_std
                        if max_ext_std > self.max_ext:
                            self.max_ext = max_ext_std
                            self.max_ext_time_idx = self.highest_high_time_idx
                    
                    # Retracement: lowest low back toward IDR low (should be negative)
                    if self.lowest_low is not None:
                        max_ret_std = (self.lowest_low - prior_range.idr_hi_close) / idr_std
                        if self.max_ret is None or max_ret_std < self.max_ret:
                            self.max_ret = max_ret_std
                            self.max_ret_time_idx = self.lowest_low_time_idx
                
                elif self.side == -1:  # Short confirmation
                    # Extension: lowest low beyond IDR low
                    if self.lowest_low is not None:
                        max_ext_std = (prior_range.idr_lo_close - self.lowest_low) / idr_std
                        if max_ext_std > self.max_ext:
                            self.max_ext = max_ext_std
                            self.max_ext_time_idx = self.lowest_low_time_idx
                    
                    # Retracement: highest high back toward IDR high (should be negative)
                    if self.highest_high is not None:
                        max_ret_std = (prior_range.idr_lo_close - self.highest_high) / idr_std
                        if self.max_ret is None or max_ret_std < self.max_ret:
                            self.max_ret = max_ret_std
                            self.max_ret_time_idx = self.highest_high_time_idx
        
        self.candle_idx += 1


@dataclass
class IntradayState:
    """State for intraday sequencing (04:00→15:55)."""
    ny_day: date
    symbol: str
    
    # Running day hi/lo since 04:00
    day_hi: Optional[float] = None
    day_lo: Optional[float] = None
    hi_time_idx: Optional[int] = None  # 5-minute index of day high
    lo_time_idx: Optional[int] = None  # 5-minute index of day low
    
    # Percentage vs ADR mid
    hi_pct: Optional[float] = None
    lo_pct: Optional[float] = None
    
    # Current tracking
    candle_idx: int = 0
    adr_mid: Optional[float] = None
    
    def update(self, bar: Bar, adr_mid: Optional[float] = None) -> None:
        """Update intraday sequencing with new bar."""
        # Update ADR mid if provided
        if adr_mid is not None:
            self.adr_mid = adr_mid
        
        # Update day high/low (use current candle_idx before incrementing)
        if self.day_hi is None:
            self.day_hi = bar.h
            self.hi_time_idx = self.candle_idx
        elif bar.h > self.day_hi:
            self.day_hi = bar.h
            self.hi_time_idx = self.candle_idx
        
        if self.day_lo is None:
            self.day_lo = bar.l
            self.lo_time_idx = self.candle_idx
        elif bar.l < self.day_lo:
            self.day_lo = bar.l
            self.lo_time_idx = self.candle_idx
        
        # Increment candle index after processing (starts at 0)
        self.candle_idx += 1
        
        # Calculate percentages vs ADR mid
        if self.adr_mid is not None and self.adr_mid > 0:
            if self.day_hi is not None:
                self.hi_pct = (self.day_hi - self.adr_mid) / self.adr_mid * 100
            if self.day_lo is not None:
                self.lo_pct = (self.day_lo - self.adr_mid) / self.adr_mid * 100


@dataclass
class WDRState:
    """State for WDR sequencing (09:30→09:25 next day)."""
    ny_day: date
    symbol: str
    
    # Running WDR hi/lo since 09:30
    wdr_hi: Optional[float] = None
    wdr_lo: Optional[float] = None
    wdr_hi_time_idx: Optional[int] = None  # 5-minute index of WDR high
    wdr_lo_time_idx: Optional[int] = None  # 5-minute index of WDR low
    
    # Percentage vs RR range IDR mid
    wdr_hi_pct: Optional[float] = None
    wdr_lo_pct: Optional[float] = None
    
    # Current tracking
    candle_idx: int = 0
    rr_idr_mid: Optional[float] = None
    
    def update(self, bar: Bar, rr_idr_mid: Optional[float] = None) -> None:
        """Update WDR sequencing with new bar."""
        # Update RR IDR mid if provided
        if rr_idr_mid is not None:
            self.rr_idr_mid = rr_idr_mid
        
        # Update WDR high/low (use current candle_idx before incrementing)
        if self.wdr_hi is None:
            self.wdr_hi = bar.h
            self.wdr_hi_time_idx = self.candle_idx
        elif bar.h > self.wdr_hi:
            self.wdr_hi = bar.h
            self.wdr_hi_time_idx = self.candle_idx
        
        if self.wdr_lo is None:
            self.wdr_lo = bar.l
            self.wdr_lo_time_idx = self.candle_idx
        elif bar.l < self.wdr_lo:
            self.wdr_lo = bar.l
            self.wdr_lo_time_idx = self.candle_idx
        
        # Increment candle index after processing (starts at 0)
        self.candle_idx += 1
        
        # Calculate percentages vs RR IDR mid
        if self.rr_idr_mid is not None and self.rr_idr_mid > 0:
            if self.wdr_hi is not None:
                self.wdr_hi_pct = (self.wdr_hi - self.rr_idr_mid) / self.rr_idr_mid * 100
            if self.wdr_lo is not None:
                self.wdr_lo_pct = (self.wdr_lo - self.rr_idr_mid) / self.rr_idr_mid * 100


@dataclass
class MidBreakState:
    """State for mid break tracking."""
    mid_type: str  # "ADR" or "ODR"
    ny_day: date
    symbol: str
    
    mid: Optional[float] = None
    broken: bool = False
    break_time: Optional[datetime] = None
    break_direction: Optional[str] = None  # "up" or "down"
    
    def update_mid(self, range_state: SessionState, fallback_mid: Optional[float] = None) -> None:
        """Update mid level from range state or fallback."""
        if range_state.dr_hi is not None and range_state.dr_lo is not None:
            self.mid = 0.5 * (range_state.dr_hi + range_state.dr_lo)
        elif fallback_mid is not None:
            self.mid = fallback_mid
    
    def check_break(self, bar: Bar) -> bool:
        """Check if mid level is broken."""
        if self.mid is None or self.broken:
            return False
        
        # Check for break through mid level
        if bar.o > self.mid and bar.l <= self.mid:  # Bearish break
            self.broken = True
            self.break_time = datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ))
            self.break_direction = "down"
            return True
        
        if bar.o < self.mid and bar.h >= self.mid:  # Bullish break
            self.broken = True
            self.break_time = datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ))
            self.break_direction = "up"
            return True
        
        return False


class ConsolidatedLiveEngine:
    """
    Consolidated Live Feature Engine
    
    Provides a single output stream with one line per 5-minute bar
    containing all essential features for model adapters.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
        # State storage keyed by (ny_day, symbol)
        self._state: Dict[Tuple[date, str], Dict[str, Any]] = {}
        
        # Session windows cache
        self._session_windows: Dict[date, Dict[str, Tuple[datetime, datetime]]] = {}
        
        # Load historical data for context
        self._load_historical_data()
    
    def _load_historical_data(self):
        """Load historical data files needed for context."""
        try:
            # Load box_daily_day for ADR mid fallback
            box_daily_path = self.data_dir / "processed" / "box_daily_day.parquet"
            if box_daily_path.exists():
                self._box_daily_df = pd.read_parquet(box_daily_path)
            else:
                csv_path = self.data_dir / "csv" / "box_daily_day.csv"
                if csv_path.exists():
                    self._box_daily_df = pd.read_csv(csv_path)
                else:
                    self._box_daily_df = pd.DataFrame()
        except Exception as e:
            print(f"Warning: Could not load box_daily_day data: {e}")
            self._box_daily_df = pd.DataFrame()
    
    def _ny_date_from_epoch(self, ts_epoch: int) -> date:
        """Convert epoch timestamp to NY trading date."""
        try:
            dt = datetime.fromtimestamp(ts_epoch, tz=ZoneInfo(NY_TZ))
            return dt.date()
        except Exception:
            return datetime.fromtimestamp(ts_epoch).date()
    
    def _get_session_windows(self, ny_day: date) -> Dict[str, Tuple[datetime, datetime]]:
        """Get session windows for a given day."""
        if ny_day not in self._session_windows:
            windows = windows_for_day(ny_day)
            self._session_windows[ny_day] = {
                w.session.value: (w.start, w.end) for w in windows
            }
        return self._session_windows[ny_day]
    
    def _get_state(self, ny_day: date, symbol: str) -> Dict[str, Any]:
        """Get or create state for a given day/symbol."""
        key = (ny_day, symbol)
        if key not in self._state:
            self._state[key] = {
                # Session states
                "AR": SessionState("AR", ny_day, symbol),
                "OR": SessionState("OR", ny_day, symbol),
                "RR": SessionState("RR", ny_day, symbol),
                "AS": SessionState("AS", ny_day, symbol),
                "OS": SessionState("OS", ny_day, symbol),
                "RS": SessionState("RS", ny_day, symbol),
                
                # Mid tracking
                "ADR_mid": MidBreakState("ADR", ny_day, symbol),
                "ODR_mid": MidBreakState("ODR", ny_day, symbol),
                
                # Intraday sequencing
                "intraday": IntradayState(ny_day, symbol),
                
                # WDR sequencing
                "wdr": WDRState(ny_day, symbol),
                
                # Current session
                "current_session": None,
                
                # Session-level range attributes (persist for entire session)
                "session_range_attrs": {
                    "AR": {"box_dir": "flat", "box_std": None, "model": "Undefined", "complete": False},
                    "OR": {"box_dir": "flat", "box_std": None, "model": "Undefined", "complete": False},
                    "RR": {"box_dir": "flat", "box_std": None, "model": "Undefined", "complete": False},
                },
            }
        return self._state[key]
    
    def _as_anchor_day(self, ny_day: date, ts_epoch: int, symbol: str) -> date:
        """Determine the anchor day for AS sessions that cross midnight.

        Prefer the immediate prior calendar day when we already have AS/AR
        progress recorded there (covers weekend gaps). Fallback to
        prev_business_day for the early-morning continuation edge case.
        """
        dt = datetime.fromtimestamp(ts_epoch, tz=ZoneInfo(NY_TZ))
        # AS runs ~20:30 → 01:55 next day; before/at 01:55 anchor to previous day
        if dt.hour < 2 or (dt.hour == 1 and dt.minute <= 55):
            prev_calendar_day = ny_day - timedelta(days=1)
            prev_state = self._state.get((prev_calendar_day, symbol))
            if prev_state:
                as_state = prev_state.get("AS")
                ar_state = prev_state.get("AR")
                if (as_state and (as_state.candle_idx > 0 or as_state.confirmed)) or (
                    ar_state and (ar_state.open0 is not None or ar_state.complete)
                ):
                    return prev_calendar_day
            return prev_business_day(ny_day)
        return ny_day
    
    def _get_persistent_state(self, bar: Bar, ny_day: date) -> Dict[str, Any]:
        """Get state that persists across midnight for multi-day sessions."""
        # Only AS crosses midnight; everything else can use the current day
        if bar.session == "AS":
            anchor = self._as_anchor_day(ny_day, bar.ts, bar.symbol)
            return self._get_state(anchor, bar.symbol)
        return self._get_state(ny_day, bar.symbol)
    
    def _get_prior_range(self, ny_day: date, symbol: str, current_session: str) -> Optional[SessionState]:
        """
        Get the prior range for the given current session.
        
        Prior range logic:
        - AR prior = RR (same ny_day)
        - OR prior = stitched AR (current ny_day)  ← stitched across midnight
        - RR prior = OR (same ny_day)
        """
        if current_session == "AR":
            st = self._get_state(ny_day, symbol)
            rr = st["RR"]
            if rr.open0 is None or not rr.complete:
                prev_day = prev_business_day(ny_day)
                prev_state = self._get_state(prev_day, symbol)
                return prev_state["RR"]
            return rr
        elif current_session == "OR":
            # OR's prior is the complete stitched AR for the current ny_day
            return self._stitch_ar_for_day(ny_day, symbol)
        elif current_session == "RR":
            st = self._get_state(ny_day, symbol)
            return st["OR"]
        
        return None

    def _stitch_ar_for_day(self, ny_day: date, symbol: str) -> SessionState:
        """
        Stitch AR across midnight for a given ny_day.
        
        AR spans midnight (≈20:30 → 01:55), so it's split across two calendar days.
        This helper combines the previous day's AR (evening part) with current day's AR (morning part).
        """
        prev_calendar_day = ny_day - timedelta(days=1)
        prev_ar_state = self._state.get((prev_calendar_day, symbol))
        if prev_ar_state and (prev_ar_state["AR"].open0 is not None or prev_ar_state["AR"].complete):
            prev_ar = prev_ar_state["AR"]
        else:
            prev_day = prev_business_day(ny_day)
            prev_ar = self._get_state(prev_day, symbol)["AR"]
        cur_ar = self._get_state(ny_day, symbol)["AR"]

        s = SessionState("AR", ny_day, symbol)

        # open/close across the whole AR (evening → 01:55)
        s.open0 = prev_ar.open0 if prev_ar.open0 is not None else cur_ar.open0
        s.closeN = cur_ar.closeN if cur_ar.closeN is not None else prev_ar.closeN

        # DR extremes
        dr_hi_candidates = [x for x in (prev_ar.dr_hi, cur_ar.dr_hi) if x is not None]
        dr_lo_candidates = [x for x in (prev_ar.dr_lo, cur_ar.dr_lo) if x is not None]
        s.dr_hi = max(dr_hi_candidates) if dr_hi_candidates else None
        s.dr_lo = min(dr_lo_candidates) if dr_lo_candidates else None

        # IDR close-to-close extremes
        idr_hi_candidates = [x for x in (prev_ar.idr_hi_close, cur_ar.idr_hi_close) if x is not None]
        idr_lo_candidates = [x for x in (prev_ar.idr_lo_close, cur_ar.idr_lo_close) if x is not None]
        s.idr_hi_close = max(idr_hi_candidates) if idr_hi_candidates else None
        s.idr_lo_close = min(idr_lo_candidates) if idr_lo_candidates else None

        # box_std across the stitched session
        if all(v is not None for v in (s.idr_hi_close, s.idr_lo_close, s.open0, s.closeN)) and s.idr_hi_close > s.idr_lo_close:
            idr_std = s.idr_hi_close - s.idr_lo_close
            s.box_std = (s.closeN - s.open0) / idr_std

        # mark "complete" when we have valid IDR
        s.complete = s.idr_hi_close is not None and s.idr_lo_close is not None
        return s

    def _finalize_range_if_needed(self, ny_day: date, symbol: str, code: str) -> None:
        """Finalize a range session (mark complete + set final_model)."""
        st = self._get_state(ny_day, symbol)
        rng = st[code]
        
        if not rng.is_range_session() or rng.complete:
            return
        
        # Get the correct prior range using the proper lookup method
        prior = self._get_prior_range(ny_day, symbol, code)
        
        if prior is None:
            rng.final_model = "Undefined"
        else:
            rng.complete = True
            rng.final_model = self._determine_model(rng, prior)
            # Once finalized, clamp allowed_models to the final model for a one-hot set
            if isinstance(rng.final_model, str) and rng.final_model in ALL_MODELS:
                rng.allowed_models = {rng.final_model}
        
        # Store range attributes at session level for persistence
        st["session_range_attrs"][code]["box_dir"] = self._get_box_direction(rng.open0, rng.closeN)
        st["session_range_attrs"][code]["box_std"] = rng.box_std
        st["session_range_attrs"][code]["model"] = rng.final_model
        st["session_range_attrs"][code]["complete"] = True

    def _update_allowed_models(self, ny_day: date, symbol: str, session: str) -> None:
        """Monotonically shrink the set of models still possible based on IDR close extremes vs prior range.

        Mirrors the legacy adapter logic but operates on SessionState objects using
        idr_lo_close/idr_hi_close fields.
        """
        if session not in ("AR", "OR", "RR"):
            return
        st = self._get_state(ny_day, symbol)
        cur = st.get(session)
        if not cur or cur.idr_lo_close is None or cur.idr_hi_close is None:
            return

        # Get prior for model classification
        prv = self._get_prior_range(ny_day, symbol, session)
        if prv is None or prv.idr_lo_close is None or prv.idr_hi_close is None:
            return

        cur_low = float(cur.idr_lo_close)
        cur_high = float(cur.idr_hi_close)
        prv_low = float(prv.idr_lo_close)
        prv_high = float(prv.idr_hi_close)
        prv_mid = (prv_low + prv_high) / 2.0

        allowed = set(cur.allowed_models) if isinstance(cur.allowed_models, set) else set(ALL_MODELS)

        # Monotone impossibility rules (once violated → permanently impossible)
        # RC needs cur_low >= prv_low and cur_high <= prv_high
        if cur_low < prv_low or cur_high > prv_high:
            allowed.discard("RC")
        # UXP needs cur_low >= prv_high
        if cur_low < prv_high:
            allowed.discard("UXP")
        # DXP needs cur_high <= prv_low
        if cur_high > prv_low:
            allowed.discard("DXP")
        # UX needs cur_low >= prv_mid
        if cur_low < prv_mid:
            allowed.discard("UX")
        # DX needs cur_high <= prv_mid
        if cur_high > prv_mid:
            allowed.discard("DX")
        # U needs cur_low >= prv_low
        if cur_low < prv_low:
            allowed.discard("U")
        # D needs cur_high <= prv_high
        if cur_high > prv_high:
            allowed.discard("D")
        # RX remains possible until completion

        cur.allowed_models = allowed
    
    def _fallback_asia_mid(self, ny_day: date, symbol: str) -> Optional[float]:
        """Get fallback ADR mid from box_daily_day data."""
        df = getattr(self, "_box_daily_df", pd.DataFrame())
        if df.empty:
            return None
        
        # Try current day first
        hit = df[(df["symbol"] == symbol) & (df["day_id"].astype(str) == str(ny_day))]
        if hit.empty:
            # Try previous business day
            prev_day = prev_business_day(ny_day)
            hit = df[(df["symbol"] == symbol) & (df["day_id"].astype(str) == str(prev_day))]
        
        if not hit.empty and pd.notna(hit.iloc[-1].get("mid_close")):
            return float(hit.iloc[-1]["mid_close"])
        return None
    
    def _detect_session_boundary(self, bar: Bar, ny_day: date, state: Dict[str, Any]) -> None:
        """Detect session boundary and reset if needed."""
        new_sess = bar.session
        prev = state["current_session"]
        
        # Finalize previous OR session when transitioning away from it
        # (OR sessions end at 03:55 but may not have a bar at exactly that time)
        if prev == "OR" and new_sess != "OR":
            self._finalize_range_if_needed(ny_day, bar.symbol, "OR")
        
        # Now init the new session, but don't wipe completed ones
        if state["current_session"] != new_sess:
            if new_sess in state:  # Only init if session exists in state
                if new_sess in ("AR", "OR", "RR"):
                    state[new_sess].__init__(new_sess, ny_day, bar.symbol)
                else:
                    state[new_sess].__init__(new_sess, ny_day, bar.symbol)
            state["current_session"] = new_sess
    
    def _get_prior_range_session(self, bar: Bar, ny_day: date, state: Dict[str, Any]) -> Optional[SessionState]:
        """
        Get the prior range session for a follow session.
        
        Follow session mappings:
        - AS follows AR (uses persistent state for midnight crossing)
        - OS follows OR (same ny_day)  
        - RS follows RR (same ny_day)
        """
        # Use persistent state for AS sessions that cross midnight
        if bar.session == "AS":
            persistent_state = self._get_persistent_state(bar, ny_day)
            return persistent_state["AR"]
        elif bar.session == "OS":  # OS follows OR (same day)
            return state["OR"]
        elif bar.session == "RS":  # RS follows RR (same day)
            return state["RR"]
        return None
    
    def _get_prior_session_name(self, current_session: str, ny_day: date, state: Dict[str, Any], bar: Optional[Bar] = None) -> str:
        """
        Get the prior session name for display in prior_session column.
        
        Prior session mappings:
        - RR and RS -> OS (they follow OS session)
        - OR and OS -> AS (they follow AS session)  
        - AR and AS -> RS (they follow RS session)
        """
        if current_session in ("RR", "RS"):
            return "OS"
        elif current_session in ("OR", "OS"):
            return "AS"
        elif current_session in ("AR", "AS"):
            return "RS"
        return ""
    
    def _get_prior_session_range_for_model(self, current_session: str, ny_day: date, state: Dict[str, Any], bar: Optional[Bar] = None) -> Optional[SessionState]:
        """
        Get the prior session's range for model lookup.
        
        Prior session range mappings:
        - AR session -> RR range (from previous day)
        - AS session -> RR range (from previous day, through AR)
        - OR session -> AR range (from previous day)
        - OS session -> AR range (from previous day)
        - RR session -> OR range (from same day)
        - RS session -> OR range (from same day)
        """
        if current_session == "AR":
            # AR needs RR range from previous day
            prev_day = prev_business_day(ny_day)
            prev_state = self._get_state(prev_day, state["AR"].symbol)
            return prev_state["RR"]
        elif current_session == "AS":
            # AS needs RR range from previous day, use persistent state for midnight crossing
            if bar:
                persistent_state = self._get_persistent_state(bar, ny_day)
                return persistent_state["RR"]
            else:
                # Fallback if no bar provided
                prev_day = prev_business_day(ny_day)
                prev_state = self._get_state(prev_day, state["AR"].symbol)
                return prev_state["RR"]
        elif current_session in ("OR", "OS"):
            # Both OR and OS need AR range from the most recent calendar day with data
            symbol = state["AR"].symbol
            prev_calendar_day = ny_day - timedelta(days=1)
            prev_state = self._state.get((prev_calendar_day, symbol))
            if prev_state and (prev_state["AR"].open0 is not None or prev_state["AR"].complete):
                return prev_state["AR"]
            prev_day = prev_business_day(ny_day)
            prev_state = self._get_state(prev_day, symbol)
            return prev_state["AR"]
        elif current_session in ("RR", "RS"):
            # Both RR and RS need OR range from same day
            return state["OR"]
        return None
    
    def _determine_model(self, current_range: SessionState, prior_range: SessionState) -> str:
        """Determine model from current and prior range."""
        if not current_range.is_range_session() or not prior_range.is_range_session():
            return "Undefined"
        
        cur_low = current_range.idr_lo_close
        cur_high = current_range.idr_hi_close
        prv_low = prior_range.idr_lo_close
        prv_high = prior_range.idr_hi_close
        
        if any(x is None for x in [cur_low, cur_high, prv_low, prv_high]):
            return "Undefined"
        
        prv_mid = (prv_high + prv_low) / 2.0
        
        if cur_low >= prv_low and cur_high <= prv_high:
            return "RC"
        elif cur_low <= prv_low and cur_high >= prv_high:
            return "RX"
        elif cur_low >= prv_mid and cur_low <= prv_high:
            return "UX"
        elif cur_high <= prv_mid and cur_high >= prv_low:
            return "DX"
        elif cur_low >= prv_high and cur_high >= prv_high:
            return "UXP"
        elif cur_low <= prv_low and cur_high <= prv_low:
            return "DXP"
        elif cur_low >= prv_low and cur_high > prv_mid:
            return "U"
        elif cur_high <= prv_high and cur_low < prv_mid:
            return "D"
        else:
            return "Undefined"
    
    def _get_box_direction(self, open0: Optional[float], closeN: Optional[float]) -> str:
        """Determine box direction from open and close."""
        if open0 is None or closeN is None:
            return "flat"
        
        if closeN > open0:
            return "up"
        elif closeN < open0:
            return "down"
        else:
            return "flat"
    
    def _is_within_intraday_window(self, bar: Bar) -> bool:
        """Check if bar is within intraday sequencing window (04:00→15:55)."""
        bar_dt = datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ))
        hour = bar_dt.hour
        minute = bar_dt.minute
        
        # 04:00 to 15:55
        return (hour >= 4) and (hour < 15 or (hour == 15 and minute <= 55))
    
    def _is_within_wdr_window(self, bar: Bar) -> bool:
        """Check if bar is within WDR sequencing window (09:30→09:25 next day)."""
        bar_dt = datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ))
        bar_time = bar_dt.time()
        start = time(9, 30)
        end = time(9, 25)
        
        # Window wraps past midnight: include bars at or after 09:30, or before/equal 09:25.
        return (bar_time >= start) or (bar_time <= end)
    
    def _is_within_adr_window(self, bar: Bar) -> bool:
        """Check if bar is within ADR tracking window (02:00→15:55)."""
        bar_dt = datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ))
        hour = bar_dt.hour
        minute = bar_dt.minute

        # 02:00 to 15:55
        return (hour > 2 or (hour == 2 and minute >= 0)) and (hour < 15 or (hour == 15 and minute <= 55))
    
    def _is_within_odr_window(self, bar: Bar) -> bool:
        """Check if bar is within ODR tracking window (08:30→15:55)."""
        bar_dt = datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ))
        hour = bar_dt.hour
        minute = bar_dt.minute
        
        # 08:30 to 15:55
        return (hour > 8 or (hour == 8 and minute >= 30)) and (hour < 15 or (hour == 15 and minute <= 55))
    
    def ingest_bar(self, bar: Bar) -> Dict[str, Any]:
        """
        Ingest a 5-minute bar and return consolidated feature output.
        
        Returns a single dictionary with all features for this bar.
        """
        ny_day = self._ny_date_from_epoch(bar.ts)
        
        # Garbage collect old state
        cutoff_day = ny_day - timedelta(days=8)
        keys_to_remove = [key for key in self._state.keys() if key[0] < cutoff_day]
        for key in keys_to_remove:
            del self._state[key]
        
        # Get state for this day/symbol
        state = self._get_state(ny_day, bar.symbol)
        
        # Detect session boundary and reset if needed
        self._detect_session_boundary(bar, ny_day, state)
        
        # Prepare session windows and timestamp
        wins = self._get_session_windows(ny_day)
        bar_dt = datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ))
        
        # Update current session state (only for range and follow sessions)
        current_session = bar.session
        current_state = None
        
        # Use persistent state for AS sessions that cross midnight
        if current_session == "AS":
            persistent_state = self._get_persistent_state(bar, ny_day)
            current_state = persistent_state[current_session]
        else:
            current_state = state.get(current_session)
        
        if current_state:
            if current_state.is_range_session():
                # First update range geometry with this bar
                current_state.update_range_geometry(bar)
                # Update allowed models as we form the range (monotone shrink)
                self._update_allowed_models(ny_day, bar.symbol, current_session)
                # Then finalize by time if the bar is at or after the session end.
                if bar.session in ("AR", "OR", "RR") and bar.session in wins:
                    _, end_ts = wins[bar.session]
                    if bar_dt >= end_ts:
                        self._finalize_range_if_needed(ny_day, bar.symbol, bar.session)
            elif current_state.is_follow_session():
                # Get prior range for follow session
                prior_range = self._get_prior_range_session(bar, ny_day, state)
                if prior_range:
                    current_state.update_follow_session(bar, prior_range)
        
        # Update mid break tracking
        if self._is_within_adr_window(bar):
            # Use stitched AR for the current ny_day (not prev day only)
            stitched_ar = self._stitch_ar_for_day(ny_day, bar.symbol)
            fallback_mid = self._fallback_asia_mid(ny_day, bar.symbol)
            state["ADR_mid"].update_mid(stitched_ar, fallback_mid=fallback_mid)
            state["ADR_mid"].check_break(bar)
        
        if self._is_within_odr_window(bar):
            # Get ODR mid from current day's OR
            state["ODR_mid"].update_mid(state["OR"])
            state["ODR_mid"].check_break(bar)
        
        # Update intraday sequencing (04:00→15:55)
        if self._is_within_intraday_window(bar):
            # Get ADR mid for percentage calculations
            adr_mid = None
            if state["ADR_mid"].mid is not None:
                adr_mid = state["ADR_mid"].mid
            state["intraday"].update(bar, adr_mid)
        
        # Update WDR sequencing (09:30→09:25 next day)
        bar_dt = datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ))
        hour = bar_dt.hour
        minute = bar_dt.minute
        bar_time = bar_dt.time()
        
        # Check if this is the start of a new WDR cycle (09:30)
        is_wdr_start = hour == 9 and minute == 30
        is_wdr_window = self._is_within_wdr_window(bar)
        
        if is_wdr_start:
            # Always start a fresh WDR state at 09:30
            state["wdr"] = WDRState(ny_day, bar.symbol)
        elif is_wdr_window and state["wdr"].wdr_hi is None:
            # Carry prior WDR progress forward whenever we enter the window with no local state yet.
            # This covers both the early-morning continuation (00:00→09:25) and multi-day gaps
            # such as the Friday→Sunday weekend where no bars arrive until the new week opens.
            prev_day = prev_business_day(ny_day)
            prev_state = self._get_state(prev_day, bar.symbol)
            prev_wdr = prev_state.get("wdr")
            if prev_wdr and prev_wdr.wdr_hi is not None:
                state["wdr"] = prev_wdr
        
        # Update WDR if within window
        if is_wdr_window:
            # Get RR IDR mid for percentage calculations
            rr_idr_mid = None
            if state["RR"].idr_hi_close is not None and state["RR"].idr_lo_close is not None:
                rr_idr_mid = (state["RR"].idr_hi_close + state["RR"].idr_lo_close) / 2
            state["wdr"].update(bar, rr_idr_mid)
        
        # Build consolidated output
        return self._build_consolidated_output(bar, ny_day, state, current_session, current_state)
    
    def _build_consolidated_output(self, bar: Bar, ny_day: date, state: Dict[str, Any], 
                                 current_session: str, current_state: SessionState) -> Dict[str, Any]:
        """Build the consolidated output with all required features."""
        
        bar_dt = datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ))
        trading_anchor = trading_day_anchor(bar_dt)
        trading_day_number = day_number_from_anchor(trading_anchor)

        # Get prior range for current session (for follow sessions)
        prior_range = self._get_prior_range_session(bar, ny_day, state)
        
        # Get prior session range for model lookup (different logic)
        prior_session_range = self._get_prior_session_range_for_model(current_session, ny_day, state, bar)
        
        # Prior session features - use session-level storage for persistence
        prior_session_model = "Undefined"
        prior_session_box_dir = "flat"
        prior_session_break_out = False
        prior_session_true = False
        prior_session_break_out_direction = None
        
        if prior_session_range and prior_session_range.is_range_session():
            prior_session_code = prior_session_range.session
            
            # Determine which state to use for looking up prior session attributes
            if current_session == "AR":
                # AR prefers the current day's RR if complete; fallback to prior business day
                if state["session_range_attrs"]["RR"]["complete"]:
                    lookup_state = state
                else:
                    prev_day = prev_business_day(ny_day)
                    lookup_state = self._get_state(prev_day, state["AR"].symbol)
            elif current_session == "AS":
                # AS uses the same anchor as AR (current day when RR is complete, otherwise prior business day)
                persistent_state = self._get_persistent_state(bar, ny_day)
                if persistent_state["session_range_attrs"]["RR"]["complete"]:
                    lookup_state = persistent_state
                else:
                    prev_day = prev_business_day(ny_day)
                    lookup_state = self._get_state(prev_day, state["AR"].symbol)
            elif current_session in ("OR", "OS"):
                # OR and OS prefer the most recent calendar day's AR; fallback to prior business day.
                symbol = state["AR"].symbol
                prev_calendar_day = ny_day - timedelta(days=1)
                prev_state = self._state.get((prev_calendar_day, symbol))
                if prev_state and (prev_state["AR"].open0 is not None or prev_state["AR"].complete):
                    lookup_state = prev_state
                else:
                    prev_day = prev_business_day(ny_day)
                    lookup_state = self._get_state(prev_day, symbol)
            elif current_session in ("RR", "RS"):
                # RR and RS need OR range from same day
                lookup_state = state
            else:
                # Default to current day's state
                lookup_state = state
            prior_session_lookup_state = lookup_state
            
            # Check if we have session-level attributes for this prior range
            attrs = lookup_state["session_range_attrs"].get(prior_session_code)
            if attrs:
                prior_session_model = attrs["model"]
                prior_session_box_dir = attrs["box_dir"]
                prior_session_break_out = attrs["complete"]
                prior_session_true = attrs["complete"]
            else:
                # Fallback to current range state if session attributes not available
                if prior_session_range.final_model:
                    prior_session_model = prior_session_range.final_model
                prior_session_box_dir = self._get_box_direction(prior_session_range.open0, prior_session_range.closeN)
                prior_session_break_out = prior_session_range.complete
                prior_session_true = prior_session_range.complete
                attrs = None
            
            # Weekend gap fallback: if primary lookup has no completion data, try previous business day.
            if (not prior_session_break_out) and bar is not None:
                fallback_day = prev_business_day(ny_day)
                if fallback_day != ny_day:
                    fallback_state = self._get_state(fallback_day, prior_session_range.symbol)
                    fallback_attrs = fallback_state["session_range_attrs"].get(prior_session_code)
                    if fallback_attrs and fallback_attrs.get("complete"):
                        prior_session_model = fallback_attrs["model"]
                        prior_session_box_dir = fallback_attrs["box_dir"]
                        prior_session_break_out = fallback_attrs["complete"]
                        prior_session_true = fallback_attrs["complete"]
                        attrs = fallback_attrs
                        prior_session_lookup_state = fallback_state
            
            # Determine prior session break out direction and truth based on which follow session confirmed it
            if prior_session_break_out:
                if prior_session_code == "RR":
                    # RR is confirmed by RS session - use appropriate state based on current session
                    if current_session == "AR":
                        # Prefer the same lookup used for attributes; fall back to prior business day if needed
                        rs_lookup_state = prior_session_lookup_state
                        if ("RS" not in rs_lookup_state or not rs_lookup_state["RS"].confirmed):
                            prev_day = prev_business_day(ny_day)
                            rs_lookup_state = self._get_state(prev_day, state["AR"].symbol)
                    elif current_session == "AS":
                        # For AS sessions, prefer the lookup used for prior-session attrs; fall back to previous business day
                        rs_lookup_state = prior_session_lookup_state
                        if ("RS" not in rs_lookup_state or not rs_lookup_state["RS"].confirmed) and bar is not None:
                            prev_day = prev_business_day(ny_day)
                            rs_lookup_state = self._get_state(prev_day, prior_session_range.symbol)
                    else:
                        # For RR/RS sessions, RS is from same day
                        rs_lookup_state = lookup_state
                    
                    if "RS" in rs_lookup_state and rs_lookup_state["RS"].confirmed:
                        rs_state = rs_lookup_state["RS"]
                        prior_session_break_out_direction = "long" if rs_state.side == 1 else "short"
                        # Propagate truth: confirmed and not false
                        prior_session_true = bool(rs_state.confirmed and not rs_state.false_detected)
                elif prior_session_code == "AR":
                    # AR is confirmed by AS session - use appropriate state based on current session
                    if current_session in ("OR", "OS"):
                        # For OR/OS sessions, AS is from previous day
                        prev_day = prev_business_day(ny_day)
                        as_lookup_state = self._get_state(prev_day, state["AR"].symbol)
                    else:
                        # For AS sessions, use persistent state for midnight crossing
                        as_lookup_state = self._get_persistent_state(bar, ny_day)
                    
                    if "AS" in as_lookup_state and as_lookup_state["AS"].confirmed:
                        as_state = as_lookup_state["AS"]
                        prior_session_break_out_direction = "long" if as_state.side == 1 else "short"
                        prior_session_true = bool(as_state.confirmed and not as_state.false_detected)
                elif prior_session_code == "OR":
                    # OR is confirmed by OS session (same day)
                    if "OS" in lookup_state and lookup_state["OS"].confirmed:
                        os_state = lookup_state["OS"]
                        prior_session_break_out_direction = "long" if os_state.side == 1 else "short"
                        prior_session_true = bool(os_state.confirmed and not os_state.false_detected)
        
        # Current session features
        current_session_model = "Undefined"
        current_session_box_dir = "flat"
        current_session_box_std = None
        current_session_break_out = False
        current_session_true = False
        current_session_max_ext = 0.0
        current_session_max_ret = None
        current_session_time_idx_retracement = None
        current_session_time_idx_extension = None
        current_session_break_out_idx = None
        current_session_break_out_direction = None
        
        if current_state:
            if current_state.is_range_session():
                # Compute current_session_model live for range sessions
                prior_for_model = self._get_prior_range(ny_day, bar.symbol, current_session)
                if prior_for_model is not None:
                    live_model = self._determine_model(current_state, prior_for_model)
                else:
                    live_model = "Undefined"

                # If the persisted attrs are not complete yet, prefer the live model
                attrs = state["session_range_attrs"].get(current_session, {})
                if attrs and attrs.get("complete"):
                    current_session_model = attrs.get("model", live_model)
                else:
                    current_session_model = live_model

                current_session_box_dir = self._get_box_direction(current_state.open0, current_state.closeN)
                current_session_box_std = current_state.box_std
                current_session_break_out = bool(attrs.get("complete")) if attrs else current_state.complete
                current_session_true = current_session_break_out
            
            elif current_state.is_follow_session():
                # For follow sessions, inherit current session model and box_std from their corresponding range session
                if current_session == "AS":
                    # AS follows AR - use persistent state for midnight crossing
                    persistent_state = self._get_persistent_state(bar, ny_day)
                    ar_attrs = persistent_state["session_range_attrs"].get("AR", {})
                    current_session_model = ar_attrs.get("model", "Undefined")
                    current_session_box_dir = ar_attrs.get("box_dir", "flat")
                    current_session_box_std = ar_attrs.get("box_std", None)
                    
                    cur_follow = persistent_state[current_session]
                    current_session_break_out = cur_follow.confirmed
                    current_session_true = cur_follow.confirmed and not cur_follow.false_detected
                    current_session_break_out_direction = ("long" if cur_follow.side == 1 else "short") if cur_follow.side in (1, -1) else None
                    current_session_max_ext = cur_follow.max_ext
                    current_session_max_ret = cur_follow.max_ret
                    current_session_time_idx_retracement = cur_follow.max_ret_time_idx
                    current_session_time_idx_extension = cur_follow.max_ext_time_idx
                    current_session_break_out_idx = cur_follow.confirm_candle_idx
                elif current_session == "OS":
                    # OS follows OR - inherit from OR range session
                    or_attrs = state["session_range_attrs"].get("OR", {})
                    current_session_model = or_attrs.get("model", "Undefined")
                    current_session_box_dir = or_attrs.get("box_dir", "flat")
                    current_session_box_std = or_attrs.get("box_std", None)
                    
                    current_session_break_out = current_state.confirmed
                    current_session_true = current_state.confirmed and not current_state.false_detected
                    current_session_break_out_direction = ("long" if current_state.side == 1 else "short") if current_state.side in (1, -1) else None
                    current_session_max_ext = current_state.max_ext
                    current_session_max_ret = current_state.max_ret
                    current_session_time_idx_retracement = current_state.max_ret_time_idx
                    current_session_time_idx_extension = current_state.max_ext_time_idx
                    current_session_break_out_idx = current_state.confirm_candle_idx
                elif current_session == "RS":
                    # RS follows RR - inherit from RR range session
                    rr_attrs = state["session_range_attrs"].get("RR", {})
                    current_session_model = rr_attrs.get("model", "Undefined")
                    current_session_box_dir = rr_attrs.get("box_dir", "flat")
                    current_session_box_std = rr_attrs.get("box_std", None)
                    
                    current_session_break_out = current_state.confirmed
                    current_session_true = current_state.confirmed and not current_state.false_detected
                    current_session_break_out_direction = ("long" if current_state.side == 1 else "short") if current_state.side in (1, -1) else None
                    current_session_max_ext = current_state.max_ext
                    current_session_max_ret = current_state.max_ret
                    current_session_time_idx_retracement = current_state.max_ret_time_idx
                    current_session_time_idx_extension = current_state.max_ext_time_idx
                    current_session_break_out_idx = current_state.confirm_candle_idx
        
        # Intraday features (04:00→15:55) - only show values within window
        intraday = state["intraday"]
        if self._is_within_intraday_window(bar):
            high_time_idx = intraday.hi_time_idx
            low_time_idx = intraday.lo_time_idx
            high_pct = intraday.hi_pct
            low_pct = intraday.lo_pct
        else:
            # Leave blank outside intraday window (15:55→04:00)
            high_time_idx = None
            low_time_idx = None
            high_pct = None
            low_pct = None
        
        # WDR sequencing features (09:30→09:25 next day)
        wdr = state["wdr"]
        wdr_high_time_idx = wdr.wdr_hi_time_idx
        wdr_low_time_idx = wdr.wdr_lo_time_idx
        wdr_high_pct = wdr.wdr_hi_pct
        wdr_low_pct = wdr.wdr_lo_pct
        
        # Current time indices (subtract 1 because candle_idx is incremented after processing)
        current_time_idx = (intraday.candle_idx - 1) if self._is_within_intraday_window(bar) else None
        current_wdr_time_idx = (wdr.candle_idx - 1) if self._is_within_wdr_window(bar) else None
        
        # Mid break features
        adr_mid_break = state["ADR_mid"].broken
        odr_mid_break = state["ODR_mid"].broken
        
        # Build consolidated output
        output = {
            # Bar info
            "timestamp": datetime.fromtimestamp(bar.ts, tz=ZoneInfo(NY_TZ)),
            "ts": bar.ts,
            "symbol": bar.symbol,
            "day_number": trading_day_number,
            "session": bar.session,
            "ny_day": ny_day,
            "open": bar.o,
            "high": bar.h,
            "low": bar.l,
            "close": bar.c,
            
            # Prior session features
            "prior_session": self._get_prior_session_name(current_session, ny_day, state, bar),
            "prior_session_model": prior_session_model,
            "prior_session_box_dir": prior_session_box_dir,
            "prior_session_break_out": prior_session_break_out,
            "prior_session_true": prior_session_true,
            "prior_session_break_out_direction": prior_session_break_out_direction,
            
            # Current session features
            "current_session": current_session,
            "current_session_model": current_session_model,
            "current_session_allowed_models": sorted(list(current_state.allowed_models)) if current_state and hasattr(current_state, "allowed_models") else None,
            "current_session_box_dir": current_session_box_dir,
            "current_session_box_std": current_session_box_std,
            "current_session_break_out": current_session_break_out,
            "current_session_true": current_session_true,
            "current_session_break_out_direction": current_session_break_out_direction,
            "current_session_max_ext": current_session_max_ext,
            "current_session_max_ret": current_session_max_ret,
            "current_session_time_idx_retracement": current_session_time_idx_retracement,
            "current_session_time_idx_extension": current_session_time_idx_extension,
            "current_session_break_out_idx": current_session_break_out_idx,
            
            # Intraday features (04:00→15:55)
            "high_time_idx": high_time_idx,
            "low_time_idx": low_time_idx,
            "high_pct": high_pct,
            "low_pct": low_pct,
            "current_time_idx": current_time_idx,
            
            # WDR features (09:30→09:25 next day)
            "wdr_high_time_idx": wdr_high_time_idx,
            "wdr_low_time_idx": wdr_low_time_idx,
            "wdr_high_pct": wdr_high_pct,
            "wdr_low_pct": wdr_low_pct,
            "current_wdr_time_idx": current_wdr_time_idx,
            
            # Mid break features
            "adr_mid_break": adr_mid_break,
            "odr_mid_break": odr_mid_break,
        }
        
        return output
