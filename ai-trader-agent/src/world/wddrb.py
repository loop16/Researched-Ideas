# src/world/wddrb.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, date, time
from zoneinfo import ZoneInfo
from typing import Optional, List, Tuple, Dict
import pandas as pd
import numpy as np
from pathlib import Path

NY_TZ = ZoneInfo("America/New_York")

# ---- Session labels (match your naming)
class Session:
    REGULAR_RANGE = "REGULAR_RANGE"
    REGULAR_SESSION = "REGULAR_SESSION"
    TRANSITION_OUT = "TRANSITION_OUT"
    TRANSITION_ASIA = "TRANSITION_ASIA"
    ASIA_RANGE = "ASIA_RANGE"
    ASIA_SESSION = "ASIA_SESSION"
    TRANSITION_OVERNIGHT = "TRANSITION_OVERNIGHT"
    OVERNIGHT_RANGE = "OVERNIGHT_RANGE"
    OVERNIGHT_SESSION = "OVERNIGHT_SESSION"
    TRANSITION_REGULAR = "TRANSITION_REGULAR"

SESSION_SHORT = {
    Session.TRANSITION_ASIA: "TA",
    Session.ASIA_RANGE: "AR",
    Session.ASIA_SESSION: "AS",
    Session.TRANSITION_OVERNIGHT: "TO",
    Session.OVERNIGHT_RANGE: "OR",
    Session.OVERNIGHT_SESSION: "OS",
    Session.TRANSITION_REGULAR: "TR",
    Session.REGULAR_RANGE: "RR",
    Session.REGULAR_SESSION: "RS",
    Session.TRANSITION_OUT: "TOUT",
}

POST_SESSION_SEQUENCE = [
    Session.TRANSITION_OUT,
    Session.TRANSITION_ASIA,
    Session.ASIA_RANGE,
    Session.ASIA_SESSION,
    Session.TRANSITION_OVERNIGHT,
    Session.OVERNIGHT_RANGE,
    Session.OVERNIGHT_SESSION,
    Session.TRANSITION_REGULAR,
]

def _format_intraday_time(ts: Optional[pd.Timestamp]) -> Optional[str]:
    """Return HH:MM (NY) for a timestamp; drop the date component."""
    if ts is None or pd.isna(ts):
        return None
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(NY_TZ)
    else:
        t = t.tz_convert(NY_TZ)
    return t.strftime("%H:%M")

def _minutes_since_reg_close(ts: Optional[pd.Timestamp],
                             reg_end: Optional[pd.Timestamp],
                             wins: List[Tuple[str, datetime, datetime]]) -> float:
    if ts is None or reg_end is None or pd.isna(ts):
        return np.nan
    t = pd.Timestamp(ts)
    t = t.tz_localize(NY_TZ) if t.tzinfo is None else t.tz_convert(NY_TZ)
    ref = pd.Timestamp(reg_end).tz_localize(NY_TZ) if pd.Timestamp(reg_end).tzinfo is None else pd.Timestamp(reg_end).tz_convert(NY_TZ)
    if t <= ref:
        return 0.0
    minutes = 0.0
    win_map = {s: (pd.Timestamp(start).tz_localize(NY_TZ) if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert(NY_TZ),
                   pd.Timestamp(end).tz_localize(NY_TZ) if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert(NY_TZ))
               for s, start, end in wins}
    for sess in POST_SESSION_SEQUENCE:
        if sess not in win_map:
            return np.nan
        start, end = win_map[sess]
        if t < start:
            return minutes
        if start <= t <= end:
            return minutes + max(0.0, (t - start).total_seconds() / 60.0)
        minutes += (end - start).total_seconds() / 60.0
    return minutes

def _post_window_total_minutes(wins: List[Tuple[str, datetime, datetime]]) -> float:
    win_map = {s: (start, end) for s, start, end in wins}
    total = 0.0
    for sess in POST_SESSION_SEQUENCE:
        if sess not in win_map:
            return np.nan
        start, end = win_map[sess]
        total += (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 60.0
    return total

def _dt(d: date, hh: int, mm: int) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=NY_TZ)

def _windows_for_day(anchor_date: date) -> List[Tuple[str, datetime, datetime]]:
    """
    Build your exact windows with 'day 1 is Tuesday' anchor logic.
    'anchor_date' is the 'day date' (Tuesday for D1, etc.) in NY time.
    """
    wd = anchor_date.weekday()  # Mon=0..Sun=6
    d0 = anchor_date
    d1 = anchor_date + timedelta(days=1)
    d_sun = anchor_date + timedelta(days=(6 - wd))        # next Sunday
    d_mon = anchor_date + timedelta(days=((7 - wd) % 7))  # next Monday

    windows: List[Tuple[str, datetime, datetime]] = []
    # Daytime on anchor_date
    windows += [
        (Session.REGULAR_RANGE,   _dt(d0, 9,30),  _dt(d0,10,25)),
        (Session.REGULAR_SESSION, _dt(d0,10,30),  _dt(d0,15,55)),
        (Session.TRANSITION_OUT,  _dt(d0,16, 0),  _dt(d0,16,55)),
    ]
    if wd == 4:  # Friday → move evening to Sun/Mon
        windows += [
            (Session.TRANSITION_ASIA,   _dt(d_sun,18, 0), _dt(d_sun,19,25)),
            (Session.ASIA_RANGE,        _dt(d_sun,19,30), _dt(d_sun,20,25)),
            (Session.ASIA_SESSION,      _dt(d_sun,20,30), _dt(d_mon, 1,55)),
            (Session.TRANSITION_OVERNIGHT,_dt(d_mon, 2, 0), _dt(d_mon, 2,55)),
            (Session.OVERNIGHT_RANGE,   _dt(d_mon, 3, 0), _dt(d_mon, 3,55)),
            (Session.OVERNIGHT_SESSION, _dt(d_mon, 4, 0), _dt(d_mon, 8,25)),
            (Session.TRANSITION_REGULAR,_dt(d_mon, 8,30), _dt(d_mon, 9,25)),
        ]
    else:
        windows += [
            (Session.TRANSITION_ASIA,   _dt(d0,18, 0), _dt(d0,19,25)),
            (Session.ASIA_RANGE,        _dt(d0,19,30), _dt(d0,20,25)),
            (Session.ASIA_SESSION,      _dt(d0,20,30), _dt(d1, 1,55)),
            (Session.TRANSITION_OVERNIGHT,_dt(d1, 2, 0), _dt(d1, 2,55)),
            (Session.OVERNIGHT_RANGE,   _dt(d1, 3, 0), _dt(d1, 3,55)),
            (Session.OVERNIGHT_SESSION, _dt(d1, 4, 0), _dt(d1, 8,25)),
            (Session.TRANSITION_REGULAR,_dt(d1, 8,30), _dt(d1, 9,25)),
        ]
    return windows

# ---- Utilities

def _in_window(ts: pd.Series, start: datetime, end: datetime) -> pd.Series:
    return (ts >= start) & (ts <= end)

def _first_breakout_times(bars: pd.DataFrame,
                          hi: float, lo: float,
                          windows: List[Tuple[str, datetime, datetime]]
                          ) -> Tuple[Optional[str], Optional[pd.Timestamp], Dict[str, Optional[pd.Timestamp]]]:
    out_by_sess: Dict[str, Optional[pd.Timestamp]] = {s: None for s, _, _ in windows}

    # Precompute first up and down timestamps over the WHOLE window
    up_mask = bars['h'] > hi
    dn_mask = bars['l'] < lo
    first_up_ts = bars.loc[up_mask, 'ts'].min() if up_mask.any() else None
    first_dn_ts = bars.loc[dn_mask, 'ts'].min() if dn_mask.any() else None

    if first_up_ts is None and first_dn_ts is None:
        side = None
        t_first = None
    elif first_up_ts is not None and (first_dn_ts is None or first_up_ts <= first_dn_ts):
        side = "up"
        t_first = pd.Timestamp(first_up_ts)
    else:
        side = "down"
        t_first = pd.Timestamp(first_dn_ts)

    # Fill per-session first hits (still useful diagnostics)
    for s, start, end in windows:
        dfw = bars[(bars['ts'] >= start) & (bars['ts'] <= end)]
        if dfw.empty:
            out_by_sess[s] = None
            continue
        t_up = dfw.loc[dfw['h'] > hi, 'ts'].min()
        t_dn = dfw.loc[dfw['l'] < lo, 'ts'].min()
        if pd.isna(t_up) and pd.isna(t_dn):
            out_by_sess[s] = None
        elif pd.isna(t_up):
            out_by_sess[s] = pd.Timestamp(t_dn)
        elif pd.isna(t_dn):
            out_by_sess[s] = pd.Timestamp(t_up)
        else:
            out_by_sess[s] = pd.Timestamp(t_up) if t_up <= t_dn else pd.Timestamp(t_dn)

    return side, t_first, out_by_sess

def _quarters(lo: float, hi: float) -> Dict[str, float]:
    rng = hi - lo
    q = rng / 4.0 if rng > 0 else np.nan
    mid = (hi + lo) / 2.0
    return {
        "q": q, "rng": rng, "mid": mid,
        "q1": lo + q, "q2": lo + 2*q, "q3": lo + 3*q
    }

def _max_ext_ret_in_Q(bars: pd.DataFrame, lo: float, hi: float, q: float) -> Tuple[float, float, float]:
    """
    Max extension beyond hi in Q units, max retracement below lo in Q units,
    and 'retracement before high' measured from initial move toward the edge.
    """
    if not np.isfinite(q) or q <= 0:
        return np.nan, np.nan, np.nan

    max_extQ = 0.0
    max_retQ = 0.0
    retrace_before_hiQ = 0.0

    # track highest high, lowest low relative to edges
    above = (bars['h'] - hi).max()
    below = (lo - bars['l']).max()
    if pd.notna(above):
        max_extQ = max(0.0, above / q)
    if pd.notna(below):
        max_retQ = max(0.0, below / q)

    # retracement before high: how far price pulls back (below hi) before first time it breaks above?
    # Find earliest bar making h > hi; before that, look how far l dips from the local swing toward hi.
    hit_up_idx = bars.index[(bars['h'] > hi)]
    if len(hit_up_idx):
        df_pre = bars.loc[:hit_up_idx[0]]
        if not df_pre.empty:
            # From the **start of RDR** path, find min distance from hi
            min_gap = (hi - df_pre['l']).max()
            if pd.notna(min_gap):
                retrace_before_hiQ = max(0.0, min_gap / q)

    return float(max_extQ), float(max_retQ), float(retrace_before_hiQ)

def _range_high_low_mid_sd(df_slice: pd.DataFrame,
                           mid: float,
                           sigma: float) -> tuple[float, float]:
    """
    Returns (high_mid_sigma, low_mid_sigma) for the provided slice.
    Mid is the RDR midpoint (treated as 0); sigma = hi - lo of that RDR box.
    """
    if (df_slice is None or df_slice.empty or
        not np.isfinite(mid) or not (np.isfinite(sigma) and sigma > 0)):
        return np.nan, np.nan

    hi = df_slice['h'].max()
    lo = df_slice['l'].min()
    hi_sd = (hi - mid) / sigma if np.isfinite(hi) else np.nan
    lo_sd = (lo - mid) / sigma if np.isfinite(lo) else np.nan
    return float(hi_sd), float(lo_sd)

def _quarter_offset(value: float, mid: float, q: float) -> float:
    """
    Express 'value' in quarter units relative to the RDR mid (mid = 0 line).
    Returns NaN when inputs are invalid or q<=0.
    """
    if not (np.isfinite(value) and np.isfinite(mid) and np.isfinite(q) and q > 0):
        return np.nan
    return (value - mid) / q

def _asia_range_model(lo_quarters: float, hi_quarters: float) -> str:
    """
    Bucket Asia Range placement relative to the RDR mid using quarter thresholds:
      - UQ: entire range ≥ +0.25 quarters
      - UM: entire range ≥ 0 quarters (above mid)
      - DQ: entire range ≤ -0.25 quarters
      - DM: entire range ≤ 0 quarters (below mid)
      - RM: range crosses the mid (default)
    Returns "NA" when inputs are not finite.
    """
    if not (np.isfinite(lo_quarters) and np.isfinite(hi_quarters)):
        return "NA"
    if lo_quarters >= 0.25:
        return "UQ"
    if lo_quarters >= 0.0:
        return "UM"
    if hi_quarters <= -0.25:
        return "DQ"
    if hi_quarters <= 0.0:
        return "DM"
    return "RM"

def _classify_next_model(curr_lo: float, curr_hi: float,
                         next_lo: float, next_hi: float,
                         curr_mid: float, next_mid: float) -> str:
    """
    Priority order ensures mutual exclusivity and makes U/D/UX/DX appear as intended.

    I  = fully inside
    O  = fully engulf
    U  = fully above prior high
    D  = fully below prior low
    UX = broke above high, AND low >= prior mid (upper-half straddle)
    DX = broke below low,  AND high <= prior mid (lower-half straddle)
    C  = overlap but none of the above
    """
    # Inside / Engulf first
    if next_lo >= curr_lo and next_hi <= curr_hi:
        return "I"
    if next_lo <= curr_lo and next_hi >= curr_hi:
        return "O"

    # Clean outside (no overlap)
    if next_lo >= curr_hi:
        return "U"
    if next_hi <= curr_lo:
        return "D"

    # Straddle-up: broke above the high but overlaps the prior range
    if next_hi > curr_hi and next_lo < curr_hi:
        return "UX" if next_lo >= curr_mid else "U"

    # Straddle-down: broke below the low but overlaps the prior range
    if next_lo < curr_lo and next_hi > curr_lo:
        return "DX" if next_hi <= curr_mid else "D"

    # Everything else is overlap chain
    return "C"

def _gap_code(next_open: float, prior_close: float, curr_lo: float, curr_hi: float, next_lo: float, next_hi: float) -> Optional[str]:
    # UG: up gap at 09:30 AND no overlap to prior (no chain)
    # DG: down gap at 09:30 AND no overlap to prior (no chain)
    overlaps = not (next_lo > curr_hi or next_hi < curr_lo)
    if next_open > prior_close and (next_lo > curr_hi) and not overlaps:
        return "UG"
    if next_open < prior_close and (next_hi < curr_lo) and not overlaps:
        return "DG"
    return None

def _open_pos_quarters(next_open: float, lo: float, hi: float, q1: float, q2: float, q3: float) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return "NA"
    if next_open < lo:
        return "below_low"
    if next_open < q1:
        return "Q0"
    if next_open < q2:
        return "Q1"
    if next_open < q3:
        return "Q2"
    if next_open < hi:
        return "Q3"
    return "above_high"

@dataclass
class WDDRBConfig:
    symbol: str
    tz: ZoneInfo = NY_TZ

class WDDRBBuilder:
    """
    Build the WDDRB daily DB from 5-minute bars in NY time.

    Input bars DataFrame must have columns:
      ['ts','o','h','l','c','v','symbol'] with 'ts' timezone-aware (NY) or UTC + we localize.

    Output schema (per 'day' where day1=Tuesday):
      - day_date (YYYY-MM-DD, NY)
      - weekday_num (Tue=1..Mon=7)
      - symbol
      - rdr_lo, rdr_hi, rdr_mid, rdr_rng, rdr_q
      - rdr_q1, rdr_q2, rdr_q3
      - asia_range_high/low + close-based highs/lows with quarter offsets vs rdr_mid (asia_range_* fields)
      - asia_range_model_hilo / asia_range_model_close (UQ/UM/RM/DM/DQ buckets)
      - first_break_side ('up'/'down'/None)
      - first_break_ts (HH:MM NY)
      - first_break_session (session code like RR/RS/AR)
      - high_sd_full/low_sd_full: full window (15:55→next 09:25) highs/lows vs RDR mid in σ units
      - high_sd_pre_break/low_sd_pre_break: highs/lows before first breakout (>=15:55)
      - high_sd_post_break/low_sd_post_break: highs/lows after first breakout
      - full_high_ts/full_low_ts (+_index_30m) for highs/lows after 15:55
      - post_high_ts/post_low_ts/post_total_ts with corresponding post_high_index_30m/post_low_index_30m/post_total_index_30m
      - next_open (next day 09:30)
      - next_open_ts (HH:MM NY) and next_open_sd (next 09:30 open vs RDR mid in σ units)
      - next_vs_1555 ('above'/'below'/'equal')
      - next_rdr_lo, next_rdr_hi, next_rdr_mid
      - model_code ('O','I','U','D','UX','DX','UG','DG','C')
      - chain_id_overlap, chain_len_overlap (original overlap-based chaining)
      - chain_id_model, chain_len_model (model-based chaining, breaks only on UG/DG)
      - is_chain_overlap, is_chain_model (boolean flags for chaining status)
      - geom_overlap_with_prev (whether ranges geometrically overlap/touch)
    """
    def __init__(self, cfg: WDDRBConfig):
        self.cfg = cfg

    @staticmethod
    def _weekday_idx_tuesday_one(d: date) -> int:
        # Python Mon=0..Sun=6; we want Tue=1..Mon=7
        mapping = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 0:7}
        return mapping[d.weekday()]

    def _ensure_ts(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['ts']):
            df['ts'] = pd.to_datetime(df['ts'], utc=True)
        if df['ts'].dt.tz is None:
            df['ts'] = df['ts'].dt.tz_localize("UTC")
        # convert to NY for window tagging
        df['ts'] = df['ts'].dt.tz_convert(self.cfg.tz)
        return df

    def _day_date_from_ts(self, ts: pd.Timestamp) -> date:
        # "anchor date" is the REGULAR day date (same as calendar date in NY)
        return ts.astimezone(self.cfg.tz).date()

    def _tag_session(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # We'll tag each row with session name based on that row's anchor date windows
        sess = []
        for i, row in df.iterrows():
            d = self._day_date_from_ts(row['ts'])
            label = None
            for s, start, end in _windows_for_day(d):
                if start <= row['ts'] <= end:
                    label = s
                    break
            sess.append(label)
        df['session'] = sess
        return df

    def build(self, bars_5m: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_ts(bars_5m)
        df = df[df['symbol'] == self.cfg.symbol].sort_values('ts').reset_index(drop=True)
        df = self._tag_session(df)

        # Precompute per-date RDR bounds
        day_candidates = sorted({self._day_date_from_ts(ts) for ts in df['ts']})
        def rdr_bounds(d: date):
            wins = _windows_for_day(d)
            rr = next((w for w in wins if w[0] == Session.REGULAR_RANGE), None)
            rs = next((w for w in wins if w[0] == Session.REGULAR_SESSION), None)
            if rr is None or rs is None:
                return None, None, pd.DataFrame()
            mask = _in_window(df['ts'], rr[1], rs[2])
            slice_df = df.loc[mask]
            if slice_df.empty:
                return None, None, slice_df
            return float(slice_df['l'].min()), float(slice_df['h'].max()), slice_df

        rdr_by_day: Dict[date, Tuple[Optional[float], Optional[float], pd.DataFrame]] = {
            d: rdr_bounds(d) for d in day_candidates
        }

        # Keep only days that actually have an RDR
        valid_days = [d for d, (lo, hi, _) in rdr_by_day.items() if lo is not None and hi is not None]
        valid_days.sort()

        records = []
        for i, d in enumerate(valid_days):
            rdr_lo, rdr_hi, rdr_slice = rdr_by_day[d]
            Q = _quarters(rdr_lo, rdr_hi)
            wins = _windows_for_day(d)

            # breakout window = from current 09:30 to next day 09:25
            start = next((s for s in wins if s[0] == Session.REGULAR_RANGE))[1]
            end = wins[-1][2]  # TRANSITION_REGULAR end (09:25 next day)
            mask_all = _in_window(df['ts'], start, end)
            df_window = df.loc[mask_all].copy()

            side, t_first, by_sess = _first_breakout_times(df_window, rdr_hi, rdr_lo, wins)

            # σ is the prior RDR range; treat it as "1 standard deviation"
            sigma = Q["rng"]

            reg_session = next((w for w in wins if w[0] == Session.REGULAR_SESSION), None)
            reg_end = reg_session[2] if reg_session else None
            if reg_end is not None:
                df_after_box = df_window[df_window['ts'] >= reg_end]
            else:
                df_after_box = df_window

            # High/low placement in σ units (mid = 0) for different slices
            full_hi_sd, full_low_sd = _range_high_low_mid_sd(df_after_box, Q["mid"], sigma)
            full_hi_ts = full_low_ts = None
            if not df_after_box.empty:
                hi_series_box = df_after_box['h'].dropna()
                lo_series_box = df_after_box['l'].dropna()
                if not hi_series_box.empty:
                    full_hi_ts = df_after_box.loc[hi_series_box.idxmax(), 'ts']
                if not lo_series_box.empty:
                    full_low_ts = df_after_box.loc[lo_series_box.idxmin(), 'ts']
            post_window_minutes = _post_window_total_minutes(wins)

            def _bucket_from_minutes(minutes: float) -> float:
                if not np.isfinite(minutes):
                    return np.nan
                max_idx = np.ceil(post_window_minutes / 30.0) - 1 if np.isfinite(post_window_minutes) else np.nan
                idx = np.floor(minutes / 30.0)
                if np.isfinite(max_idx) and max_idx >= 0:
                    idx = min(idx, max_idx)
                return float(max(0.0, idx))

            full_high_minutes = _minutes_since_reg_close(full_hi_ts, reg_end, wins)
            full_low_minutes = _minutes_since_reg_close(full_low_ts, reg_end, wins)
            full_high_index_30m = _bucket_from_minutes(full_high_minutes)
            full_low_index_30m = _bucket_from_minutes(full_low_minutes)

            if t_first is not None and reg_end is not None:
                if t_first <= reg_end:
                    df_pre = pd.DataFrame(columns=df_window.columns)
                    df_post = df_after_box
                else:
                    df_pre = df_after_box[df_after_box['ts'] < t_first]
                    df_post = df_after_box[df_after_box['ts'] >= t_first]
            elif t_first is not None:
                df_pre = df_window[df_window['ts'] < t_first]
                df_post = df_window[df_window['ts'] >= t_first]
            else:
                df_pre = df_after_box
                df_post = pd.DataFrame(columns=df_window.columns)
            pre_hi_sd, pre_low_sd = _range_high_low_mid_sd(df_pre, Q["mid"], sigma)
            post_hi_sd, post_low_sd = _range_high_low_mid_sd(df_post, Q["mid"], sigma)

            # Time-to-extreme metrics after breakout (30 minute buckets)
            post_high_ts = post_low_ts = post_end_ts = None
            post_high_index_30m = post_low_index_30m = post_total_index_30m = np.nan
            if t_first is not None and not df_post.empty:
                post_end_ts = df_post['ts'].max()

                h_series = df_post['h'].dropna()
                if not h_series.empty:
                    hi_ts = df_post.loc[h_series.idxmax(), 'ts']
                    post_high_ts = hi_ts
                    minutes = _minutes_since_reg_close(hi_ts, reg_end, wins)
                    post_high_index_30m = _bucket_from_minutes(minutes)

                l_series = df_post['l'].dropna()
                if not l_series.empty:
                    low_ts = df_post.loc[l_series.idxmin(), 'ts']
                    post_low_ts = low_ts
                    minutes = _minutes_since_reg_close(low_ts, reg_end, wins)
                    post_low_index_30m = _bucket_from_minutes(minutes)

                minutes = _minutes_since_reg_close(post_end_ts, reg_end, wins)
                post_total_index_30m = _bucket_from_minutes(minutes)
            else:
                post_end_ts = None
            # Asia Range (ADR) stats relative to RDR mid quarters
            asia_win = next((w for w in wins if w[0] == Session.ASIA_RANGE), None)
            if asia_win is not None:
                asia_mask = _in_window(df['ts'], asia_win[1], asia_win[2])
                df_asia = df.loc[asia_mask]
            else:
                df_asia = pd.DataFrame()

            if df_asia.empty:
                asia_hi = asia_lo = asia_high_close = asia_low_close = np.nan
            else:
                asia_hi = float(df_asia['h'].max())
                asia_lo = float(df_asia['l'].min())
                asia_high_close = float(df_asia['c'].max())
                asia_low_close = float(df_asia['c'].min())

            asia_hi_quarters = _quarter_offset(asia_hi, Q["mid"], Q["q"])
            asia_lo_quarters = _quarter_offset(asia_lo, Q["mid"], Q["q"])
            asia_high_close_quarters = _quarter_offset(asia_high_close, Q["mid"], Q["q"])
            asia_low_close_quarters = _quarter_offset(asia_low_close, Q["mid"], Q["q"])

            asia_model_hilo = _asia_range_model(asia_lo_quarters, asia_hi_quarters)
            asia_model_close = _asia_range_model(asia_low_close_quarters, asia_high_close_quarters)

            # Close 15:55 (end of REGULAR_SESSION)
            close_1555 = df.loc[df['ts'] == reg_end, 'c'] if reg_end is not None else pd.Series(dtype=float)
            if close_1555.empty and reg_end is not None:
                # fallback: nearest at/before 15:55
                close_1555 = df.loc[df['ts'] <= reg_end, 'c'].tail(1)
            close_1555 = float(close_1555.iloc[0]) if len(close_1555) else np.nan

            # --- Next VALID day and 09:30 open (Friday→Monday handled by valid_days) ---
            if i + 1 < len(valid_days):
                d_next = valid_days[i+1]
                wins_next = _windows_for_day(d_next)
                rr_next = next((w for w in wins_next if w[0] == Session.REGULAR_RANGE), None)
                next_open_ts = rr_next[1] if rr_next else None

                next_open = np.nan
                if next_open_ts is not None:
                    row = df.loc[df['ts'] == next_open_ts]
                    if row.empty:
                        row = df.loc[df['ts'] >= next_open_ts].head(1)
                    if not row.empty:
                        next_open = float(row['o'].iloc[0])

                next_lo, next_hi, _ = rdr_by_day.get(d_next, (np.nan, np.nan, pd.DataFrame()))
                next_mid = (next_hi + next_lo)/2.0 if np.isfinite(next_lo) and np.isfinite(next_hi) else np.nan
            else:
                d_next, next_open_ts, next_open, next_lo, next_hi, next_mid = None, None, np.nan, np.nan, np.nan, np.nan

            # --- Next open relative to RDR mid (mid = 0 in σ units) ---
            next_open_sd = (next_open - Q["mid"]) / sigma if np.isfinite(next_open) and sigma > 0 else np.nan

            # 15:55 close compare (keep as you had)
            next_vs_1555 = "NA"
            if np.isfinite(next_open) and np.isfinite(close_1555):
                if abs(next_open - close_1555) < 1e-9:
                    next_vs_1555 = "equal"
                else:
                    next_vs_1555 = "above" if next_open > close_1555 else "below"

            model_code = _classify_next_model(rdr_lo, rdr_hi, next_lo, next_hi, Q["mid"], next_mid) if next_lo is not None and next_hi is not None and np.isfinite(next_lo) and np.isfinite(next_hi) else "NA"

            # Gap code (UG/DG) if applicable (and only when it doesn't "chain")
            gap_code = _gap_code(next_open, close_1555, rdr_lo, rdr_hi, next_lo, next_hi) if np.isfinite(next_open) and np.isfinite(close_1555) and next_lo is not None and next_hi is not None and np.isfinite(next_lo) and np.isfinite(next_hi) else None
            if gap_code in ("UG","DG"):
                model_code = gap_code

            rec = {
                "day_date": d.isoformat(),
                "weekday_num": self._weekday_idx_tuesday_one(d),
                "symbol": self.cfg.symbol,
                "rdr_lo": rdr_lo, "rdr_hi": rdr_hi, "rdr_mid": Q["mid"], "rdr_rng": Q["rng"], "rdr_q": Q["q"],
                "rdr_q1": Q["q1"], "rdr_q2": Q["q2"], "rdr_q3": Q["q3"],
                "asia_range_high": asia_hi,
                "asia_range_low": asia_lo,
                "asia_range_high_close": asia_high_close,
                "asia_range_low_close": asia_low_close,
                "asia_range_high_quarters": asia_hi_quarters,
                "asia_range_low_quarters": asia_lo_quarters,
                "asia_range_high_close_quarters": asia_high_close_quarters,
                "asia_range_low_close_quarters": asia_low_close_quarters,
                "asia_range_model_hilo": asia_model_hilo,
                "asia_range_model_close": asia_model_close,
                "first_break_side": side,
                "first_break_ts": _format_intraday_time(t_first),
                "first_break_session": None,
                "next_open_ts": _format_intraday_time(next_open_ts),
                "next_open": next_open,
                "next_open_sd": next_open_sd,

                "next_vs_1555": next_vs_1555,
                "next_rdr_lo": next_lo,
                "next_rdr_hi": next_hi,
                "next_rdr_mid": next_mid,
                "model_code": model_code,
            }
            
            # Add signed σ metrics (post-breakout)
            rec.update({
                "high_sd_full": full_hi_sd,
                "low_sd_full": full_low_sd,
                "high_sd_pre_break": pre_hi_sd,
                "low_sd_pre_break": pre_low_sd,
                "high_sd_post_break": post_hi_sd,
                "low_sd_post_break": post_low_sd,
                "full_high_ts": _format_intraday_time(full_hi_ts),
                "full_high_index_30m": full_high_index_30m,
                "full_low_ts": _format_intraday_time(full_low_ts),
                "full_low_index_30m": full_low_index_30m,
                "post_high_ts": _format_intraday_time(post_high_ts),
                "post_high_index_30m": post_high_index_30m,
                "post_low_ts": _format_intraday_time(post_low_ts),
                "post_low_index_30m": post_low_index_30m,
                "post_total_ts": _format_intraday_time(post_end_ts),
                "post_total_index_30m": post_total_index_30m,
            })
            # derive first_break_session from which session contains t_first
            if t_first is not None:
                t_fb = pd.Timestamp(t_first).tz_convert(NY_TZ)
                for s, start, end in wins:
                    if start <= t_fb <= end:
                        rec["first_break_session"] = SESSION_SHORT.get(s, s)
                        break

            records.append(rec)

        df_out = pd.DataFrame.from_records(records).sort_values("day_date").reset_index(drop=True)

        # ---- A) Keep your current overlap-based chain (unchanged) ----
        # Chains: consecutive overlap (touch/overlap) vs non-overlap
        chain_id = 0
        lens = []
        ids = []
        run_len = 0
        prev_hi = prev_lo = None
        for idx, row in df_out.iterrows():
            if prev_hi is None:
                chain_id += 1
                run_len = 1
            else:
                overlaps = not (row['rdr_lo'] > prev_hi or row['rdr_hi'] < prev_lo)
                if overlaps:
                    run_len += 1
                else:
                    # new chain
                    # backfill previous chain length
                    for j in range(idx - run_len, idx):
                        lens.append((j, run_len, chain_id))
                    chain_id += 1
                    run_len = 1
            prev_lo, prev_hi = row['rdr_lo'], row['rdr_hi']
            ids.append(chain_id)
        # finalize last chain backfill
        if run_len > 0 and len(df_out):
            start = len(df_out) - run_len
            for j in range(start, len(df_out)):
                lens.append((j, run_len, chain_id))
        # apply
        lens_sorted = sorted(lens, key=lambda x: x[0])
        df_out["chain_len"] = [L[1] for L in lens_sorted]
        df_out["chain_id"] = [L[2] for L in lens_sorted]

        # Preserve them for reference
        df_out = df_out.rename(columns={
            "chain_id": "chain_id_overlap",
            "chain_len": "chain_len_overlap"
        })

        # Quick helper
        def _overlaps(prev_lo, prev_hi, cur_lo, cur_hi) -> bool:
            # "touching counts as overlap" (chain) – keep your original semantics
            return not (cur_lo > prev_hi or cur_hi < prev_lo)

        # ---- B) NEW: model-based chaining (only UG/DG break the chain) ----
        n = len(df_out)
        model_chain_id = [None] * n
        model_chain_len = [None] * n

        cid = 0
        run_start = 0
        for i in range(n):
            code = df_out.at[i, "model_code"]
            # Start a new chain on first row or when we see UG/DG (non-chaining models)
            if i == 0 or (isinstance(code, str) and code in ("UG", "DG")):
                # backfill previous chain lens
                if i > 0:
                    L = i - run_start
                    for j in range(run_start, i):
                        model_chain_len[j] = L
                cid += 1
                run_start = i
            model_chain_id[i] = cid

        # finalize last chain
        if n > 0:
            L = n - run_start
            for j in range(run_start, n):
                model_chain_len[j] = L

        df_out["chain_id_model"] = model_chain_id
        df_out["chain_len_model"] = model_chain_len

        # convenience flags
        df_out["is_chain_overlap"] = True  # by definition of chain_id_overlap groups
        df_out.loc[df_out["chain_len_overlap"].isna(), "is_chain_overlap"] = False

        df_out["is_chain_model"] = ~df_out["model_code"].isin(["UG", "DG"])
        df_out.loc[df_out["model_code"].isna(), "is_chain_model"] = False  # NA = unknown → treat as break if you prefer

        # Optional: also expose whether the *ranges* overlap/touch (geometry),
        # so you can analyze where U/D/UX/DX are geometrically non-overlapping but still chained by the model rule.
        geom_overlap = [True] * n
        for i in range(n):
            if i == 0:
                geom_overlap[i] = True
                continue
            prev_lo = df_out.at[i-1, "rdr_lo"]
            prev_hi = df_out.at[i-1, "rdr_hi"]
            cur_lo  = df_out.at[i,   "rdr_lo"]
            cur_hi  = df_out.at[i,   "rdr_hi"]
            geom_overlap[i] = _overlaps(prev_lo, prev_hi, cur_lo, cur_hi)

        df_out["geom_overlap_with_prev"] = geom_overlap

        return df_out

    @staticmethod
    def save(df_out: pd.DataFrame, out_dir: Path, basename: str = "wddrb") -> Tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        pq = out_dir / f"{basename}.parquet"
        cs = out_dir / f"{basename}.csv"
        df_out.to_parquet(pq, index=False)
        df_out.to_csv(cs, index=False)
        return pq, cs


# -------- Convenience runner --------

def build_wddrb_from_csv(csv_path: Path, symbol: str, out_dir: Path) -> Tuple[Path, Path]:
    """
    csv_path must have columns: ts,o,h,l,c,v,symbol
    ts can be UTC or NY, we will normalize to NY internally.
    """
    bars = pd.read_csv(csv_path)
    # Accept int epoch ts as well
    if np.issubdtype(bars['ts'].dtype, np.number):
        bars['ts'] = pd.to_datetime(bars['ts'], unit='s', utc=True)
    cfg = WDDRBConfig(symbol=symbol)
    builder = WDDRBBuilder(cfg)
    df_out = builder.build(bars)
    return builder.save(df_out, out_dir)
