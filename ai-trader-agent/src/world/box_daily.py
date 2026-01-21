from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, Tuple, Optional

from .sessions import windows_for_day
from .calendars import next_business_day

# Post-Asia sessions included in session-level output (after AS); exclude TOUT
POST_ASIA = ["OR", "OS", "TR", "RR", "RS"]

def _slice_open(bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # OPEN timestamps ⇒ include [start, end)
    return bars.loc[(bars["ts"] >= start) & (bars["ts"] < end)]

def _hi_lo(df: pd.DataFrame) -> tuple[float, float, pd.Timestamp, pd.Timestamp]:
    if df.empty:
        return (np.nan, np.nan, pd.NaT, pd.NaT)
    hi = float(df["high"].max()); lo = float(df["low"].min())
    hit = df.loc[df["high"].idxmax(), "ts"]; lot = df.loc[df["low"].idxmin(), "ts"]
    return (hi, lo, hit, lot)

def _range_pct_change(hi_close: float, lo_close: float) -> float:
    if pd.isna(lo_close) or lo_close == 0:
        return np.nan
    return (hi_close - lo_close) / lo_close * 100.0

def _size_bucket(range_pct: float, p20: float, p50: float, p80: float) -> str:
    if pd.isna(range_pct):
        return "unknown"
    if range_pct <= p20: return "0-20%"
    if range_pct <= p50: return "20-50%"
    if range_pct <= p80: return "50-80%"
    return "80%+"

def _confirm_side_text(side_val) -> str:
    # metrics.Confirmation_Side: 1=long/up, 0=short/down, NaN=None
    if pd.isna(side_val): return "none"
    return "up" if int(side_val) == 1 else "down"

def _sd_levels(anchor_up: float, anchor_down: float, R_asia: float, confirm_side: str) -> Dict[str, float]:
    if confirm_side == "none" or pd.isna(R_asia) or R_asia <= 0:
        return {}
    anchor = anchor_up if confirm_side == "up" else anchor_down
    sign = 1.0 if confirm_side == "up" else -1.0
    out = {}
    for k in [0.2, 0.5, 0.8, 1.0]:
        out[f"sd_ext_{k}_price"] = anchor + sign * k * R_asia
        out[f"sd_ret_{k}_price"] = anchor - sign * k * R_asia
    return out

def _price_levels(mid_close: float, R_asia: float) -> Dict[str, float]:
    if pd.isna(R_asia):
        return {}
    levels = {}
    for k in [1.5, 3.0, 5.5, 8.5]:
        levels[f"+{k}"] = mid_close + k * R_asia
        levels[f"-{k}"] = mid_close - k * R_asia
    return levels

def _session_extremes_as_sd(
    part: pd.DataFrame,
    session_start: pd.Timestamp,
    mid_close: float,
    R_asia: float,
) -> Dict[str, float]:
    if part.empty or pd.isna(R_asia) or R_asia <= 0:
        return {
            "ext_sd": np.nan, "ret_sd": np.nan,
            "time_to_session_high_min": np.nan, "time_to_session_low_min": np.nan,
            "session_high": np.nan, "session_low": np.nan
        }
    hi, lo, hit, lot = _hi_lo(part)
    t_high = (hit - session_start).total_seconds() / 60.0 if pd.notna(hit) else np.nan
    t_low  = (lot - session_start).total_seconds() / 60.0 if pd.notna(lot) else np.nan

    # Always anchor to Asia mid; positive above mid, negative below
    ext_sd = (hi - mid_close) / R_asia
    ret_sd = (lo - mid_close) / R_asia

    return {
        "ext_sd": ext_sd,
        "ret_sd": ret_sd,
        "time_to_session_high_min": t_high,
        "time_to_session_low_min": t_low,
        "session_high": hi,
        "session_low": lo,
    }

def _cycle_bucket(first_ext: float, second_ret: float) -> str:
    """Bucket a pair of SD values into bins like (1.5)(-3.0), preserving sign and magnitude bands."""
    def _cat(v: float) -> str:
        if pd.isna(v): return "?"
        if v >= 0:
            if v <= 1.5: return "1.5"
            if v <= 3.0: return "3.0"
            if v <= 5.5: return "5.5"
            if v <= 8.5: return "8.5"
            return "8.5+"
        else:
            if v >= -1.5: return "-1.5"
            if v >= -3.0: return "-3.0"
            if v >= -5.5: return "-5.5"
            if v >= -8.5: return "-8.5"
            return "-8.5-"
    return f"({_cat(first_ext)})({_cat(second_ret)})"

def build_daily_box_model(
    bars_with_sessions: pd.DataFrame,   # from attach_sessions()
    ranges_df: pd.DataFrame,            # from compute_range_table()
    metrics_df: pd.DataFrame,           # from compute_confirmation_metrics()
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Asia-normalized daily box model.
    Returns (day_level_df, session_level_df).
    """
    # Pre-views
    rix = ranges_df.set_index(["symbol","day_date","session"]).sort_index()
    mix = metrics_df.set_index(["symbol","day_date","range_session"]).sort_index()

    # Build distribution of Asia range %s (for bucketing)
    aranges = ranges_df[ranges_df["session"] == "AR"].copy()
    aranges["R_asia"] = aranges["IDR_high_close"] - aranges["IDR_low_close"]
    aranges["range_pct_change"] = (aranges["R_asia"] / aranges["IDR_low_close"]) * 100.0
    # Global fallbacks (used when insufficient history exists for running thresholds)
    p20_global = float(aranges["range_pct_change"].quantile(0.20)) if len(aranges) else np.nan
    p50_global = float(aranges["range_pct_change"].quantile(0.50)) if len(aranges) else np.nan
    p80_global = float(aranges["range_pct_change"].quantile(0.80)) if len(aranges) else np.nan
    # Running (expanding) thresholds per symbol based on prior history only
    def _add_running_quantiles(df_sym: pd.DataFrame) -> pd.DataFrame:
        df_sym = df_sym.sort_values(["day_date"]).copy()
        hist = df_sym["range_pct_change"].shift(1)
        # Require a small minimum history for stability; else NaN → fallback to global later
        df_sym["p20_running"] = hist.expanding(min_periods=10).quantile(0.20)
        df_sym["p50_running"] = hist.expanding(min_periods=10).quantile(0.50)
        df_sym["p80_running"] = hist.expanding(min_periods=10).quantile(0.80)
        return df_sym
    if not aranges.empty:
        aranges = aranges.groupby("symbol", group_keys=False).apply(_add_running_quantiles)
        thr_map = aranges.set_index(["symbol","day_date"])[["p20_running","p50_running","p80_running"]].to_dict("index")
    else:
        thr_map = {}

    day_rows = []
    sess_rows = []

    for (symbol, d), _ in ranges_df[ranges_df["session"]=="AR"].groupby(["symbol","day_date"]):
        # Pull Asia range stats for this day
        try:
            ar = rix.loc[(symbol, d, "AR")]
        except KeyError:
            continue

        hi_close = float(ar["IDR_high_close"])
        lo_close = float(ar["IDR_low_close"])
        mid_close = float(ar["IDR_mid"])
        R_asia = hi_close - lo_close
        range_pct = _range_pct_change(hi_close, lo_close)
        # Fetch running thresholds for this (symbol, day); fallback to global if missing/NaN
        thr = thr_map.get((symbol, d)) or {}
        p20 = float(thr.get("p20_running", np.nan))
        p50 = float(thr.get("p50_running", np.nan))
        p80 = float(thr.get("p80_running", np.nan))
        if not np.isfinite(p20): p20 = p20_global
        if not np.isfinite(p50): p50 = p50_global
        if not np.isfinite(p80): p80 = p80_global
        size_bucket = _size_bucket(range_pct, p20, p50, p80)

        # Confirmation side from metrics (AR → AS)
        try:
            m = mix.loc[(symbol, d, "AR")]
            confirm_side = _confirm_side_text(m["Confirmation_Side"])
        except KeyError:
            confirm_side = "none"

        # Session windows for this day
        wins = {w.session.value: (w.start, w.end) for w in windows_for_day(d)}
        # Day window: AS start (on day d) → RS end of NEXT business day
        if "AS" not in wins:
            continue
        day_start = wins["AS"][0]
        next_d = next_business_day(d)
        wins_next = {w.session.value: (w.start, w.end) for w in windows_for_day(next_d)}
        if "RS" not in wins_next:
            continue
        day_end = wins_next["RS"][1]

        # Slice bars for this symbol and day window
        sdf = bars_with_sessions[bars_with_sessions["symbol"] == symbol]
        day_win = _slice_open(sdf, day_start, day_end)

        # Day extremes in Asia-SD units (anchored to Asia mid; positive above mid, negative below)
        day_max_ext_sd = np.nan
        day_min_ret_sd = np.nan
        close_loc_sd   = np.nan
        day_high = day_low = day_close = np.nan

        if not day_win.empty and R_asia > 0:
            dh, dl, hit, lot = _hi_lo(day_win)
            day_high = dh; day_low = dl; day_close = float(day_win.iloc[-1]["close"])
            # SDs relative to Asia mid (positive above, negative below)
            sd_high = (day_high - mid_close) / R_asia
            sd_low  = (day_low  - mid_close) / R_asia
            day_max_ext_sd = sd_high
            day_min_ret_sd = sd_low
            close_loc_sd   = (day_close - mid_close) / R_asia

        # Order the pair by which extreme occurred first
        if not day_win.empty and pd.notna(day_max_ext_sd) and pd.notna(day_min_ret_sd):
            first_sd, second_sd = (sd_high, sd_low) if (pd.notna(hit) and pd.notna(lot) and hit <= lot) else (sd_low, sd_high)
            cycle_category = _cycle_bucket(first_sd, second_sd)
        else:
            cycle_category = "unknown"

        # Levels
        px_lvls = _price_levels(mid_close, R_asia)

        # --- day-level row
        day_rows.append({
            "symbol": symbol,
            "day_id": d,
            "asia_anchor_id": f"{symbol}-{d:%Y%m%d}",
            "R_asia": R_asia,
            "range_pct_change": range_pct,
            "range_size_category": size_bucket,
            "anchor_asia_up": hi_close,
            "anchor_asia_down": lo_close,
            "mid_close": mid_close,
            "asia_confirm_side": confirm_side,
            **px_lvls,
            "day_max_ext_sd": day_max_ext_sd,
            "day_min_ret_sd": day_min_ret_sd,
            "close_location_sd": close_loc_sd,
            "day_high": day_high,
            "day_low": day_low,
            "day_close": day_close,
            "extreme_cycle_category": cycle_category,
        })

        # --- session-level rows for post-Asia sessions on this day
        # Build post-Asia window map: OR/OS/TR from current wins (after AS start),
        # RR/RS/TOUT from next business day's windows
        wins_post: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
        as_start = wins["AS"][0]
        for code in ["OR","OS","TR"]:
            if code in wins and wins[code][0] >= as_start:
                wins_post[code] = wins[code]
        for code in ["RR","RS"]:
            if code in wins_next:
                wins_post[code] = wins_next[code]

        for code in POST_ASIA:
            if code not in wins_post:
                continue
            s, e = wins_post[code]
            part = _slice_open(sdf, s, e)
            sess_ext = _session_extremes_as_sd(part, s, mid_close, R_asia)
            sess_rows.append({
                "symbol": symbol,
                "day_id": d,
                "asia_anchor_id": f"{symbol}-{d:%Y%m%d}",
                "session_code": code,
                "session_name": code,
                "R_asia": R_asia,
                "range_pct_change": range_pct,
                "range_size_category": size_bucket,
                "anchor_asia_up": hi_close,
                "anchor_asia_down": lo_close,
                "mid_close": mid_close,
                "asia_confirm_side": confirm_side,
                **sess_ext,
            })

    day_df = pd.DataFrame(day_rows).sort_values(["symbol","day_id"]).reset_index(drop=True)
    sess_df = pd.DataFrame(sess_rows).sort_values(["symbol","day_id","session_code"]).reset_index(drop=True)
    return day_df, sess_df

