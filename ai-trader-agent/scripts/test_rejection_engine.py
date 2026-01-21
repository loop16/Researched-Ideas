#!/usr/bin/env python3
"""
Ad-hoc test harness for the Interaction/ Rejection Engine.

Unlike the cluster-based harness, this builder derives all levels from the
current data feed:
  • Asia box levels are rebuilt daily from the latest AR range (no historical pull)
  • Session levels (AR/OR/RR) use only the current session plus the three prior
    occurrences, taking the same metrics that `clusters.py` emits.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import typer

# Ensure `src/` is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from world.io import load_bars  # noqa: E402
from world.engines.interaction_engine import (  # noqa: E402
    InteractionEngine,
    LevelDefinition,
)


app = typer.Typer(add_completion=False)


DATA_DIR = Path(__file__).parent.parent / "data"
NY_TZ = "America/New_York"
TRADING_DAY_START = (20, 30)  # 20:30 NY time (evening setup)
TRADING_DAY_END = (16, 0)     # 16:00 NY time (carry into next RTH close)
CLUSTER_FAMILIES = {
    "Monthly": "clusters_monthly",
    "Weekly": "clusters_weekly",
    "DarkPool": "clusters_darkpool",
}


def _naive_trading_day(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(NY_TZ).tz_localize(None)
    return ts.normalize()


def _trading_day_for_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    """Map a timestamp to its trading day (20:30 → 16:00 next day window)."""
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY_TZ)
    else:
        ts = ts.tz_convert(NY_TZ)

    hour = ts.hour
    minute = ts.minute

    start_h, start_m = TRADING_DAY_START
    end_h, end_m = TRADING_DAY_END

    # After 20:30 belongs to the NEXT calendar day label (setup for that RTH day).
    if (hour > start_h) or (hour == start_h and minute >= start_m):
        return (ts + pd.Timedelta(days=1)).normalize().tz_localize(None)

    # Before/equal 16:00 belongs to the current calendar day label.
    if (hour < end_h) or (hour == end_h and minute <= end_m):
        return ts.normalize().tz_localize(None)

    # Between 16:00 and 20:30: keep current calendar day label (no active box).
    return ts.normalize().tz_localize(None)


def _asia_box_window(trading_day: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return start/end timestamps for the Asia box validity window for a trading day."""
    start_ts = (pd.Timestamp(trading_day) - pd.Timedelta(days=1)).tz_localize(NY_TZ) + pd.Timedelta(
        hours=TRADING_DAY_START[0], minutes=TRADING_DAY_START[1]
    )
    end_ts = pd.Timestamp(trading_day).tz_localize(NY_TZ) + pd.Timedelta(
        hours=TRADING_DAY_END[0], minutes=TRADING_DAY_END[1]
    )
    return start_ts, end_ts


def _resolve_instrument_dir(instrument: str, data_root: Path | None = None) -> Path:
    base = data_root or DATA_DIR
    return base / "processed" / instrument.upper()


def load_bars_for_range(
    instrument: str,
    start_date: str,
    end_date: str,
    data_root: Path | None = None,
) -> pd.DataFrame:
    """Load instrument bars and trim to the requested inclusive date span."""
    inst_dir = _resolve_instrument_dir(instrument, data_root)
    bars_path = inst_dir / f"{instrument.upper()}_bars_with_sessions.parquet"
    if not bars_path.exists():
        raise FileNotFoundError(f"Bars file not found: {bars_path}")

    typer.echo(f"[load] Reading bars from {bars_path}")
    df = pd.read_parquet(bars_path)
    df["ts"] = pd.to_datetime(df["ts"])

    start_dt = pd.Timestamp(start_date, tz=NY_TZ)
    # Extend the end bound to cover the cross-midnight tail of the final trading day.
    end_dt = pd.Timestamp(end_date, tz=NY_TZ) + pd.Timedelta(days=2)
    mask = (df["ts"] >= start_dt) & (df["ts"] < end_dt)
    trimmed = df.loc[mask].copy()
    if trimmed.empty:
        raise ValueError(f"No bars found between {start_date} and {end_date}")

    trimmed = trimmed.sort_values("ts").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        if col not in trimmed.columns:
            raise ValueError(f"Missing required column '{col}' in bars file.")
    if "volume" not in trimmed.columns:
        trimmed["volume"] = 0.0
    typer.echo(f"[load] Loaded {len(trimmed):,} bars for testing window.")
    return trimmed


def load_ranges_table(
    instrument: str,
    data_root: Path | None = None,
) -> pd.DataFrame:
    inst_dir = _resolve_instrument_dir(instrument, data_root)
    ranges_path = inst_dir / f"{instrument.upper()}_ranges.parquet"
    if not ranges_path.exists():
        raise FileNotFoundError(f"Ranges file not found: {ranges_path}")
    df = pd.read_parquet(ranges_path)
    if "day_date" not in df.columns:
        raise ValueError("Ranges table missing 'day_date' column.")
    df["day_date"] = pd.to_datetime(df["day_date"]).dt.normalize()
    return df


def load_cluster_tables(
    instrument: str,
    *,
    data_root: Path | None = None,
) -> Dict[str, pd.DataFrame]:
    """Load cluster parquet files for monthly/weekly/darkpool families."""
    inst_dir = _resolve_instrument_dir(instrument, data_root)
    tables: Dict[str, pd.DataFrame] = {}
    for family, suffix in CLUSTER_FAMILIES.items():
        path = inst_dir / f"{instrument.upper()}_{suffix}.parquet"
        if not path.exists():
            typer.echo(f"[warn] Missing cluster file for {family}: {path}")
            continue
        df = pd.read_parquet(path)
        if "anchor_date" not in df.columns:
            typer.echo(f"[warn] Cluster file {path} lacks 'anchor_date'; skipping.")
            continue
        df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.normalize()
        tables[family] = df
    return tables


def _build_cluster_levels_for_day(
    cluster_tables: Dict[str, pd.DataFrame] | None,
    symbol: str,
    day: pd.Timestamp,
    *,
    lookback: int = 12,
) -> Dict[str, List[LevelDefinition]]:
    if not cluster_tables:
        return {}
    day_naive = _naive_trading_day(day)
    out: Dict[str, List[LevelDefinition]] = {}
    stat_columns = [
        ("cluster_high_high", "cluster_high_high"),
        ("cluster_high_close", "cluster_high_close"),
        ("cluster_low_low", "cluster_low_low"),
        ("cluster_low_close", "cluster_low_close"),
        ("session_open", "session_open"),
        ("session_close", "session_close"),
    ]
    for family, df in cluster_tables.items():
        subset = (
            df[(df["symbol"] == symbol) & (df["anchor_date"] < day_naive)]
            .sort_values("anchor_date", ascending=False)
            .head(lookback)
        )
        if subset.empty:
            continue
        levels: List[LevelDefinition] = []
        for _, row in subset.iterrows():
            anchor = pd.to_datetime(row["anchor_date"])
            session_code = str(row.get("session_code") or "")
            base_id = f"{family}_{session_code}_{anchor:%Y%m%d}" if session_code else f"{family}_{anchor:%Y%m%d}"
            for col, stat in stat_columns:
                price = row.get(col)
                if pd.isna(price):
                    continue
                level_id = f"{base_id}_{stat.upper()}"
                levels.append(
                    LevelDefinition(
                        level_id=level_id,
                        family=family,
                        price=float(price),
                        metadata={
                            "day": f"{anchor:%Y-%m-%d}",
                            "stat": stat,
                            "session": session_code,
                        },
                    )
                )
        if levels:
            out[family] = levels
    return out


def _build_box_levels_for_day(
    bars_df: pd.DataFrame,
    symbol: str,
    day: pd.Timestamp,
    *,
    multiples: Iterable[float] = (1.5, 3.0, 5.5, 8.5),
    include_current_day: bool = True,
) -> List[LevelDefinition]:
    """Derive Asia box from the live 19:30–20:25 window and carry it into the next RTH close."""
    if not include_current_day:
        return []
    day_naive = _naive_trading_day(day)
    source_day = day_naive
    valid_start, valid_end = _asia_box_window(source_day)
    # Build window: 19:30–20:25 on the evening prior to the trading day label.
    build_start = (pd.Timestamp(source_day) - pd.Timedelta(days=1)).tz_localize(NY_TZ) + pd.Timedelta(
        hours=19, minutes=30
    )
    build_end = (pd.Timestamp(source_day) - pd.Timedelta(days=1)).tz_localize(NY_TZ) + pd.Timedelta(
        hours=20, minutes=25
    )

    df = bars_df.copy()
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    mask = (df["ts"] >= build_start) & (df["ts"] <= build_end)
    window = df.loc[mask]
    if window.empty:
        return []
    hi_close = float(window["close"].max())
    lo_close = float(window["close"].min())
    mid = (hi_close + lo_close) / 2.0
    valid_from = valid_start.isoformat()
    valid_to = valid_end.isoformat()
    r_asia = hi_close - lo_close
    if r_asia <= 0 or not np.isfinite(r_asia):
        return []

    levels: List[LevelDefinition] = [
        LevelDefinition(
            level_id=f"AR_MID_{source_day:%Y%m%d}",
            family="AsiaBox",
            price=mid,
            metadata={
                "day": f"{source_day:%Y-%m-%d}",
                "stat": "mid",
                "valid_from": valid_from,
                "valid_to": valid_to,
            },
        ),
        LevelDefinition(
            level_id=f"AR_HIGH_CLOSE_{source_day:%Y%m%d}",
            family="AsiaBox",
            price=hi_close,
            metadata={
                "day": f"{source_day:%Y-%m-%d}",
                "stat": "high_close",
                "valid_from": valid_from,
                "valid_to": valid_to,
            },
        ),
        LevelDefinition(
            level_id=f"AR_LOW_CLOSE_{source_day:%Y%m%d}",
            family="AsiaBox",
            price=lo_close,
            metadata={
                "day": f"{source_day:%Y-%m-%d}",
                "stat": "low_close",
                "valid_from": valid_from,
                "valid_to": valid_to,
            },
        ),
    ]

    for mult in multiples:
        delta = mult * r_asia
        levels.append(
            LevelDefinition(
                level_id=f"AR_SD+{mult}_{source_day:%Y%m%d}",
                family="AsiaBox",
                price=mid + delta,
                metadata={
                    "day": f"{source_day:%Y-%m-%d}",
                    "stat": f"+{mult}",
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                },
            )
        )
        levels.append(
            LevelDefinition(
                level_id=f"AR_SD-{mult}_{source_day:%Y%m%d}",
                family="AsiaBox",
                price=mid - delta,
                metadata={
                    "day": f"{source_day:%Y-%m-%d}",
                    "stat": f"-{mult}",
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                },
            )
        )

    return levels


def _build_recent_session_levels(
    ranges_df: pd.DataFrame,
    symbol: str,
    session_code: str,
    as_of_day: pd.Timestamp,
    *,
    depth: int = 3,
    include_current_day: bool = False,
) -> List[LevelDefinition]:
    """
    Build levels from the current session plus the prior `depth-1` occurrences.

    Metrics mirror the cluster builder: range high/low, open/close, and
    highest/lowest close (IDR high/low close).
    """
    day_naive = _naive_trading_day(as_of_day)
    subset = ranges_df[
        (ranges_df["symbol"] == symbol)
        & (ranges_df["session"] == session_code)
        & (
            ranges_df["day_date"] < day_naive
            if not include_current_day
            else ranges_df["day_date"] <= day_naive
        )
    ].sort_values("day_date", ascending=False)
    if subset.empty:
        return []

    levels: List[LevelDefinition] = []
    for _, row in subset.head(depth).iterrows():
        day = pd.to_datetime(row["day_date"])
        suffix = f"{session_code}_{day:%Y%m%d}"
        stats = {
            "RANGE_HIGH": row.get("DR_high"),
            "RANGE_LOW": row.get("DR_low"),
            "RANGE_OPEN": row.get("range_open"),
            "RANGE_CLOSE": row.get("range_end_close"),
            "HIGH_CLOSE": row.get("IDR_high_close"),
            "LOW_CLOSE": row.get("IDR_low_close"),
            "MID": row.get("IDR_mid"),
        }
        for stat, price in stats.items():
            if pd.isna(price):
                continue
            levels.append(
                LevelDefinition(
                    level_id=f"{suffix}_{stat}",
                    family=f"{session_code}_Session",
                    price=float(price),
                    metadata={"day": f"{day:%Y-%m-%d}", "stat": stat},
                )
            )
    return levels


def _build_recent_session_levels_from_bars(
    bars_df: pd.DataFrame,
    symbol: str,
    session_code: str,
    trading_day: pd.Timestamp,
    *,
    depth: int = 3,
) -> List[LevelDefinition]:
    """
    Build session levels from raw bars for the last `depth` completed sessions
    before `trading_day`. Requires a 'session' column on bars and a '_trading_day'
    column already computed.
    """
    if "session" not in bars_df.columns or "_trading_day" not in bars_df.columns:
        return []
    df = bars_df
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    # sessions completed before this trading day
    mask = (df["session"] == session_code) & (df["_trading_day"] < trading_day)
    df = df.loc[mask]
    if df.empty:
        return []
    levels: List[LevelDefinition] = []
    grouped = df.groupby("_trading_day")
    recent_days = sorted(grouped.groups.keys(), reverse=True)[:depth]
    for day_key in recent_days:
        g = grouped.get_group(day_key).sort_values("ts")
        if g.empty:
            continue
        day = pd.to_datetime(day_key)
        open_px = float(g.iloc[0]["open"])
        close_px = float(g.iloc[-1]["close"])
        high_px = float(g["high"].max())
        low_px = float(g["low"].min())
        hi_close = float(g["close"].max())
        lo_close = float(g["close"].min())
        mid = (hi_close + lo_close) / 2.0
        suffix = f"{session_code}_{day:%Y%m%d}"
        stats = {
            "RANGE_HIGH": high_px,
            "RANGE_LOW": low_px,
            "RANGE_OPEN": open_px,
            "RANGE_CLOSE": close_px,
            "HIGH_CLOSE": hi_close,
            "LOW_CLOSE": lo_close,
            "MID": mid,
        }
        for stat, price in stats.items():
            if not np.isfinite(price):
                continue
            levels.append(
                LevelDefinition(
                    level_id=f"{suffix}_{stat}",
                    family=f"{session_code}_Session",
                    price=float(price),
                    metadata={"day": f"{day:%Y-%m-%d}", "stat": stat},
                )
            )
    return levels


def build_dynamic_level_registry(
    ranges_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    symbol: str,
    day: pd.Timestamp,
    *,
    session_codes: Iterable[str] = ("AR", "OR", "RR"),
    session_depth: int = 3,
    cluster_tables: Dict[str, pd.DataFrame] | None = None,
    cluster_lookback: int = 12,
    include_current_day_box: bool = True,
    include_current_day_sessions: bool = False,
) -> Dict[str, List[LevelDefinition]]:
    registry: Dict[str, List[LevelDefinition]] = {}
    cluster_levels = _build_cluster_levels_for_day(
        cluster_tables, symbol, day, lookback=cluster_lookback
    )
    registry.update(cluster_levels)
    box_levels = _build_box_levels_for_day(
        bars_df, symbol, day, include_current_day=include_current_day_box
    )
    if box_levels and include_current_day_box:
        registry["AsiaBox"] = box_levels
    for code in session_codes:
        sess_levels = _build_recent_session_levels_from_bars(
            bars_df, symbol, code, day, depth=session_depth
        )
        if not sess_levels:
            sess_levels = _build_recent_session_levels(
                ranges_df,
                symbol,
                code,
                day,
                depth=session_depth,
                include_current_day=include_current_day_sessions,
            )
        if sess_levels:
            registry[f"{code}_Session"] = sess_levels
    return registry


def flatten_events(outputs: Dict[datetime, Dict[str, object]]) -> pd.DataFrame:
    """Collect RejectionEvent dataclasses into a tabular structure."""
    rows = []
    for ts, payload in outputs.items():
        for evt in payload.get("events", []):
            if isinstance(evt, dict):
                # Should not happen, but guard in case of serialization upstream.
                rows.append(evt)
                continue
            rows.append(
                {
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "level_id": evt.level_id,
                    "family": evt.level_family,
                    "price": evt.level_price,
                    "event_type": evt.event_type,
                    "side": evt.side,
                    "strength": evt.strength_score,
                    "is_wick_perfect": evt.is_wick_perfect,
                    "is_close_perfect": evt.is_close_perfect,
                    "ticks_error": evt.ticks_error,
                    "ticks_past": getattr(evt, "ticks_past", 0.0),
                }
            )
    return pd.DataFrame(rows)


def flatten_confluence(outputs: Dict[datetime, Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for ts, payload in outputs.items():
        for conf in payload.get("confluence", []):
            primary_side = getattr(conf.primary_event, "side", 0)
            level_count = len(conf.contributing_levels)
            families = sorted({evt.level_family for evt in getattr(conf, "events", []) or []})
            family_combo = " + ".join(families) if families else ""
            rows.append(
                {
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "primary_level": conf.primary_event.level_id,
                    "direction": primary_side,
                    "score": conf.confluence_score,
                    "levels": family_combo,
                    "level_count": level_count,
                    "is_perfect_cluster": conf.is_perfect_cluster,
                }
            )
    return pd.DataFrame(rows)


def _iter_trading_days(df: pd.DataFrame) -> List[pd.Timestamp]:
    if "day_date" in df.columns:
        return sorted(pd.to_datetime(df["day_date"]).dt.normalize().unique())
    return sorted(df["ts"].apply(_trading_day_for_timestamp).unique())


@app.command()
def run_test(
    instrument: str = typer.Option("ES", help="Instrument prefix (matches data folder)."),
    start_date: str = typer.Option("2024-10-09", help="Inclusive start date (YYYY-MM-DD)."),
    end_date: str = typer.Option("2024-10-20", help="Inclusive end date (YYYY-MM-DD)."),
    session_depth: int = typer.Option(3, help="Number of prior session occurrences to include."),
    tick_size: float = typer.Option(0.25, help="Instrument tick size."),
    zone_ticks: float = typer.Option(4.0, help="Interaction zone tolerance in ticks."),
    confluence_ticks: float = typer.Option(1.0, help="Confluence tolerance in ticks."),
    cluster_lookback: int = typer.Option(12, help="Number of cluster anchors per family to include."),
    include_current_day_box: bool = typer.Option(
        True,
        help="Include current-day Asia box levels (active only for that day).",
    ),
    include_current_day_sessions: bool = typer.Option(
        False,
        help="Include current-day session stats; keep off to mimic live replay without lookahead.",
    ),
    raw_events_only: bool = typer.Option(
        False,
        help="Emit only raw rejection events (skip confluence, hit summaries, and registry dumps).",
    ),
    emit_combined: bool = typer.Option(
        True,
        help="Emit a combined view with events merged to top confluence per timestamp.",
    ),
    output_dir: str = typer.Option(
        str(Path(__file__).parent.parent / "consolidated_live_outputs"),
        help="Folder for CSV exports.",
    ),
) -> None:
    """Replay a window of data through the InteractionEngine and inspect events."""
    typer.echo("=" * 80)
    typer.echo(f"Testing Rejection/Interaction Engine :: {instrument} {start_date} → {end_date}")
    typer.echo("=" * 80)

    bars = load_bars_for_range(instrument, start_date, end_date)
    ranges_df = load_ranges_table(instrument)
    cluster_tables = load_cluster_tables(instrument)
    symbol = str(bars["symbol"].iloc[0]) if "symbol" in bars.columns else instrument.upper()

    # Assign each bar to a trading day that spans 20:30 → 15:55 the next day.
    bars["_trading_day"] = bars["ts"].apply(_trading_day_for_timestamp)

    start_day = _naive_trading_day(start_date)
    end_day = _naive_trading_day(end_date)
    day_list = [d for d in _iter_trading_days(bars) if start_day <= pd.Timestamp(d).normalize() <= end_day]
    if not day_list:
        raise ValueError("No trading days in selected window.")

    all_outputs: Dict[datetime, Dict[str, object]] = {}
    level_rows: List[Dict[str, object]] = []
    for day in day_list:
        day_naive = _naive_trading_day(day)
        # Trading day spans (day-1 @ 20:30) → (day @ 16:00) NY time.
        start_ts = (pd.Timestamp(day_naive) - pd.Timedelta(days=1)).tz_localize(NY_TZ) + pd.Timedelta(
            hours=TRADING_DAY_START[0], minutes=TRADING_DAY_START[1]
        )
        end_ts = pd.Timestamp(day_naive).tz_localize(NY_TZ) + pd.Timedelta(
            hours=TRADING_DAY_END[0], minutes=TRADING_DAY_END[1]
        )
        day_mask = (bars["ts"] >= start_ts) & (bars["ts"] <= end_ts)
        day_bars = bars.loc[day_mask].copy()
        if day_bars.empty:
            continue
        registry = build_dynamic_level_registry(
            ranges_df,
            bars,
            symbol,
            day_naive,
            session_depth=session_depth,
            cluster_tables=cluster_tables,
            cluster_lookback=cluster_lookback,
            include_current_day_box=include_current_day_box,
            include_current_day_sessions=include_current_day_sessions,
        )
        if not registry:
            typer.echo(f"[warn] Day {day.date()} produced no levels; skipping.")
            continue
        for fam, levels in registry.items():
            for lvl in levels:
                level_rows.append(
                    {
                        "day": f"{day_naive:%Y-%m-%d}",
                        "family": fam,
                        "level_id": lvl.level_id,
                        "price": lvl.price,
                        "metadata": json.dumps(lvl.metadata or {}, sort_keys=True),
                    }
                )
        ohlcv = day_bars.set_index("ts")[["open", "high", "low", "close", "volume"]]
        engine = InteractionEngine(
            ohlcv_df=ohlcv,
            level_registry=registry,
            tick_size=tick_size,
            zone_tolerance_ticks=zone_ticks,
            confluence_tolerance_ticks=confluence_ticks,
        )
        typer.echo(
            f"[engine] {day_naive.date()} :: {len(ohlcv)} bars, "
            f"{sum(len(v) for v in registry.values())} levels."
        )
        day_outputs = engine.run()
        all_outputs.update(day_outputs)

    if not all_outputs:
        typer.echo("[result] No outputs generated.")
        return

    event_df = flatten_events(all_outputs)
    conf_df = pd.DataFrame() if raw_events_only else flatten_confluence(all_outputs)

    if event_df.empty:
        typer.echo("[result] No rejection events detected in the selected window.")
    else:
        typer.echo(
            f"[result] Captured {len(event_df):,} rejection events "
            f"({event_df['event_type'].value_counts().to_dict()})."
        )

    if not conf_df.empty:
        typer.echo(
            f"[result] Identified {len(conf_df):,} confluence clusters; "
            f"mean score={conf_df['score'].mean():.2f}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    stem = f"{instrument.upper()}_{start_date}_to_{end_date}"
    if not event_df.empty:
        event_path = out_dir / f"rejection_events_{stem}.csv"
        event_df.to_csv(event_path, index=False)
        typer.echo(f"[write] Rejection events → {event_path}")
        if not raw_events_only:
            touched = (
                event_df.groupby(["level_id", "family", "price"])
                .agg(
                    first_seen=("timestamp", "min"),
                    last_seen=("timestamp", "max"),
                    hits=("timestamp", "count"),
                    max_strength=("strength", "max"),
                    mean_strength=("strength", "mean"),
                    min_ticks_error=("ticks_error", "min"),
                    max_ticks_error=("ticks_error", "max"),
                )
                .reset_index()
                .sort_values(["hits", "max_strength"], ascending=[False, False])
            )
            touched_path = out_dir / f"touched_levels_{stem}.csv"
            touched.to_csv(touched_path, index=False)
            typer.echo(f"[write] Level hit summary → {touched_path}")
    if not raw_events_only and not conf_df.empty:
        conf_path = out_dir / f"rejection_confluence_{stem}.csv"
        conf_df.to_csv(conf_path, index=False)
        typer.echo(f"[write] Confluence clusters → {conf_path}")
    if emit_combined and not conf_df.empty and not event_df.empty:
        best_conf = (
            conf_df.sort_values("score", ascending=False)
            .groupby("timestamp")
            .head(1)
            .rename(columns={"levels": "confluence_levels", "score": "confluence_score"})
        )
        combined = event_df.merge(best_conf, on="timestamp", how="left")
        combined_path = out_dir / f"rejection_events_combined_{stem}.csv"
        combined.to_csv(combined_path, index=False)
        typer.echo(f"[write] Combined events → {combined_path}")
    if not raw_events_only and level_rows:
        level_df = pd.DataFrame(level_rows)
        level_path = out_dir / f"levels_{stem}.csv"
        level_df.to_csv(level_path, index=False)
        typer.echo(f"[write] Level registry → {level_path}")

    typer.echo("\nSample events:")
    if not event_df.empty:
        typer.echo(event_df.head(10).to_string(index=False))
    else:
        typer.echo("  (none)")

    if not raw_events_only:
        typer.echo("\nSample confluence clusters:")
        if not conf_df.empty:
            typer.echo(conf_df.head(10).to_string(index=False))
        else:
            typer.echo("  (none)")

    typer.echo("\nDone.")


if __name__ == "__main__":
    app()
