#!/usr/bin/env python3
"""
Test script: Range Session Metrics (v3) from ConsolidatedLiveEngine

Loads ES bars for 2025-05-20 to 2025-05-27, feeds them into ModelAdaptersV3,
and exports RangeSessionOut fields for inspection.
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from world.fusion import Bar
from world.fusion import ModelAdaptersV3


def load_es_bars(start_date: str, end_date: str) -> pd.DataFrame:
    data_dir = Path(__file__).parent.parent / "data" / "processed" / "ES"
    parquet_path = data_dir / "ES_bars_with_sessions.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing ES bars file: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    df["datetime"] = pd.to_datetime(df["ts"], unit="s")
    df["date"] = df["datetime"].dt.date

    s = datetime.fromisoformat(start_date).date()
    e = datetime.fromisoformat(end_date).date()
    return df[(df["date"] >= s) & (df["date"] <= e)].copy()


def to_bars(df: pd.DataFrame) -> list:
    bars = []
    for _, r in df.iterrows():
        ts = int(r["ts"].timestamp()) if isinstance(r["ts"], pd.Timestamp) else int(r["ts"])
        bars.append(
            Bar(
                ts=ts,
                symbol=str(r["symbol"]),
                o=float(r["open"]),
                h=float(r["high"]),
                l=float(r["low"]),
                c=float(r["close"]),
                v=float(r.get("volume", 1000)),
                session=str(r["session"]),
            )
        )
    return bars


def main():
    start = "2025-05-20"
    end = "2025-05-27"
    print(f"Loading ES bars {start} → {end} ...")
    df = load_es_bars(start, end)
    print(f"Bars loaded: {len(df)}")

    bars = to_bars(df)
    data_dir = Path(__file__).parent.parent / "data"
    adapters = ModelAdaptersV3(str(data_dir))

    rows = []
    for i, bar in enumerate(bars):
        # Call adapter first (it ingests internally). Then fetch the cached consolidated snapshot.
        rs = adapters.get_range_session_metrics_v3(bar)
        out = adapters._last_out(bar) or {}
        rows.append({
            "ts": out.get("ts", bar.ts),
            "timestamp": out.get("timestamp"),
            "symbol": bar.symbol,
            "session": bar.session,
            "day_number": rs.day_number,
            "box_std": rs.box_std,
            "confirm_side": rs.confirm_side,
            "is_post_confirm": rs.is_post_confirm,
            "confirm_idx": rs.confirm_candle_index,
            "p_false_session": rs.p_false_session,
            "p_m7_ret_sd": rs.p_m7_ret_sd,
            "p_m7_time": rs.p_m7_time,
            "p_max_ext_sd": rs.p_max_ext_sd,
            "p_max_ext_time": rs.p_max_ext_time,
            "p_max_ret_sd": rs.p_max_ret_sd,
            "p_max_ret_time": rs.p_max_ret_time,
            "current_ext_sd": rs.current_ext_sd,
            "current_ret_sd": rs.current_ret_sd,
            "session_close_std": rs.session_close_std,
            "p_session_close_std": rs.p_session_close_std,
            "max_ext_candle_index": rs.max_ext_candle_index,
            "max_ret_candle_index": rs.max_ret_candle_index,
        })

        if (i + 1) % 500 == 0:
            print(f"Processed {i+1}/{len(bars)} bars")

    out_dir = Path(__file__).parent.parent / "consolidated_live_outputs"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "range_session_metrics_v3_2025-05-20_2025-05-27.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✓ Exported {len(rows)} rows to {out_csv}")

    # Show a small preview
    print(pd.DataFrame(rows).head(10).to_string())


if __name__ == "__main__":
    main()

