from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
from typing import Any, Dict, List, Mapping, Optional, Sequence, Literal

import numpy as np
import pandas as pd


RejectionType = Literal["Wick_Reject", "Close_Pin", "Breakout", "Test"]


@dataclass(slots=True, frozen=True)
class LevelDefinition:
    """Normalized description of a tradable level."""

    level_id: str
    family: str
    price: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RejectionEvent:
    """Atomic interaction between a single candle and a level."""

    timestamp: datetime
    level_id: str
    level_family: str
    level_price: float
    event_type: RejectionType
    side: int  # 1 support (from above), -1 resistance (from below)
    strength_score: float
    is_wick_perfect: bool
    is_close_perfect: bool
    ticks_error: float
    ticks_past: float


@dataclass(slots=True)
class ConfluenceEvent:
    """Aggregated view for tightly clustered level interactions."""

    timestamp: datetime
    primary_event: RejectionEvent
    contributing_levels: List[str]
    confluence_score: float
    is_perfect_cluster: bool
    events: List[RejectionEvent] = field(default_factory=list)


class InteractionEngine:
    """
    Translate raw OHLCV bars into structured rejection events.

    Parameters
    ----------
    ohlcv_df:
        Pandas DataFrame indexed by datetime with open/high/low/close/volume columns.
    level_registry:
        Mapping of level families → iterables describing the price levels to monitor.
    tick_size:
        Smallest tradable increment. Used to convert tolerances into price space.
    zone_tolerance_ticks:
        Number of ticks allowed away from a level to still consider it an interaction.
    """

    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
        level_registry: Mapping[str, Any] | Sequence[Any] | None,
        *,
        tick_size: float = 0.25,
        zone_tolerance_ticks: float = 4.0,
        confluence_tolerance_ticks: float = 1.0,
        volume_lookback: int = 20,
    ) -> None:
        if ohlcv_df is None or ohlcv_df.empty:
            raise ValueError("InteractionEngine requires a non-empty OHLCV dataframe.")

        self.tick_size = float(tick_size) if tick_size > 0 else 0.25
        self.zone_tolerance_ticks = float(zone_tolerance_ticks)
        self.confluence_tolerance_ticks = float(confluence_tolerance_ticks)
        self.volume_lookback = max(1, int(volume_lookback))

        self._df = self._normalize_dataframe(ohlcv_df)
        self._levels = self._normalize_level_registry(level_registry)
        self._level_lookup = {lvl.level_id: lvl for lvl in self._levels}
        self._level_windows = self._extract_level_windows(self._levels)
        self._recent_hits: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=16))
        self._hit_lookback = 10
        self._zone_tolerance_price = self.zone_tolerance_ticks * self.tick_size
        # Allow wick rejections that probe a bit deeper than the base zone, but keep this tight.
        self._wick_reject_tolerance_price = max(self._zone_tolerance_price, self.tick_size * 2)
        self._confluence_tolerance_price = self.confluence_tolerance_ticks * self.tick_size
        self._rolling_vol = (
            self._df["volume"].rolling(self.volume_lookback, min_periods=1).mean()
        )
        self._special_levels = self._resolve_special_levels()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def run(self) -> Dict[datetime, Dict[str, Any]]:
        """
        Iterate through the OHLCV stream and emit events, confluence, and agent state.
        """
        outputs: Dict[datetime, Dict[str, Any]] = {}
        bar_idx = 0

        for ts, row in self._df.iterrows():
            ts_dt = self._to_datetime(ts)
            rolling_vol = float(self._rolling_vol.loc[ts])
            events = self._scan_bar(ts_dt, row, bar_idx, rolling_vol)
            confluence_events = self.aggregate_confluence(events)
            distances = self._compute_distances(float(row["close"]))
            high_vol = self._is_high_volatility(float(row["volume"]), rolling_vol)
            agent_state = build_agent_state(
                ts_dt,
                events,
                confluence_events,
                distances=distances,
                is_high_volatility=high_vol,
            )

            outputs[ts_dt] = {
                "events": events,
                "confluence": confluence_events,
                "agent_state": agent_state,
            }
            bar_idx += 1

        return outputs

    def aggregate_confluence(self, events: List[RejectionEvent]) -> List[ConfluenceEvent]:
        """
        Group events whose level prices sit within the confluence tolerance.
        """
        if not events:
            return []

        events = sorted(events, key=lambda e: e.level_price)
        groups: List[List[RejectionEvent]] = []
        curr: List[RejectionEvent] = [events[0]]

        for event in events[1:]:
            if abs(event.level_price - curr[-1].level_price) <= self._confluence_tolerance_price:
                curr.append(event)
            else:
                groups.append(curr)
                curr = [event]
        groups.append(curr)

        confluence_events: List[ConfluenceEvent] = []
        for grp in groups:
            if not grp:
                continue
            primary = max(grp, key=lambda e: e.strength_score)
            spread = max(e.level_price for e in grp) - min(e.level_price for e in grp)
            tight_bonus = 0.0
            if self._confluence_tolerance_price > 0:
                tight_bonus = max(
                    0.0,
                    (self._confluence_tolerance_price - spread) / self._confluence_tolerance_price,
                ) * 0.2
            score = min(2.0, sum(e.strength_score for e in grp) + tight_bonus)
            is_perfect = any(e.is_wick_perfect or e.is_close_perfect for e in grp)
            confluence_events.append(
                ConfluenceEvent(
                    timestamp=primary.timestamp,
                    primary_event=primary,
                    contributing_levels=[e.level_id for e in grp],
                    confluence_score=score,
                    is_perfect_cluster=is_perfect,
                    events=list(grp),
                )
            )

        return confluence_events

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _scan_bar(
        self,
        timestamp: datetime,
        row: pd.Series,
        bar_idx: int,
        rolling_vol: float,
    ) -> List[RejectionEvent]:
        events: List[RejectionEvent] = []
        for level in self._levels:
            if self._level_windows and not self._is_level_active(level.level_id, timestamp):
                continue
            evt = self._evaluate_interaction(timestamp, row, level, bar_idx, rolling_vol)
            if evt is not None:
                events.append(evt)
        return events

    def _evaluate_interaction(
        self,
        timestamp: datetime,
        row: pd.Series,
        level: LevelDefinition,
        bar_idx: int,
        rolling_vol: float,
    ) -> Optional[RejectionEvent]:
        open_px = float(row["open"])
        high_px = float(row["high"])
        low_px = float(row["low"])
        close_px = float(row["close"])
        volume = float(row["volume"])
        price = float(level.price)

        wick_proximity = min(abs(high_px - price), abs(low_px - price))
        close_proximity = abs(close_px - price)
        # Treat perfect wick as within 0.05 points of the level.
        perfect_wick_tol = 0.05
        # Allow wick touches within one tick for interaction.
        touch_tol = self.tick_size * 0.5
        wick_hits_level = (
            (high_px >= price >= low_px)
            and (min(abs(high_px - price), abs(low_px - price)) <= touch_tol)
        )
        is_wick_perfect = bool(
            np.isclose(high_px, price, atol=perfect_wick_tol)
            or np.isclose(low_px, price, atol=perfect_wick_tol)
        )
        is_close_perfect = bool(np.isclose(close_px, price, atol=self.tick_size * 0.1))
        # Directional wick rejection definitions (must touch the level).
        wick_reject_down = wick_hits_level and (open_px < price) and (high_px >= price) and (close_px < price)
        wick_reject_up = wick_hits_level and (open_px > price) and (low_px <= price) and (close_px > price)
        wick_touch = wick_reject_down or wick_reject_up
        # Require an actual touch: either wick touches within the tight tolerance or close is perfect.
        interacted = wick_touch or is_wick_perfect or is_close_perfect
        if not interacted:
            return None

        metrics = [
            ("high", abs(high_px - price), high_px),
            ("low", abs(low_px - price), low_px),
            ("close", abs(close_px - price), close_px),
        ]
        metric_kind, metric_diff, metric_value = min(metrics, key=lambda m: m[1])
        ticks_error = metric_diff / self.tick_size if self.tick_size else metric_diff

        if metric_kind == "low":
            side = 1
        elif metric_kind == "high":
            side = -1
        else:
            side = 1 if close_px >= price else -1

        event_type: RejectionType = "Test"
        penetration = close_px - price
        # Allow wick-cross rejections slightly beyond the normal zone, but do not
        # force perfect flags unless the wick truly matches the level.
        if wick_touch:
            event_type = "Wick_Reject"
        elif is_wick_perfect:
            event_type = "Wick_Reject"
        elif is_close_perfect:
            event_type = "Close_Pin"
        elif (side == -1 and penetration > self._zone_tolerance_price) or (
            side == 1 and penetration < -self._zone_tolerance_price
        ):
            event_type = "Breakout"
        elif (open_px - price) * (close_px - price) < 0:
            # Closed across the level within the tolerance window
            event_type = "Breakout"

        score = 0.5
        if is_wick_perfect:
            score += 0.3
        if is_close_perfect:
            score += 0.2
        if self._volume_boost(volume, rolling_vol):
            score += 0.1

        hit_count = self._register_hit(level.level_id, bar_idx)
        if hit_count > 3:
            score -= min(0.3, 0.05 * (hit_count - 3))

        if event_type == "Test":
            score = max(0.0, score - 0.1)

        score = float(max(0.0, min(1.0, score)))
        ticks_past = 0.0
        if wick_touch:
            penetration_dist = max(high_px - price, price - low_px, 0.0)
            ticks_past = penetration_dist / self.tick_size if self.tick_size else penetration_dist
            # Do not call it "perfect" if the wick traveled far past the level.
            if ticks_past > self.zone_tolerance_ticks:
                is_wick_perfect = False

        return RejectionEvent(
            timestamp=timestamp,
            level_id=level.level_id,
            level_family=level.family,
            level_price=price,
            event_type=event_type,
            side=side,
            strength_score=score,
            is_wick_perfect=is_wick_perfect,
            is_close_perfect=is_close_perfect,
            ticks_error=float(round(ticks_error, 4)),
            ticks_past=float(round(ticks_past, 4)),
        )

    def _volume_boost(self, volume: float, avg_volume: float) -> bool:
        if not np.isfinite(volume) or not np.isfinite(avg_volume):
            return False
        if avg_volume <= 0:
            return False
        return volume > avg_volume

    def _is_high_volatility(self, volume: float, avg_volume: float) -> bool:
        if not np.isfinite(volume) or not np.isfinite(avg_volume):
            return False
        if avg_volume <= 0:
            return False
        return volume >= avg_volume * 1.5

    def _register_hit(self, level_id: str, bar_idx: int) -> int:
        dq = self._recent_hits[level_id]
        dq.append(bar_idx)
        while dq and (bar_idx - dq[0]) >= self._hit_lookback:
            dq.popleft()
        return len(dq)

    def _compute_distances(self, close_px: float) -> Dict[str, float]:
        distances: Dict[str, float] = {}
        monthly_high = self._special_levels.get("monthly_high")
        if monthly_high:
            distances["dist_to_monthly_high"] = abs(monthly_high.price - close_px)
        asia_mid = self._special_levels.get("asia_mid")
        if asia_mid:
            distances["dist_to_asia_mid"] = abs(asia_mid.price - close_px)
        return distances

    def _resolve_special_levels(self) -> Dict[str, LevelDefinition]:
        specials: Dict[str, LevelDefinition] = {}
        monthly_levels = [lvl for lvl in self._levels if "month" in lvl.family.lower()]
        if monthly_levels:
            specials["monthly_high"] = max(monthly_levels, key=lambda lvl: lvl.price)

        asia_levels = [lvl for lvl in self._levels if "asia" in lvl.family.lower()]
        mid_candidates = [lvl for lvl in asia_levels if "mid" in lvl.level_id.lower()]
        if not mid_candidates and asia_levels:
            # Fallback to the level closest to the median of the Asia family.
            median_price = np.median([lvl.price for lvl in asia_levels])
            mid_candidates = [
                min(asia_levels, key=lambda lvl: abs(lvl.price - median_price))
            ]
        if mid_candidates:
            specials["asia_mid"] = mid_candidates[0]
        return specials

    # --------------------------------------------------------------- #
    # Normalization utilities
    # --------------------------------------------------------------- #
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("OHLCV dataframe index must be a DatetimeIndex.")
        data = data.sort_index()

        col_lower = {c.lower(): c for c in data.columns}
        alias_map = {
            "open": ("open", "o"),
            "high": ("high", "h"),
            "low": ("low", "l"),
            "close": ("close", "c"),
            "volume": ("volume", "vol", "v"),
        }
        rename: Dict[str, str] = {}
        for target, aliases in alias_map.items():
            if target in data.columns:
                continue
            for alias in aliases:
                if alias in col_lower:
                    rename[col_lower[alias]] = target
                    break

        if rename:
            data = data.rename(columns=rename)

        required = {"open", "high", "low", "close"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"OHLCV dataframe missing columns: {sorted(missing)}")

        if "volume" not in data.columns:
            data["volume"] = 0.0

        for col in ["open", "high", "low", "close", "volume"]:
            series = pd.to_numeric(data[col], errors="coerce")
            data[col] = series.ffill().fillna(0.0)

        return data

    def _normalize_level_registry(
        self, registry: Mapping[str, Any] | Sequence[Any] | None
    ) -> List[LevelDefinition]:
        if registry is None:
            return []

        entries: List[LevelDefinition] = []
        if isinstance(registry, Mapping):
            for family, payload in registry.items():
                entries.extend(self._coerce_family_levels(str(family), payload))
        elif isinstance(registry, Sequence) and not isinstance(registry, (str, bytes)):
            for obj in registry:
                lvl = self._coerce_level(obj, "custom")
                if lvl:
                    entries.append(lvl)
        else:
            raise TypeError("level_registry must be a mapping or iterable of level specs.")

        uniq: Dict[str, LevelDefinition] = {}
        for lvl in entries:
            if lvl and np.isfinite(lvl.price):
                uniq[lvl.level_id] = lvl
        return list(uniq.values())

    def _coerce_family_levels(self, family: str, payload: Any) -> List[LevelDefinition]:
        if isinstance(payload, Mapping) and {"price", "level_id"}.issubset(payload.keys()):
            lvl = self._coerce_level(payload, family)
            return [lvl] if lvl else []

        levels: List[LevelDefinition] = []
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                if isinstance(value, Mapping):
                    spec = {"level_id": key, "family": family, **value}
                else:
                    spec = {"level_id": key, "family": family, "price": value}
                lvl = self._coerce_level(spec, family)
                if lvl:
                    levels.append(lvl)
            return levels

        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            for obj in payload:
                lvl = self._coerce_level(obj, family)
                if lvl:
                    levels.append(lvl)
            return levels

        lvl = self._coerce_level(payload, family)
        return [lvl] if lvl else []

    def _coerce_level(self, obj: Any, fallback_family: str) -> Optional[LevelDefinition]:
        if obj is None:
            return None
        if isinstance(obj, LevelDefinition):
            return obj
        if isinstance(obj, Mapping):
            raw_id = obj.get("level_id") or obj.get("id") or obj.get("name")
            price_val = obj.get("price") or obj.get("level_price") or obj.get("value")
            if price_val is None:
                return None
            try:
                price = float(price_val)
            except (TypeError, ValueError):
                return None
            level_id = str(raw_id) if raw_id else f"{fallback_family}_{abs(hash((price, fallback_family))) % 10_000}"
            family = str(obj.get("family") or fallback_family)
            metadata = {
                k: v
                for k, v in obj.items()
                if k not in {"level_id", "id", "name", "price", "level_price", "value", "family"}
            }
            return LevelDefinition(level_id=level_id, family=family, price=price, metadata=metadata)
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)) and len(obj) >= 2:
            level_id = str(obj[0])
            try:
                price = float(obj[1])
            except (TypeError, ValueError):
                return None
            metadata = obj[2] if len(obj) > 2 and isinstance(obj[2], Mapping) else {}
            return LevelDefinition(level_id=level_id, family=fallback_family, price=price, metadata=dict(metadata))
        if isinstance(obj, (float, int)):
            return LevelDefinition(
                level_id=f"{fallback_family}_{abs(hash(obj)) % 10_000}",
                family=fallback_family,
                price=float(obj),
                metadata={},
            )
        return None

    # ------------------------------------------------------------------ #
    # Time window helpers for levels
    # ------------------------------------------------------------------ #
    def _extract_level_windows(self, levels: List[LevelDefinition]) -> Dict[str, tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]:
        """
        Allow levels to define validity windows via metadata keys:
        - valid_from / valid_to: ISO-parseable timestamps. If tz-naive, assume same tz as OHLCV index.
        """
        windows: Dict[str, tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = {}
        idx_tz = self._df.index.tz
        for lvl in levels:
            meta = lvl.metadata or {}
            start_raw = meta.get("valid_from") or meta.get("from")
            end_raw = meta.get("valid_to") or meta.get("to")
            if not start_raw and not end_raw:
                continue
            def _coerce(value):
                if value is None:
                    return None
                ts = pd.to_datetime(value, errors="coerce")
                if ts is pd.NaT:
                    return None
                if ts.tzinfo is None and idx_tz is not None:
                    ts = ts.tz_localize(idx_tz)
                return ts
            start_ts = _coerce(start_raw)
            end_ts = _coerce(end_raw)
            windows[lvl.level_id] = (start_ts, end_ts)
        return windows

    def _is_level_active(self, level_id: str, timestamp: datetime) -> bool:
        window = self._level_windows.get(level_id)
        if not window:
            return True
        start_ts, end_ts = window
        ts = timestamp if isinstance(timestamp, datetime) else self._to_datetime(timestamp)
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        if start_ts and ts < start_ts.to_pydatetime():
            return False
        if end_ts and ts > end_ts.to_pydatetime():
            return False
        return True

    @staticmethod
    def _to_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        raise TypeError(f"Unsupported timestamp type: {type(value)!r}")


def build_agent_state(
    timestamp: datetime,
    events: Sequence[RejectionEvent],
    confluence_events: Sequence[ConfluenceEvent],
    *,
    distances: Optional[Dict[str, float]] = None,
    is_high_volatility: bool = False,
) -> Dict[str, Any]:
    """
    Convert low-level interaction signals into an agent-friendly JSON payload.
    """
    ts = timestamp if isinstance(timestamp, datetime) else InteractionEngine._to_datetime(timestamp)

    primary_event = max(events, key=lambda e: e.strength_score) if events else None
    precision_block = {
        "is_wick_perfect": False,
        "is_close_perfect": False,
        "ticks_error": 0.0,
    }
    primary_block: Dict[str, Any] | None = None

    if primary_event:
        precision_block = {
            "is_wick_perfect": primary_event.is_wick_perfect,
            "is_close_perfect": primary_event.is_close_perfect,
            "ticks_error": primary_event.ticks_error,
        }
        primary_block = {
            "level_id": primary_event.level_id,
            "family": primary_event.level_family,
            "type": primary_event.event_type,
            "side": primary_event.side,
            "precision": precision_block,
            "strength": round(primary_event.strength_score, 4),
        }
    else:
        primary_block = {
            "level_id": None,
            "family": None,
            "type": None,
            "side": 0,
            "precision": precision_block,
            "strength": 0.0,
        }

    if confluence_events:
        best_confluence = max(confluence_events, key=lambda c: c.confluence_score)
        confluence_block = {
            "is_confluence": True,
            "score": round(best_confluence.confluence_score, 4),
            "levels": best_confluence.contributing_levels,
        }
    else:
        confluence_block = {"is_confluence": False, "score": 0.0, "levels": []}

    payload = {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "state_flags": {
            "has_active_rejection": bool(events),
            "is_high_volatility": bool(is_high_volatility),
        },
        "primary_interaction": primary_block,
        "confluence_context": confluence_block,
        "distances": distances or {},
    }
    return payload


__all__ = [
    "LevelDefinition",
    "RejectionEvent",
    "ConfluenceEvent",
    "InteractionEngine",
    "build_agent_state",
]
