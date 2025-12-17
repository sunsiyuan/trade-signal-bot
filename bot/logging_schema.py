"""Structured signal logging helpers (schema v2).

Schema 2.4 removes the settings symbol along with prior removal of order book
line items and tracked symbol lists to avoid logging sensitive details while
keeping the rest of the market snapshot intact.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config import Settings
from .models import MarketSnapshot
from .signal_engine import TradeSignal


SCHEMA_VERSION = "2.4"


def _safe_iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _plan_dict(plan):
    if plan is None:
        return None
    if is_dataclass(plan):
        return asdict(plan)
    if isinstance(plan, dict):
        return plan
    return None


def _tf_to_timedelta(tf: str) -> timedelta:
    tf = tf.lower()
    if tf.endswith("m"):
        return timedelta(minutes=int(tf[:-1]))
    if tf.endswith("h"):
        return timedelta(hours=int(tf[:-1]))
    raise ValueError(f"Unsupported timeframe: {tf}")


def _timeframe_block(tf) -> Dict[str, Any]:
    price_last = getattr(tf, "price_last", None)
    close_price = getattr(tf, "close", None)

    return {
        "tf": tf.timeframe,
        "last_candle_open_utc": _safe_iso(tf.last_candle_open_utc),
        "last_candle_close_utc": _safe_iso(tf.last_candle_close_utc),
        "is_last_candle_closed": tf.is_last_candle_closed,
        "bars_used": tf.bars_used,
        "lookback_window": tf.lookback_window,
        "missing_bars_count": tf.missing_bars_count,
        "gap_list": tf.gap_list,
        "prices": {
            "close": close_price,
            "price_last": price_last if price_last is not None else close_price,
            "price_mid": getattr(tf, "price_mid", None),
            "typical_price": getattr(tf, "typical_price", None),
            "mark": getattr(tf, "mark_price", None),
            "index": getattr(tf, "index_price", None),
            "return_last": getattr(tf, "return_last", None),
        },
        "volume": getattr(tf, "volume", None),
        "trend_label": getattr(tf, "trend_label", None),
        "volatility": {
            "atr": tf.atr,
            "atr_rel": tf.atr_rel,
            "tr_last": tf.tr_last,
        },
        "indicators": {
            "rsi_6": tf.rsi6,
            "rsi_12": tf.rsi12,
            "rsi_24": tf.rsi24,
            "macd_value": tf.macd,
            "macd_signal": tf.macd_signal,
            "macd_hist": tf.macd_hist,
            "ma7": tf.ma7,
            "ma25": tf.ma25,
            "ma99": tf.ma99,
            "ma_angle": getattr(tf, "ma_angle", None),
        },
    }


def _settings_snapshot(settings: Settings) -> Dict[str, Any]:
    return {
        "timeframes": {
            "tf_4h": settings.tf_4h,
            "tf_1h": settings.tf_1h,
            "tf_15m": settings.tf_15m,
        },
        "thresholds": {
            "min_confidence": getattr(settings, "min_confidence", None),
            "signal_confidence_threshold": getattr(
                settings, "signal_confidence_threshold", None
            ),
        },
        "notification": getattr(settings, "notification", None),
    }


def build_signal_event(
    snapshot: MarketSnapshot,
    signal: TradeSignal,
    settings: Settings,
    exchange_id: str = "",
    run_ts: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build a schema v2 log event for a signal run."""

    run_ts = run_ts or datetime.now(timezone.utc)

    tf_map = {
        snapshot.tf_15m.timeframe: snapshot.tf_15m,
        snapshot.tf_1h.timeframe: snapshot.tf_1h,
        snapshot.tf_4h.timeframe: snapshot.tf_4h,
    }
    tf_list = [settings.tf_15m, settings.tf_1h, settings.tf_4h]
    anchor_tf = settings.tf_1h
    anchor = tf_map.get(anchor_tf)
    anchor_close = anchor.last_candle_close_utc if anchor else None

    tf_close_delta_sec: Dict[str, Optional[float]] = {}
    alignment_ok = True
    alignment_reason = ""
    for tf_name, tf in tf_map.items():
        if anchor_close and tf.last_candle_close_utc:
            delta = (tf.last_candle_close_utc - anchor_close).total_seconds()
            tf_close_delta_sec[tf_name] = delta
            if abs(delta) > 1e-6:
                alignment_ok = False
                alignment_reason = f"{tf_name} close misaligned vs {anchor_tf}"
        else:
            tf_close_delta_sec[tf_name] = None
            alignment_ok = False
            alignment_reason = alignment_reason or "missing_close_time"

        if not tf.is_last_candle_closed:
            alignment_ok = False
            alignment_reason = alignment_reason or f"{tf_name} last candle open"

    latest_close = max(
        (tf.last_candle_close_utc for tf in tf_map.values() if tf.last_candle_close_utc),
        default=run_ts,
    )
    data_latency_sec = (run_ts - latest_close).total_seconds()

    anchor_delta = _tf_to_timedelta(anchor_tf)
    valid_until_utc = (anchor_close + anchor_delta) if anchor_close else None

    debug_scores = signal.debug_scores or {}
    long_score = debug_scores.get("long")
    short_score = debug_scores.get("short")
    best_score = None
    if long_score is not None and short_score is not None:
        best_score = max(long_score, short_score)

    try:
        source_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        source_commit = ""

    def _action() -> str:
        if signal.direction == "long":
            return "LONG"
        if signal.direction == "short":
            return "SHORT"
        return "NO_TRADE"

    return {
        "schema_version": SCHEMA_VERSION,
        "event_type": "signal",
        "run": {
            "ts_utc": run_ts.isoformat(),
            "valid_until_utc": _safe_iso(valid_until_utc),
            "data_latency_sec": data_latency_sec,
            "tf_list": tf_list,
            "source_commit": source_commit,
        },
        "market": {
            "symbol": snapshot.symbol,
            "exchange": exchange_id,
            "price": getattr(snapshot.deriv, "mark_price", None)
            or getattr(snapshot.tf_15m, "close", None),
            "snapshot_ts_utc": _safe_iso(getattr(snapshot, "ts", None)),
            "settings_snapshot": _settings_snapshot(settings),
        },
        "alignment": {
            "anchor_tf": anchor_tf,
            "anchor_close_utc": _safe_iso(anchor_close),
            "tf_close_delta_sec": tf_close_delta_sec,
            "alignment_ok": alignment_ok,
            "alignment_reason": alignment_reason or "ok",
        },
        "timeframes": [_timeframe_block(tf_map[tf]) for tf in tf_list if tf in tf_map],
        "derivatives": {
            "funding": snapshot.deriv.funding,
            "open_interest": snapshot.deriv.open_interest,
            "oi_change_24h": snapshot.deriv.oi_change_24h,
            "mark_price": getattr(snapshot.deriv, "mark_price", None),
            "liquidity": {
                "ask_wall_size": snapshot.deriv.ask_wall_size,
                "bid_wall_size": snapshot.deriv.bid_wall_size,
                "ask_to_bid_ratio": snapshot.deriv.ask_to_bid_ratio,
                "has_large_ask_wall": snapshot.deriv.has_large_ask_wall,
                "has_large_bid_wall": snapshot.deriv.has_large_bid_wall,
                "comment": snapshot.deriv.liquidity_comment,
            },
        },
        "regime": {
            "regime": snapshot.regime,
            "reason": snapshot.regime_reason,
            "rsidev": snapshot.rsidev,
            "atrrel": snapshot.atrrel,
            "rsi_15m": getattr(snapshot, "rsi_15m", None),
            "rsi_1h": getattr(snapshot, "rsi_1h", None),
        },
        "signal": {
            "action": _action(),
            "direction": signal.direction,
            "setup_type": signal.setup_type,
            "trade_confidence": signal.trade_confidence,
            "edge_confidence": signal.edge_confidence,
            "edge_type": getattr(signal, "edge_type", None),
            "execution_intent": _plan_dict(getattr(signal, "execution_intent", None)),
            "conditional_plan": _plan_dict(signal.conditional_plan),
            "entry": signal.entry,
            "tp": {
                "tp1": signal.tp1,
                "tp2": signal.tp2,
                "tp3": signal.tp3,
            },
            "sl": signal.sl,
            "core_position_pct": signal.core_position_pct,
            "add_position_pct": signal.add_position_pct,
            "rejected_reasons": signal.rejected_reasons or [],
            "thresholds_snapshot": signal.thresholds_snapshot or {},
            "valid_until_utc": _safe_iso(valid_until_utc),
        },
        "debug": {
            "scores": debug_scores,
            "best_score": best_score,
            "long_score": long_score,
            "short_score": short_score,
            "tf_close_delta_sec": tf_close_delta_sec,
            "conditional_plan_debug": getattr(signal, "conditional_plan_debug", None),
        },
    }


def write_jsonl_event(event: Dict[str, Any], path: str) -> None:
    """Append a JSONL event to the given path, creating parents if needed."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


__all__ = ["SCHEMA_VERSION", "build_signal_event", "write_jsonl_event"]
