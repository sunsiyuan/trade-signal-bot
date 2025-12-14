"""Conditional plan builder anchored on 4H structure."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from .models import MarketSnapshot


@dataclass
class PlanContext:
    snap: MarketSnapshot
    direction: str
    edge_confidence: float
    trade_confidence: float
    long_score: float
    short_score: float
    min_confidence: float
    watch_edge_conf: float

    @property
    def direction_gap(self) -> float:
        return abs(self.long_score - self.short_score)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _next_4h_close(snap: MarketSnapshot) -> Optional[datetime]:
    close = snap.tf_4h.last_candle_close_utc
    if not close:
        return None
    return close + timedelta(hours=4)


def _base_level(snap: MarketSnapshot, direction: str) -> Optional[float]:
    tf4h = snap.tf_4h
    tf1h = snap.tf_1h
    if direction == "short":
        for candidate in (getattr(tf4h, "recent_high", None), tf4h.ma25, tf1h.ma25):
            if candidate is not None:
                return candidate
    else:
        for candidate in (getattr(tf4h, "recent_low", None), tf4h.ma25, tf1h.ma25):
            if candidate is not None:
                return candidate
    return None


def _entry_zone_trending(snap: MarketSnapshot, direction: str) -> Optional[Dict[str, float]]:
    tf4h, tf1h, tf15 = snap.tf_4h, snap.tf_1h, snap.tf_15m
    base = _base_level(snap, direction)
    if base is None:
        return None

    band = 0.8 * tf15.atr
    if direction == "short":
        low = base - band
        high = base + 0.5 * band
        max_high = tf4h.ma25 + 0.5 * tf1h.atr
        high = min(high, max_high)
    else:
        low = base - 0.5 * band
        high = base + band
        min_low = tf4h.ma25 - 0.5 * tf1h.atr
        low = max(low, min_low)

    if low >= high:
        return None

    price = tf15.close
    if low <= price <= high:
        return None

    return {"low": low, "high": high, "base": base, "band": band}


def _entry_zone_ranging(snap: MarketSnapshot, direction: str) -> Optional[Dict[str, float]]:
    tf4h, tf15 = snap.tf_4h, snap.tf_15m
    atr = tf15.atr
    price = tf15.close
    if direction == "long":
        base = getattr(tf4h, "recent_low", None) or tf4h.ma25 - atr
        low = base
        high = base + atr
    else:
        base = getattr(tf4h, "recent_high", None) or tf4h.ma25 + atr
        low = base - atr
        high = base

    if low >= high:
        return None

    if low <= price <= high:
        return None

    return {"low": low, "high": high, "base": base, "band": atr}


def _entry_logic(direction: str) -> str:
    if direction == "short":
        return "15m 收盘进入区间且 MACD_hist <= 0"
    return "15m 收盘进入区间且 MACD_hist >= 0"


def _risk_block(entry_low: float, entry_high: float, base: float, snap: MarketSnapshot, direction: str) -> Dict[str, Any]:
    tf1h, tf15 = snap.tf_1h, snap.tf_15m
    atr1h = tf1h.atr
    atr15 = tf15.atr
    mid = (entry_low + entry_high) / 2
    if direction == "short":
        sl = max(base + 0.3 * atr1h, entry_high + 0.8 * atr15)
        tp1 = mid - 1.2 * atr1h
        tp2 = tp1 - 1.0 * atr1h
        tp3 = tp2 - 1.0 * atr1h
    else:
        sl = min(base - 0.3 * atr1h, entry_low - 0.8 * atr15)
        tp1 = mid + 1.2 * atr1h
        tp2 = tp1 + 1.0 * atr1h
        tp3 = tp2 + 1.0 * atr1h

    rr = abs(mid - tp1) / max(abs(sl - mid), 1e-9)
    return {
        "sl": round(sl, 4),
        "tp": [round(tp1, 4), round(tp2, 4), round(tp3, 4)],
        "tp_mode": "ladder",
        "rr_estimate": round(rr, 2),
        "entry_mid": mid,
    }


def _confidence_if_triggered(plan_ctx: PlanContext, plan: Dict[str, Any], risk: Dict[str, Any]) -> Dict[str, Any]:
    tf1h, tf4h = plan_ctx.snap.tf_1h, plan_ctx.snap.tf_4h
    direction = plan_ctx.direction
    entry_mid = risk["entry_mid"]
    bonuses = {}

    proximity_bonus = 0.05 if abs(entry_mid - tf4h.ma25) <= 0.3 * tf1h.atr else 0.0
    bonuses["proximity_to_4h_level_bonus"] = proximity_bonus

    bids = plan_ctx.snap.bids or 0.0
    asks = plan_ctx.snap.asks or 0.0
    ob_bias = (bids - asks) / (bids + asks + 1e-9)
    orderbook_bonus = 0.03 if (direction == "long" and ob_bias > 0) or (direction == "short" and ob_bias < 0) else 0.0
    bonuses["orderbook_alignment_bonus"] = orderbook_bonus

    rsi_alignment = plan_ctx.snap.rsi_1h or tf1h.rsi6
    rsi_bonus = 0.05 if ((direction == "long" and rsi_alignment < 50) or (direction == "short" and rsi_alignment > 50)) else 0.0
    bonuses["rsi_alignment_bonus"] = rsi_bonus

    alignment_penalty = 0.05 if ((direction == "long" and tf4h.trend_label == "down") or (direction == "short" and tf4h.trend_label == "up")) else 0.0
    bonuses["4h_alignment_penalty"] = alignment_penalty

    confidence = _clamp(
        plan_ctx.edge_confidence + proximity_bonus + orderbook_bonus + rsi_bonus - alignment_penalty,
        0.0,
        1.0,
    )
    return {"confidence": confidence, "components": bonuses}


def _build_plan_block(plan_ctx: PlanContext, zone: Dict[str, float], regime: str) -> Optional[Dict[str, Any]]:
    entry_low = zone["low"]
    entry_high = zone["high"]
    base = zone["base"]
    direction = plan_ctx.direction
    tf4h = plan_ctx.snap.tf_4h

    plan_type = "WAIT_PULLBACK" if regime == "trending" else ("WAIT_BOUNCE_LONG" if direction == "long" else "WAIT_REJECT_SHORT")
    if direction == "short" and regime != "trending":
        plan_type = "WAIT_REJECT_SHORT"
    elif direction == "long" and regime != "trending":
        plan_type = "WAIT_BOUNCE_LONG"

    risk = _risk_block(entry_low, entry_high, base, plan_ctx.snap, direction)
    confidence_block = _confidence_if_triggered(plan_ctx, zone, risk)
    entry_logic = _entry_logic(direction)
    invalidation_close = entry_high + 0.5 * plan_ctx.snap.tf_1h.atr if direction == "short" else entry_low - 0.5 * plan_ctx.snap.tf_1h.atr

    return {
        "plan_type": plan_type,
        "direction": direction,
        "entry_zone": [round(entry_low, 4), round(entry_high, 4)],
        "entry_logic": entry_logic,
        "confidence_if_triggered": round(confidence_block["confidence"], 2),
        "confidence_components": confidence_block["components"],
        "invalidation": {
            "type": "STRUCTURE_BREAK",
            "anchor_tf": "4h",
            "rule": f"4h close {'>=' if direction == 'short' else '<='} {invalidation_close:.2f}",
        },
        "risk": {
            "sl": risk["sl"],
            "tp": risk["tp"],
            "tp_mode": "ladder",
            "rr_estimate": risk["rr_estimate"],
        },
        "reason": "4H 下跌结构未破，等待回抽至关键压力区" if direction == "short" else "4H 上涨结构未破，等待回踩关键支撑区",
    }


def _validity_block(snap: MarketSnapshot) -> Dict[str, Any]:
    return {
        "type": "UNTIL_NEXT_4H_CLOSE",
        "anchor_tf": "4h",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "valid_until_utc": (_next_4h_close(snap) or datetime.now(timezone.utc)).isoformat(),
    }


def build_conditional_plan(signal: Any, snap: MarketSnapshot, settings: Any) -> Optional[Dict[str, Any]]:
    cfg = getattr(settings, "notification", {}) or {}
    watch_edge_conf = cfg.get("watch_edge_conf", 0.8)
    min_confidence = getattr(settings, "min_confidence", 0.3)

    debug_scores = signal.debug_scores or {}
    long_score = debug_scores.get("long")
    short_score = debug_scores.get("short")
    if long_score is None or short_score is None:
        return None

    direction = "long" if long_score - short_score >= 0.1 else "short" if short_score - long_score >= 0.1 else None
    if direction is None:
        return None

    if snap.regime not in {"trending", "high_vol_ranging", "low_vol_ranging"}:
        return None
    if not snap.tf_4h.is_last_candle_closed:
        return None
    if signal.edge_confidence < watch_edge_conf:
        return None
    if (signal.trade_confidence or 0.0) >= min_confidence:
        return None

    plan_ctx = PlanContext(
        snap=snap,
        direction=direction,
        edge_confidence=signal.edge_confidence or 0.0,
        trade_confidence=signal.trade_confidence or 0.0,
        long_score=long_score,
        short_score=short_score,
        min_confidence=min_confidence,
        watch_edge_conf=watch_edge_conf,
    )

    if plan_ctx.direction_gap < 0.1:
        return None

    if snap.regime == "trending":
        zone = _entry_zone_trending(snap, direction)
    else:
        zone = _entry_zone_ranging(snap, direction)
    if not zone:
        return None

    plan_block = _build_plan_block(plan_ctx, zone, snap.regime)
    if not plan_block:
        return None

    return {
        "enabled": True,
        "validity": _validity_block(snap),
        "plans": [plan_block],
        "debug": {
            "long_score": long_score,
            "short_score": short_score,
            "direction_gap": plan_ctx.direction_gap,
            "regime": snap.regime,
        },
    }


__all__ = ["build_conditional_plan"]
