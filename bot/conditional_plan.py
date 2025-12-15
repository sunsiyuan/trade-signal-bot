"""Conditional plan compiler for execution intents (unified 4H TTL)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from .models import ConditionalPlan, ExecutionIntent, MarketSnapshot


def now_plus_hours(hours: int) -> str:
    """Return UTC ISO string after given hours."""

    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def resolve_atr_4h(snapshot: MarketSnapshot) -> Optional[float]:
    if snapshot.tf_4h and snapshot.tf_4h.atr:
        return snapshot.tf_4h.atr
    if snapshot.tf_1h and snapshot.tf_1h.atr:
        return snapshot.tf_1h.atr * 2.0
    if snapshot.tf_15m and snapshot.tf_15m.atr:
        return snapshot.tf_15m.atr * 4.0
    return None


def _cancel_rules(valid_until: Optional[str]) -> dict:
    return {
        "invalidation_crossed": True,
        "regime_changed": True,
        "expired": valid_until is not None,
    }


def build_conditional_plan_from_intent(
    intent: ExecutionIntent, snap: MarketSnapshot
) -> ConditionalPlan:
    if intent.direction == "none":
        return ConditionalPlan(
            execution_mode="WATCH_ONLY",
            direction="none",
            entry_price=None,
            valid_until_utc=None,
            cancel_if=_cancel_rules(None),
            explain="No execution intent",
        )

    current = getattr(snap, "price_last", None) or getattr(snap.tf_15m, "close", None)
    atr = intent.atr_4h
    entry_price = intent.entry_price

    if (
        intent.allow_execute_now
        and current is not None
        and entry_price is not None
        and atr
        and abs(current - entry_price) <= 0.35 * atr
    ):
        return ConditionalPlan(
            execution_mode="EXECUTE_NOW",
            direction=intent.direction,
            entry_price=current,
            valid_until_utc=None,
            cancel_if=_cancel_rules(None),
            explain="Entry reached, execute now",
        )

    if atr and current is not None and entry_price is not None:
        if abs(current - entry_price) <= 1.5 * atr:
            valid_until = now_plus_hours(intent.ttl_hours)
            return ConditionalPlan(
                execution_mode="PLACE_LIMIT_4H",
                direction=intent.direction,
                entry_price=entry_price,
                valid_until_utc=valid_until,
                cancel_if=_cancel_rules(valid_until),
                explain="Place 4h limit order at ideal entry",
            )

    return ConditionalPlan(
        execution_mode="WATCH_ONLY",
        direction=intent.direction,
        entry_price=None,
        valid_until_utc=None,
        cancel_if=_cancel_rules(None),
        explain="Too far from entry, watch only",
    )


__all__ = [
    "build_conditional_plan_from_intent",
    "now_plus_hours",
    "resolve_atr_4h",
]
