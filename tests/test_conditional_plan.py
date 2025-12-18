import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from bot.conditional_plan import build_conditional_plan_from_intent, now_plus_hours
from bot.models import DerivativeIndicators, ExecutionIntent, MarketSnapshot, TimeframeIndicators


def _tf(tf: str, close: float, now: datetime, atr: float) -> TimeframeIndicators:
    return TimeframeIndicators(
        timeframe=tf,
        close=close,
        ma7=close,
        ma25=close,
        ma99=close,
        rsi6=55.0,
        rsi12=50.0,
        rsi24=50.0,
        macd=0.1,
        macd_signal=0.05,
        macd_hist=-0.02,
        atr=atr,
        volume=100.0,
        trend_label="down",
        last_candle_open_utc=now - timedelta(hours=4),
        last_candle_close_utc=now,
        is_last_candle_closed=True,
    )


def _snapshot(price: float = 100.0, atr_4h: float = 10.0) -> MarketSnapshot:
    now = datetime.now(timezone.utc)
    tf_4h = _tf("4h", price, now, atr=atr_4h)
    tf_1h = _tf("1h", price, now, atr=atr_4h / 2)
    tf_15m = _tf("15m", price, now, atr=atr_4h / 4)
    deriv = DerivativeIndicators(
        funding=0.01,
        open_interest=1000.0,
        oi_change_24h=1.0,
        orderbook_asks=[],
        orderbook_bids=[],
    )
    snap = MarketSnapshot(
        symbol="BTC/USDT",
        ts=now,
        tf_4h=tf_4h,
        tf_1h=tf_1h,
        tf_15m=tf_15m,
        deriv=deriv,
        rolling_candidate="trending",
        rolling_candidate_dir="up",
        rolling_candidate_streak=2,
    )
    snap.price_last = price
    return snap


def _intent(
    entry: float,
    atr: float = 10.0,
    allow_execute_now: bool = True,
    direction: str = "long",
) -> ExecutionIntent:
    return ExecutionIntent(
        symbol="BTC/USDT",
        direction=direction,
        entry_price=entry,
        entry_reason="test",
        invalidation_price=entry - 2,
        atr_4h=atr,
        ttl_hours=4,
        allow_execute_now=allow_execute_now,
        reason="unit-test",
        debug=None,
    )


def test_execute_now_when_price_close_to_entry():
    snap = _snapshot(price=100.0)
    intent = _intent(entry=99.0, atr=10.0)

    plan = build_conditional_plan_from_intent(intent, snap)

    assert plan.execution_mode == "EXECUTE_NOW"
    assert plan.entry_price == snap.price_last
    assert plan.valid_until_utc is None
    assert "execute now" in plan.explain


def test_place_limit_when_within_one_atr():
    snap = _snapshot(price=115.0)
    intent = _intent(entry=110.0, atr=10.0, allow_execute_now=False)

    plan = build_conditional_plan_from_intent(intent, snap)

    assert plan.execution_mode == "PLACE_LIMIT_4H"
    assert plan.entry_price == intent.entry_price
    valid_dt = datetime.fromisoformat(plan.valid_until_utc)
    delta_hours = (valid_dt - datetime.now(timezone.utc)).total_seconds() / 3600
    assert 3.0 <= delta_hours <= 5.0
    assert "limit" in plan.explain


def test_long_entry_above_current_does_not_place_limit():
    snap = _snapshot(price=100.0)
    intent = _intent(entry=105.0, atr=10.0, allow_execute_now=False)

    plan = build_conditional_plan_from_intent(intent, snap)

    assert plan.execution_mode == "WATCH_ONLY"
    assert plan.entry_price is None


def test_short_entry_below_current_does_not_place_limit():
    snap = _snapshot(price=100.0)
    intent = _intent(entry=95.0, atr=10.0, allow_execute_now=False, direction="short")

    plan = build_conditional_plan_from_intent(intent, snap)

    assert plan.execution_mode == "WATCH_ONLY"
    assert plan.entry_price is None


def test_watch_only_when_far():
    snap = _snapshot(price=150.0)
    intent = _intent(entry=100.0, atr=10.0)

    plan = build_conditional_plan_from_intent(intent, snap)

    assert plan.execution_mode == "WATCH_ONLY"
    assert plan.entry_price is None
    assert plan.valid_until_utc is None


def test_direction_none_returns_watch():
    snap = _snapshot(price=100.0)
    intent = ExecutionIntent(
        symbol="BTC/USDT",
        direction="none",
        entry_price=None,
        entry_reason="none",
        invalidation_price=None,
        atr_4h=None,
        reason="no intent",
        debug=None,
    )

    plan = build_conditional_plan_from_intent(intent, snap)

    assert plan.execution_mode == "WATCH_ONLY"
    assert plan.direction == "none"
