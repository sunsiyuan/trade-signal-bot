from datetime import datetime, timedelta, timezone

from bot.main import format_action_plan_message
from bot.models import (
    ConditionalPlan,
    DerivativeIndicators,
    ExecutionIntent,
    MarketSnapshot,
    TimeframeIndicators,
)
from bot.signal_engine import TradeSignal


def _tf(tf: str, close: float, rsi6: float = 55.0) -> TimeframeIndicators:
    now = datetime.now(timezone.utc)
    return TimeframeIndicators(
        timeframe=tf,
        close=close,
        ma7=close,
        ma25=close,
        ma99=close,
        rsi6=rsi6,
        rsi12=rsi6,
        rsi24=rsi6,
        macd=0.1,
        macd_signal=0.05,
        macd_hist=-0.02,
        atr=2.0,
        volume=100.0,
        last_candle_open_utc=now - timedelta(minutes=15),
        last_candle_close_utc=now,
    )


def _snapshot(mark_price: float | None, price_last: float) -> MarketSnapshot:
    tf_4h = _tf("4h", 190.0)
    tf_1h = _tf("1h", 188.0)
    tf_15m = _tf("15m", price_last, rsi6=61.2)
    tf_15m.price_last = price_last
    tf_15m.prices = {"mark": mark_price, "price_last": price_last}
    deriv = DerivativeIndicators(
        funding=0.01,
        open_interest=1000.0,
        oi_change_24h=1.0,
        orderbook_asks=[],
        orderbook_bids=[],
    )
    return MarketSnapshot(
        symbol="TEST/USDC:USDC",
        ts=datetime.now(timezone.utc),
        tf_4h=tf_4h,
        tf_1h=tf_1h,
        tf_15m=tf_15m,
        deriv=deriv,
    )


def _signal(snapshot: MarketSnapshot, tp1=130.0, tp2=None, sl=95.0) -> TradeSignal:
    sig = TradeSignal(
        symbol=snapshot.symbol,
        direction="long",
        snapshot=snapshot,
        tp1=tp1,
        tp2=tp2,
        sl=sl,
    )
    sig.execution_intent = ExecutionIntent(
        symbol=snapshot.symbol,
        direction="long",
        entry_price=120.0,
        entry_reason="test",
        invalidation_price=94.5,
    )
    return sig


def test_format_action_plan_message_with_mark_and_multiple_tp():
    snap = _snapshot(mark_price=123.4567, price_last=120.1234)
    signal = _signal(snap, tp1=130.0, tp2=135.0)
    plan = ConditionalPlan(
        execution_mode="PLACE_LIMIT_4H",
        direction="long",
        entry_price=120.0,
        valid_until_utc="2024-01-01T00:00:00+00:00",
        cancel_if={"expired": True},
        explain="test plan",
    )

    msg = format_action_plan_message(signal, snap, plan, signal_id="sig-1", event="CREATED")

    assert "现价: 123.4567" in msg
    assert "TP: 130.0000/135.0000" in msg
    assert "SL: 95.0000" in msg
    assert "15m RSI6: 61.2" in msg
    assert "2024-01-01" in msg


def test_format_action_plan_message_fallbacks_and_no_none_strings():
    snap = _snapshot(mark_price=None, price_last=110.9876)
    signal = _signal(snap, tp1=125.5, tp2=None, sl=None)
    plan = {
        "execution_mode": "EXECUTE_NOW",
        "direction": "long",
        "entry_price": 111.0,
        "valid_until_utc": None,
        "cancel_if": {},
        "explain": "execute immediately",
    }

    msg = format_action_plan_message(signal, snap, plan, signal_id="sig-2", event="EXECUTE_NOW")

    assert "现价: 110.9876" in msg
    assert "TP: 125.5000" in msg
    assert "SL: 94.5000" in msg  # from execution_intent invalidation
    assert "None" not in msg
