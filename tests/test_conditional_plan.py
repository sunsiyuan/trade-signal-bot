import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from bot.conditional_plan import build_conditional_plan
from bot.models import DerivativeIndicators, MarketSnapshot, TimeframeIndicators
from bot.signal_engine import TradeSignal


def _tf(tf: str, close: float, now: datetime, trend_label: str = "down", atr: float = 2.0) -> TimeframeIndicators:
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
        trend_label=trend_label,
        last_candle_open_utc=now - timedelta(hours=4),
        last_candle_close_utc=now,
        is_last_candle_closed=True,
    )


def _snapshot(price: float = 410.0, trend_label: str = "down") -> MarketSnapshot:
    now = datetime.now(timezone.utc)
    tf_4h = _tf("4h", price, now, trend_label=trend_label, atr=5.0)
    tf_1h = _tf("1h", price - 2, now, trend_label=trend_label, atr=3.0)
    tf_15m = _tf("15m", price - 4, now, trend_label=trend_label, atr=1.5)
    deriv = DerivativeIndicators(
        funding=0.01,
        open_interest=1000.0,
        oi_change_24h=1.0,
        orderbook_asks=[],
        orderbook_bids=[],
    )
    return MarketSnapshot(
        symbol="ZEC/USDC:USDC",
        ts=now,
        tf_4h=tf_4h,
        tf_1h=tf_1h,
        tf_15m=tf_15m,
        deriv=deriv,
        market_mode="trending",
        regime="trending",
        regime_reason="trend intact",
        bids=120.0,
        asks=100.0,
    )


def _make_signal(edge: float, trade: float, long_score: float, short_score: float) -> TradeSignal:
    return TradeSignal(
        symbol="ZEC/USDC:USDC",
        direction="none",
        trade_confidence=trade,
        edge_confidence=edge,
        snapshot=None,
        debug_scores={"long": long_score, "short": short_score},
    )


def test_edge_high_trade_low_generates_plan():
    snap = _snapshot()
    signal = _make_signal(0.85, 0.1, long_score=0.1, short_score=0.9)

    plan = build_conditional_plan(signal, snap, {})

    assert plan is not None
    assert plan["enabled"] is True
    assert plan["plans"][0]["plan_type"] == "WAIT_PULLBACK"


def test_skip_when_price_already_in_zone():
    snap = _snapshot(price=395.0)
    snap.tf_15m.close = snap.tf_4h.ma25  # force price inside likely zone
    signal = _make_signal(0.9, 0.1, long_score=0.2, short_score=0.8)

    plan = build_conditional_plan(signal, snap, {})

    assert plan is None


def test_entry_zone_is_valid_range():
    snap = _snapshot()
    signal = _make_signal(0.85, 0.05, long_score=0.15, short_score=0.85)

    plan = build_conditional_plan(signal, snap, {})

    entry_low, entry_high = plan["plans"][0]["entry_zone"]
    assert entry_low < entry_high


def test_short_sl_above_entry_high():
    snap = _snapshot()
    signal = _make_signal(0.9, 0.05, long_score=0.05, short_score=0.95)

    plan = build_conditional_plan(signal, snap, {})

    entry_low, entry_high = plan["plans"][0]["entry_zone"]
    sl = plan["plans"][0]["risk"]["sl"]
    assert sl > entry_high


def test_valid_until_next_4h_close():
    snap = _snapshot()
    signal = _make_signal(0.85, 0.05, long_score=0.1, short_score=0.9)

    plan = build_conditional_plan(signal, snap, {})

    expected = (snap.tf_4h.last_candle_close_utc + timedelta(hours=4)).isoformat()
    assert plan["validity"]["valid_until_utc"] == expected
