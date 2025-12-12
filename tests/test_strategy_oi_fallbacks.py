from datetime import datetime

import pytest

from bot.models import TimeframeIndicators, DerivativeIndicators, MarketSnapshot
from bot.strategy_mean_reversion import build_mean_reversion_signal
from bot.strategy_liquidity_hunt import build_liquidity_hunt_signal


def make_timeframe(
    timeframe: str,
    close: float,
    ma25: float,
    rsi6: float,
    atr: float,
    recent_high: float = None,
    recent_low: float = None,
    post_spike_small_body_count: int = 0,
    high_last_n=None,
    low_last_n=None,
):
    return TimeframeIndicators(
        timeframe=timeframe,
        close=close,
        ma7=ma25,
        ma25=ma25,
        ma99=ma25,
        rsi6=rsi6,
        rsi12=rsi6,
        rsi24=rsi6,
        macd=0.0,
        macd_signal=0.0,
        macd_hist=0.0,
        atr=atr,
        volume=100,
        recent_high=recent_high,
        recent_low=recent_low,
        high_last_n=high_last_n or [],
        low_last_n=low_last_n or [],
        post_spike_small_body_count=post_spike_small_body_count,
    )


def make_snapshot(tf_1h: TimeframeIndicators, deriv: DerivativeIndicators):
    tf_4h = make_timeframe("4h", tf_1h.close, tf_1h.ma25, tf_1h.rsi6, tf_1h.atr)
    tf_15m = make_timeframe("15m", tf_1h.close, tf_1h.ma25, tf_1h.rsi6, tf_1h.atr)
    return MarketSnapshot(
        symbol="TEST",
        ts=datetime.utcnow(),
        tf_4h=tf_4h,
        tf_1h=tf_1h,
        tf_15m=tf_15m,
        deriv=deriv,
        regime="high_vol_ranging",
        rsidev=0.0,
        atrrel=0.0,
    )


def make_deriv(
    oi_change_pct=None,
    has_large_ask_wall=False,
    has_large_bid_wall=False,
    ask_wall_size=0.0,
    bid_wall_size=0.0,
):
    return DerivativeIndicators(
        funding=0.0,
        open_interest=0.0,
        oi_change_24h=0.0,
        oi_change_pct=oi_change_pct,
        orderbook_asks=[],
        orderbook_bids=[],
        ask_wall_size=ask_wall_size,
        bid_wall_size=bid_wall_size,
        has_large_ask_wall=has_large_ask_wall,
        has_large_bid_wall=has_large_bid_wall,
    )


def test_mean_reversion_allows_fallback_when_oi_missing():
    tf = make_timeframe("1h", close=90, ma25=100, rsi6=10, atr=5)
    snap = make_snapshot(tf, make_deriv(oi_change_pct=None))

    signal = build_mean_reversion_signal(
        snap,
        regime="high_vol_ranging",
        settings={"mean_reversion": {"allow_oi_missing_fallback": True}},
    )

    assert signal is not None
    assert signal.confidence == pytest.approx(0.6375)
    assert signal.core_position_pct == pytest.approx(0.25)
    assert "OI missing → fallback mode" in signal.reason


def test_mean_reversion_requires_oi_when_fallback_disabled():
    tf = make_timeframe("1h", close=90, ma25=100, rsi6=10, atr=5)
    snap = make_snapshot(tf, make_deriv(oi_change_pct=None))

    signal = build_mean_reversion_signal(
        snap,
        regime="high_vol_ranging",
        settings={"mean_reversion": {"require_oi": True, "allow_oi_missing_fallback": False}},
    )

    assert signal is None


def test_mean_reversion_triggers_with_oi_present():
    tf = make_timeframe("1h", close=90, ma25=100, rsi6=10, atr=5)
    snap = make_snapshot(tf, make_deriv(oi_change_pct=-5.0))

    signal = build_mean_reversion_signal(
        snap,
        regime="high_vol_ranging",
        settings={"mean_reversion": {}},
    )

    assert signal is not None
    assert signal.confidence > 0.6
    assert "flushing out" in signal.reason


def test_liquidity_hunt_uses_fallback_when_oi_missing():
    tf = make_timeframe(
        "1h",
        close=100.0,
        ma25=100.0,
        rsi6=50,
        atr=2.0,
        recent_high=100.2,
        recent_low=99.8,
        post_spike_small_body_count=3,
        high_last_n=[100.2],
        low_last_n=[99.8],
    )
    deriv = make_deriv(
        oi_change_pct=None,
        has_large_ask_wall=True,
        ask_wall_size=10.0,
        bid_wall_size=1.0,
    )
    snap = make_snapshot(tf, deriv)

    signal = build_liquidity_hunt_signal(
        snap,
        regime="high_vol_ranging",
        settings={"liquidity_hunt": {"allow_oi_missing_fallback": True}},
    )

    assert signal is not None
    assert signal.confidence == 0.65
    assert signal.core_position_pct == pytest.approx(0.25)
    assert "OI missing → fallback mode" in signal.reason


def test_liquidity_hunt_requires_oi_when_fallback_disabled():
    tf = make_timeframe(
        "1h",
        close=100.0,
        ma25=100.0,
        rsi6=50,
        atr=2.0,
        recent_high=100.2,
        recent_low=99.8,
        post_spike_small_body_count=3,
        high_last_n=[100.2],
        low_last_n=[99.8],
    )
    snap = make_snapshot(tf, make_deriv(oi_change_pct=None, has_large_ask_wall=True))

    signal = build_liquidity_hunt_signal(
        snap,
        regime="high_vol_ranging",
        settings={"liquidity_hunt": {"require_oi": True, "allow_oi_missing_fallback": False}},
    )

    assert signal is None


def test_liquidity_hunt_triggers_with_oi_spike():
    tf = make_timeframe(
        "1h",
        close=100.0,
        ma25=100.0,
        rsi6=50,
        atr=2.0,
        recent_high=100.2,
        recent_low=99.8,
        post_spike_small_body_count=3,
        high_last_n=[100.2],
        low_last_n=[99.8],
    )
    deriv = make_deriv(oi_change_pct=6.0, has_large_ask_wall=True)
    snap = make_snapshot(tf, deriv)

    signal = build_liquidity_hunt_signal(
        snap,
        regime="high_vol_ranging",
        settings={"liquidity_hunt": {}},
    )

    assert signal is not None
    assert signal.confidence == 0.75
    assert "spike" in signal.reason
