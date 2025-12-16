from datetime import datetime

import pandas as pd

from bot.config import Settings
from bot.data_client import HyperliquidDataClient
from bot.models import DerivativeIndicators, MarketSnapshot, TimeframeIndicators
from bot.strategy_liquidity_hunt import build_liquidity_hunt_signal


class DummyExchange:
    def market(self, symbol):
        return {"symbol": symbol, "id": symbol}


def make_sample_df(rows: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC")
    base_price = 100.0
    data = {
        "timestamp": timestamps,
        "open": [base_price + i * 0.05 for i in range(rows)],
        "high": [base_price + i * 0.05 + 0.2 for i in range(rows)],
        "low": [base_price + i * 0.05 - 0.2 for i in range(rows)],
        "close": [base_price + i * 0.05 + 0.01 for i in range(rows)],
        "volume": [1000 + i for i in range(rows)],
    }
    return pd.DataFrame(data)


def make_timeframe(
    timeframe: str,
    close: float,
    ma25: float,
    rsi6: float,
    atr: float,
    recent_high=None,
    recent_low=None,
    post_spike_small_body_count=None,
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
        high_last_n=high_last_n,
        low_last_n=low_last_n,
        post_spike_small_body_count=post_spike_small_body_count,
    )


def make_deriv(
    oi_change_24h=0.0,
    has_large_ask_wall=False,
    has_large_bid_wall=False,
    ask_wall_size=0.0,
    bid_wall_size=0.0,
):
    return DerivativeIndicators(
        funding=0.0,
        open_interest=0.0,
        oi_change_24h=oi_change_24h,
        orderbook_asks=[],
        orderbook_bids=[],
        ask_wall_size=ask_wall_size,
        bid_wall_size=bid_wall_size,
        has_large_ask_wall=has_large_ask_wall,
        has_large_bid_wall=has_large_bid_wall,
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


def test_tf_indicators_fill_lh_fields():
    settings = Settings()
    client = HyperliquidDataClient(settings, exchange=DummyExchange(), funding_rates={})

    tf = client._build_tf_indicators(make_sample_df(), "1h")

    assert tf.recent_high is not None
    assert tf.recent_low is not None
    assert tf.post_spike_small_body_count is not None


def test_derivative_indicators_expose_wall_fields():
    deriv = make_deriv()
    assert hasattr(deriv, "ask_wall_size")
    assert hasattr(deriv, "bid_wall_size")
    assert hasattr(deriv, "has_large_ask_wall")
    assert hasattr(deriv, "has_large_bid_wall")


def test_missing_fields_trigger_fallback_reason():
    tf = make_timeframe(
        "1h",
        close=100.0,
        ma25=100.0,
        rsi6=50,
        atr=2.0,
        recent_high=None,
        recent_low=None,
        post_spike_small_body_count=None,
    )
    deriv = make_deriv(has_large_ask_wall=True, ask_wall_size=5.0, bid_wall_size=1.0)
    snap = make_snapshot(tf, deriv)

    signal = build_liquidity_hunt_signal(
        snap,
        regime="high_vol_ranging",
        settings={"liquidity_hunt": {"allow_fallback_when_missing": True}},
    )

    assert signal is not None
    assert "lh_missing" in signal.reason
    assert signal.trade_confidence == 0.35


def test_missing_fields_block_signal_when_disabled():
    tf = make_timeframe(
        "1h",
        close=100.0,
        ma25=100.0,
        rsi6=50,
        atr=2.0,
        recent_high=None,
        recent_low=None,
        post_spike_small_body_count=None,
    )
    deriv = make_deriv(has_large_ask_wall=True, ask_wall_size=5.0, bid_wall_size=1.0)
    snap = make_snapshot(tf, deriv)

    signal = build_liquidity_hunt_signal(
        snap,
        regime="high_vol_ranging",
        settings={"liquidity_hunt": {"allow_fallback_when_missing": False}},
    )

    assert signal is None
    assert hasattr(snap, "lh_missing_reason")


def test_positive_liquidity_hunt_trigger():
    tf = make_timeframe(
        "1h",
        close=100.0,
        ma25=100.0,
        rsi6=50,
        atr=2.0,
        recent_high=100.2,
        recent_low=99.0,
        post_spike_small_body_count=4,
        high_last_n=100.2,
        low_last_n=99.0,
    )
    deriv = make_deriv(
        oi_change_24h=6.0,
        has_large_ask_wall=True,
        ask_wall_size=12.0,
        bid_wall_size=1.0,
    )
    snap = make_snapshot(tf, deriv)

    signal = build_liquidity_hunt_signal(snap, regime="high_vol_ranging", settings={})

    assert signal is not None
    assert signal.direction == "short"
    assert signal.trade_confidence == 0.75
