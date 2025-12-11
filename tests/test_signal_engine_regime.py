from datetime import datetime

from bot.config import Settings
from bot.models import TimeframeIndicators, DerivativeIndicators, MarketSnapshot
from bot.regime_detector import detect_regime
from bot.signal_engine import SignalEngine
from bot.strategy_liquidity_hunt import build_liquidity_hunt_signal


def make_tf(
    timeframe: str,
    close: float,
    ma7: float,
    ma25: float,
    ma99: float,
    rsi6: float,
    rsi12: float,
    rsi24: float,
    macd: float,
    macd_signal: float,
    macd_hist: float,
    atr: float,
    volume: float,
    ma25_history=None,
    rsi6_history=None,
    recent_high=None,
    recent_low=None,
    high_last_n=None,
    low_last_n=None,
    post_spike_small_body_count: int = 0,
    trend_label: str = "range",
):
    return TimeframeIndicators(
        timeframe=timeframe,
        close=close,
        ma7=ma7,
        ma25=ma25,
        ma99=ma99,
        rsi6=rsi6,
        rsi12=rsi12,
        rsi24=rsi24,
        macd=macd,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        atr=atr,
        volume=volume,
        ma25_history=ma25_history or [],
        rsi6_history=rsi6_history or [],
        recent_high=recent_high,
        recent_low=recent_low,
        high_last_n=high_last_n or [],
        low_last_n=low_last_n or [],
        post_spike_small_body_count=post_spike_small_body_count,
        trend_label=trend_label,
    )


def make_deriv(
    funding: float = 0.0,
    open_interest: float = 0.0,
    oi_change_24h: float = 0.0,
    oi_change_pct: float = 0.0,
    orderbook_asks=None,
    orderbook_bids=None,
    liquidity_comment: str = "",
    ask_wall_size: float = 0.0,
    bid_wall_size: float = 0.0,
    has_large_ask_wall: bool = False,
    has_large_bid_wall: bool = False,
):
    return DerivativeIndicators(
        funding=funding,
        open_interest=open_interest,
        oi_change_24h=oi_change_24h,
        oi_change_pct=oi_change_pct,
        orderbook_asks=orderbook_asks or [],
        orderbook_bids=orderbook_bids or [],
        liquidity_comment=liquidity_comment,
        ask_wall_size=ask_wall_size,
        bid_wall_size=bid_wall_size,
        has_large_ask_wall=has_large_ask_wall,
        has_large_bid_wall=has_large_bid_wall,
    )


def test_trending_regime_prefers_trend_follow():
    tf4h = make_tf(
        "4h",
        close=105,
        ma7=104,
        ma25=102,
        ma99=98,
        rsi6=70,
        rsi12=68,
        rsi24=65,
        macd=1.2,
        macd_signal=1.0,
        macd_hist=0.2,
        atr=1.2,
        volume=1000,
        ma25_history=[96, 98, 100, 101, 102, 103, 104],
        rsi6_history=[65, 66, 67, 68, 70],
        trend_label="up",
    )
    tf1h = make_tf(
        "1h",
        close=105,
        ma7=104,
        ma25=103,
        ma99=95,
        rsi6=65,
        rsi12=63,
        rsi24=60,
        macd=0.8,
        macd_signal=0.6,
        macd_hist=0.2,
        atr=0.8,
        volume=500,
        ma25_history=[98, 99, 100, 101, 102, 103],
        rsi6_history=[55, 58, 60, 62, 64, 65],
        trend_label="up",
    )
    tf15 = make_tf(
        "15m",
        close=105,
        ma7=104.5,
        ma25=103.5,
        ma99=100,
        rsi6=75,
        rsi12=70,
        rsi24=65,
        macd=0.5,
        macd_signal=0.4,
        macd_hist=0.1,
        atr=0.2,
        volume=200,
        trend_label="up",
    )
    deriv = make_deriv()

    snap = MarketSnapshot(
        symbol="TEST",
        ts=datetime.utcnow(),
        tf_4h=tf4h,
        tf_1h=tf1h,
        tf_15m=tf15,
        deriv=deriv,
    )

    regime_signal = detect_regime(snap, Settings())
    assert regime_signal.regime == "trending"

    engine = SignalEngine(Settings())
    signal = engine.generate_signal(snap)
    assert "[trend]" in signal.reason


def test_mean_reversion_long_in_ranging_regime():
    tf4h = make_tf(
        "4h",
        close=100,
        ma7=100,
        ma25=100,
        ma99=99,
        rsi6=50,
        rsi12=50,
        rsi24=50,
        macd=0.0,
        macd_signal=0.0,
        macd_hist=0.0,
        atr=1.5,
        volume=800,
        ma25_history=[100, 100, 100, 100, 100],
        rsi6_history=[49, 51, 50, 52, 48],
        trend_label="range",
    )
    tf1h = make_tf(
        "1h",
        close=95,
        ma7=100,
        ma25=100,
        ma99=100,
        rsi6=10,
        rsi12=25,
        rsi24=30,
        macd=-0.5,
        macd_signal=-0.4,
        macd_hist=-0.1,
        atr=3.0,
        volume=600,
        ma25_history=[100, 100, 100, 100, 100],
        rsi6_history=[49, 52, 48, 51, 49],
        trend_label="range",
    )
    tf15 = make_tf(
        "15m",
        close=95,
        ma7=99,
        ma25=100,
        ma99=100,
        rsi6=15,
        rsi12=25,
        rsi24=35,
        macd=-0.3,
        macd_signal=-0.2,
        macd_hist=-0.1,
        atr=1.5,
        volume=400,
        trend_label="range",
    )
    deriv = make_deriv(oi_change_pct=-5.0)

    snap = MarketSnapshot(
        symbol="TEST",
        ts=datetime.utcnow(),
        tf_4h=tf4h,
        tf_1h=tf1h,
        tf_15m=tf15,
        deriv=deriv,
    )

    engine = SignalEngine(Settings())
    signal = engine.generate_signal(snap)

    assert signal.direction == "long"
    assert "mean_reversion" in signal.reason
    assert signal.entry == tf1h.close


def test_liquidity_hunt_short_setup():
    tf4h = make_tf(
        "4h",
        close=110,
        ma7=110,
        ma25=110,
        ma99=109,
        rsi6=50,
        rsi12=50,
        rsi24=50,
        macd=0.0,
        macd_signal=0.0,
        macd_hist=0.0,
        atr=2.5,
        volume=900,
        ma25_history=[110, 110, 110, 110, 110],
        rsi6_history=[51, 49, 52, 48, 51],
        trend_label="range",
    )
    tf1h = make_tf(
        "1h",
        close=109.6,
        ma7=110,
        ma25=110,
        ma99=109,
        rsi6=55,
        rsi12=50,
        rsi24=50,
        macd=0.1,
        macd_signal=0.1,
        macd_hist=0.0,
        atr=2.2,
        volume=700,
        ma25_history=[110, 110, 110, 110, 110],
        rsi6_history=[49, 51, 50, 49, 52],
        recent_high=110.0,
        high_last_n=[109.8, 109.9, 110.0, 110.1],
        post_spike_small_body_count=3,
        trend_label="range",
    )
    tf15 = make_tf(
        "15m",
        close=109.6,
        ma7=109.8,
        ma25=109.9,
        ma99=109.5,
        rsi6=60,
        rsi12=55,
        rsi24=52,
        macd=0.05,
        macd_signal=0.04,
        macd_hist=0.01,
        atr=1.2,
        volume=500,
        trend_label="range",
    )
    deriv = make_deriv(
        oi_change_pct=6.0,
        ask_wall_size=300,
        bid_wall_size=50,
        has_large_ask_wall=True,
    )

    snap = MarketSnapshot(
        symbol="TEST",
        ts=datetime.utcnow(),
        tf_4h=tf4h,
        tf_1h=tf1h,
        tf_15m=tf15,
        deriv=deriv,
    )

    regime_signal = detect_regime(snap, Settings())
    assert regime_signal.regime == "high_vol_ranging"

    lh_signal = build_liquidity_hunt_signal(snap, regime_signal.regime, Settings())

    assert lh_signal is not None
    assert lh_signal.direction == "short"
    assert "liquidity" in lh_signal.reason.lower()
