from datetime import datetime

from bot.config import Settings
from bot.models import TimeframeIndicators, DerivativeIndicators, MarketSnapshot
from bot.regime_detector import RegimeSignal, detect_regime
from bot.signal_engine import SignalEngine, TradeSignal


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
        high_last_n=high_last_n,
        low_last_n=low_last_n,
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


def make_regime_snapshot(ma25_history, rsi6_history) -> MarketSnapshot:
    tf = make_tf(
        "1h",
        close=101,
        ma7=101,
        ma25=100,
        ma99=99,
        rsi6=rsi6_history[-1],
        rsi12=rsi6_history[-1],
        rsi24=rsi6_history[-1],
        macd=0.5,
        macd_signal=0.4,
        macd_hist=0.1,
        atr=1.0,
        volume=500,
        ma25_history=ma25_history,
        rsi6_history=rsi6_history,
        trend_label="up",
    )
    deriv = make_deriv()

    return MarketSnapshot(
        symbol="TEST",
        ts=datetime.utcnow(),
        tf_4h=tf,
        tf_1h=tf,
        tf_15m=tf,
        deriv=deriv,
    )


def test_ranging_mid_zone_stays_sidelines_with_edge_confidence():
    tf4h = make_tf(
        "4h",
        close=100,
        ma7=100,
        ma25=100,
        ma99=99,
        rsi6=60,
        rsi12=58,
        rsi24=55,
        macd=0.2,
        macd_signal=0.1,
        macd_hist=0.1,
        atr=1.0,
        volume=500,
        trend_label="range",
    )
    tf1h = make_tf(
        "1h",
        close=100,
        ma7=100,
        ma25=100,
        ma99=100,
        rsi6=57,
        rsi12=52,
        rsi24=50,
        macd=0.0,
        macd_signal=0.0,
        macd_hist=0.0,
        atr=1.5,
        volume=400,
        ma25_history=[100] * 6,
        trend_label="range",
    )
    tf15 = make_tf(
        "15m",
        close=100,
        ma7=100,
        ma25=100,
        ma99=100,
        rsi6=45,
        rsi12=45,
        rsi24=45,
        macd=0.0,
        macd_signal=0.0,
        macd_hist=0.0,
        atr=0.8,
        volume=300,
        trend_label="range",
    )
    deriv = make_deriv(orderbook_asks=[{"size": 50}], orderbook_bids=[{"size": 60}])

    snap = MarketSnapshot(
        symbol="TEST",
        ts=datetime.utcnow(),
        tf_4h=tf4h,
        tf_1h=tf1h,
        tf_15m=tf15,
        deriv=deriv,
        regime="high_vol_ranging",
        rsidev=0.5,
        atrrel=0.01,
        rsi_15m=tf15.rsi6,
        rsi_1h=tf1h.rsi6,
        asks=50,
        bids=60,
    )

    engine = SignalEngine(Settings())
    signal = engine.generate_signal(snap)

    assert signal.direction == "none"
    assert signal.trade_confidence == 0.0
    assert signal.edge_confidence > 0.0


def test_range_edge_long_signal_emerges():
    tf4h = make_tf(
        "4h",
        close=100,
        ma7=100,
        ma25=100,
        ma99=99,
        rsi6=65,
        rsi12=62,
        rsi24=60,
        macd=0.3,
        macd_signal=0.1,
        macd_hist=0.2,
        atr=1.0,
        volume=600,
        trend_label="range",
    )
    tf1h = make_tf(
        "1h",
        close=95,
        ma7=100,
        ma25=100,
        ma99=100,
        rsi6=30,
        rsi12=35,
        rsi24=40,
        macd=-0.2,
        macd_signal=-0.1,
        macd_hist=-0.1,
        atr=1.5,
        volume=500,
        ma25_history=[100] * 6,
        trend_label="range",
    )
    tf15 = make_tf(
        "15m",
        close=95,
        ma7=99,
        ma25=100,
        ma99=100,
        rsi6=20,
        rsi12=25,
        rsi24=30,
        macd=-0.3,
        macd_signal=-0.2,
        macd_hist=-0.1,
        atr=0.9,
        volume=350,
        trend_label="range",
    )
    deriv = make_deriv(orderbook_asks=[{"size": 40}], orderbook_bids=[{"size": 120}])

    snap = MarketSnapshot(
        symbol="TEST",
        ts=datetime.utcnow(),
        tf_4h=tf4h,
        tf_1h=tf1h,
        tf_15m=tf15,
        deriv=deriv,
        regime="high_vol_ranging",
        rsidev=3.5,
        atrrel=0.01,
        rsi_15m=tf15.rsi6,
        rsi_1h=tf1h.rsi6,
        asks=40,
        bids=120,
    )

    engine = SignalEngine(Settings())
    signal = engine.generate_signal(snap)

    assert signal.direction == "none"
    assert signal.setup_type == "none"
    assert "no LH/MR trigger" in signal.reason
    assert signal.edge_confidence >= 0.5


def test_trending_flow_still_uses_trend_logic():
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
        rsi6=70,
        rsi12=63,
        rsi24=60,
        macd=0.8,
        macd_signal=0.6,
        macd_hist=0.2,
        atr=0.8,
        volume=500,
        ma25_history=[98, 99, 100, 101, 102, 103],
        rsi6_history=[70, 30, 70, 70, 70, 70],
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
    deriv = make_deriv(liquidity_comment="bids>asks")

    snap = MarketSnapshot(
        symbol="TEST",
        ts=datetime.utcnow(),
        tf_4h=tf4h,
        tf_1h=tf1h,
        tf_15m=tf15,
        deriv=deriv,
    )
    snap.maangle = 0.1
    snap.osc = 1

    engine = SignalEngine(Settings())
    signal = engine.generate_signal(snap)

    assert signal.setup_type in {"trend_long", "trend_short", "none"}
    assert "[trending]" in (signal.reason or "")


def test_detect_regime_trending_not_overridden_when_ma_angle_valid():
    snap = make_regime_snapshot(
        ma25_history=[100, 100.2, 100.4, 100.6, 100.8, 101.0],
        rsi6_history=[70, 30, 70, 30, 70, 30],
    )

    signal = detect_regime(snap, Settings())

    assert signal.regime == "trending"
    assert "weak_trend_override" not in signal.reason


def test_detect_regime_overrides_weak_trend_inside_detector():
    snap = make_regime_snapshot(
        ma25_history=[100, 100.2, 100.4, 100.6, 100.8, 101.0],
        rsi6_history=[70, 72, 74, 75, 76, 77],
    )

    signal = detect_regime(snap, Settings())

    assert signal.regime == "high_vol_ranging"
    assert "weak_trend_override=1" in signal.reason


def test_reason_prefix_uses_regime(monkeypatch):
    tf = make_tf(
        "4h",
        close=100,
        ma7=100,
        ma25=100,
        ma99=100,
        rsi6=50,
        rsi12=50,
        rsi24=50,
        macd=0.0,
        macd_signal=0.0,
        macd_hist=0.0,
        atr=1.0,
        volume=100,
        ma25_history=[100] * 6,
        rsi6_history=[50, 51, 49, 50],
        trend_label="range",
    )
    deriv = make_deriv()

    snap = MarketSnapshot(
        symbol="TEST",
        ts=datetime.utcnow(),
        tf_4h=tf,
        tf_1h=tf,
        tf_15m=tf,
        deriv=deriv,
    )

    engine = SignalEngine(Settings())

    def fake_detect_regime(snap_arg, settings):
        return RegimeSignal(
            regime="high_vol_ranging",
            reason="mock_regime_reason",
            ma_angle=0.0,
            atr_rel=0.02,
            rsi_avg_dev=0.0,
            osc_count=0,
        )

    def fake_decide(self, snap_arg, regime_signal):
        return TradeSignal(
            symbol=snap_arg.symbol,
            direction="none",
            reason="base reason",
            trade_confidence=0.1,
            edge_confidence=0.2,
            setup_type="none",
            snapshot=snap_arg,
        )

    monkeypatch.setattr("bot.signal_engine.detect_regime", fake_detect_regime)
    monkeypatch.setattr(SignalEngine, "decide", fake_decide)
    monkeypatch.setattr("bot.signal_engine.build_conditional_plan", lambda *_, **__: None)

    signal = engine.generate_signal(snap)

    assert signal.reason.startswith("[high_vol_ranging]")
    assert "regime=" not in signal.reason
