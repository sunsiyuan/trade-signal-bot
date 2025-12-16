from datetime import datetime

import pytest

from bot.config import Settings
from bot.models import MarketSnapshot, TimeframeIndicators, DerivativeIndicators
from bot.regime_detector import RegimeSignal
from bot.strategy_trend_following import build_trend_following_signal


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
    trend_label: str,
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
        trend_label=trend_label,
    )


def make_snapshot(
    tf4h: TimeframeIndicators,
    tf1h: TimeframeIndicators,
    tf15: TimeframeIndicators,
    deriv: DerivativeIndicators,
):
    return MarketSnapshot(
        symbol="TEST",
        ts=datetime.utcnow(),
        tf_4h=tf4h,
        tf_1h=tf1h,
        tf_15m=tf15,
        deriv=deriv,
    )


def trending_signal() -> RegimeSignal:
    return RegimeSignal(
        regime="trending",
        reason="manual",
        ma_angle=0.01,
        atr_rel=0.01,
        rsi_avg_dev=5.0,
        osc_count=0,
    )


@pytest.fixture(autouse=True)
def patch_trigger(monkeypatch):
    # Ensure price confirmation does not block the signal so we can focus on gating logic.
    monkeypatch.setattr(
        "bot.strategy_trend_following._compute_trigger",
        lambda entry, atr, direction: entry,
    )


def test_low_conf_still_trades_with_reduced_size():
    settings = Settings()

    tf4h = make_tf("4h", 100, 99, 98, 100, 55, 55, 55, 0.0, 0.0, 0.0, 1.0, 1000, "down")
    tf1h = make_tf("1h", 100, 99, 98, 100, 50, 50, 50, 0.0, 0.0, 0.0, 1.0, 800, "down")
    tf15 = make_tf("15m", 100, 99, 98, 99, 68, 65, 63, 0.0, 0.0, 0.0, 1.0, 600, "range")
    deriv = DerivativeIndicators(
        funding=0.03,
        open_interest=100,
        oi_change_24h=0.0,
        liquidity_comment="balanced",
    )

    snap = make_snapshot(tf4h, tf1h, tf15, deriv)

    signal = build_trend_following_signal(
        snap,
        trending_signal(),
        min_confidence=settings.min_confidence,
        settings=settings,
    )

    assert signal.direction == "short"
    assert signal.core_position_pct == pytest.approx(0.2)
    assert signal.add_position_pct == 0.0
    assert signal.trade_confidence == pytest.approx(0.65)
    assert signal.debug_scores["gate_tag"] == "low_conf"


def test_high_conf_boosts_confidence_and_size():
    settings = Settings()
    settings.trend_following.update({"high_conf_core_mult": 1.2, "high_conf_add_mult": 1.1})

    tf4h = make_tf("4h", 100, 101, 102, 99, 65, 60, 55, 0.0, 0.0, 0.0, 1.0, 1000, "up")
    tf1h = make_tf("1h", 100, 101, 102, 99, 55, 55, 55, 0.0, 0.0, 0.0, 1.0, 800, "up")
    tf15 = make_tf("15m", 101, 101, 101, 95, 20, 25, 28, 0.02, 0.01, 0.01, 1.0, 600, "range")
    deriv = DerivativeIndicators(
        funding=-0.05,
        open_interest=100,
        oi_change_24h=0.0,
        liquidity_comment="bids dominant",
    )

    snap = make_snapshot(tf4h, tf1h, tf15, deriv)

    signal = build_trend_following_signal(
        snap,
        trending_signal(),
        min_confidence=settings.min_confidence,
        settings=settings,
    )

    assert signal.direction == "long"
    assert signal.trade_confidence > 0.75  # Base confidence before bonus
    assert signal.trade_confidence == pytest.approx(1.0)
    assert signal.core_position_pct == pytest.approx(0.84)
    assert signal.add_position_pct == pytest.approx(0.33)
    assert signal.debug_scores["gate_tag"] == "high_conf"
