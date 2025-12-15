from datetime import datetime, timedelta, timezone

from bot.main import format_summary_line
from bot.models import DerivativeIndicators, MarketSnapshot, TimeframeIndicators
from bot.signal_engine import TradeSignal


def _make_snapshot() -> MarketSnapshot:
    now = datetime.now(timezone.utc)
    tf_kwargs = dict(
        ma7=99.0,
        ma25=98.0,
        ma99=97.0,
        rsi6=50.0,
        rsi12=48.0,
        rsi24=52.0,
        macd=0.1,
        macd_signal=0.05,
        macd_hist=0.05,
        atr=1.0,
        volume=10.0,
        last_candle_open_utc=now - timedelta(minutes=15),
        last_candle_close_utc=now,
    )

    tf_4h = TimeframeIndicators(timeframe="4h", close=100.0, **tf_kwargs)
    tf_1h = TimeframeIndicators(timeframe="1h", close=100.0, **tf_kwargs)
    tf_15m = TimeframeIndicators(timeframe="15m", close=100.0, **tf_kwargs)

    deriv = DerivativeIndicators(
        funding=0.01,
        open_interest=1000.0,
        oi_change_24h=1.0,
        orderbook_asks=[],
        orderbook_bids=[],
    )

    return MarketSnapshot(
        symbol="TEST/USDC:USDC",
        ts=now,
        tf_4h=tf_4h,
        tf_1h=tf_1h,
        tf_15m=tf_15m,
        deriv=deriv,
        regime="trending",
        regime_reason="test",
    )


def test_trade_signal_does_not_expose_legacy_confidence_field():
    assert "confidence" not in TradeSignal.__dataclass_fields__


def test_trade_confidence_used_in_formatting():
    snapshot = _make_snapshot()
    signal = TradeSignal(
        symbol=snapshot.symbol,
        direction="long",
        trade_confidence=0.6,
        edge_confidence=0.4,
        setup_type="trend_long",
        core_position_pct=0.1,
        add_position_pct=0.0,
        snapshot=snapshot,
    )

    line = format_summary_line(signal.symbol, snapshot, signal)

    assert "Trade 60%" in line
