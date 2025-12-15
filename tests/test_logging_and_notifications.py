import json
from datetime import datetime, timedelta, timezone

from bot.config import Settings
from bot.logging_schema import build_signal_event
from bot.main import (
    format_action_line,
    format_conditional_plan_line,
    format_summary_line,
    is_actionable,
)
from bot.models import (
    ConditionalPlan,
    DerivativeIndicators,
    MarketSnapshot,
    TimeframeIndicators,
)
from bot.signal_engine import TradeSignal


def _make_tf(tf: str, now: datetime) -> TimeframeIndicators:
    return TimeframeIndicators(
        timeframe=tf,
        close=100.0,
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
        last_candle_open_utc=now - timedelta(hours=1),
        last_candle_close_utc=now,
    )


def _make_snapshot() -> MarketSnapshot:
    now = datetime.now(timezone.utc)
    tf_4h = _make_tf("4h", now)
    tf_1h = _make_tf("1h", now)
    tf_15m = _make_tf("15m", now)
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
        regime_reason="trend strong",
    )


def _make_signal(snapshot: MarketSnapshot, **kwargs) -> TradeSignal:
    base_kwargs = dict(
        symbol=snapshot.symbol,
        direction="long",
        trade_confidence=0.8,
        edge_confidence=0.9,
        setup_type="trend_long",
        entry=99.0,
        tp1=101.0,
        tp2=102.0,
        tp3=103.0,
        sl=97.0,
        core_position_pct=0.3,
        add_position_pct=0.2,
        snapshot=snapshot,
        debug_scores={"long": 0.8, "short": 0.2},
        rejected_reasons=["test_reason"],
        thresholds_snapshot={"gate": "test_gate"},
    )
    base_kwargs.update(kwargs)
    return TradeSignal(**base_kwargs)


def test_build_signal_event_serializes():
    snapshot = _make_snapshot()
    signal = _make_signal(snapshot)
    event = build_signal_event(snapshot, signal, Settings(), exchange_id="test")

    assert event["schema_version"] == "2.1"
    assert event["signal"]["thresholds_snapshot"]
    # Ensure JSON serialization does not raise
    json.dumps(event)


def test_formatters_handle_missing_fields():
    snapshot = _make_snapshot()
    signal = _make_signal(snapshot)

    action_line = format_action_line(
        signal.symbol, snapshot, signal, action_level="EXECUTE", bias="LONG"
    )
    summary_line = format_summary_line(signal.symbol, snapshot, signal)

    assert "Levels" in action_line
    assert signal.symbol in summary_line


def test_is_actionable_classification():
    snapshot = _make_snapshot()
    settings = Settings()

    execute_signal = _make_signal(snapshot)
    assert is_actionable(execute_signal, snapshot, settings) == (
        True,
        "EXECUTE",
        "LONG",
    )

    watch_signal = _make_signal(
        snapshot,
        direction="none",
        trade_confidence=0.7,
        edge_confidence=0.85,
        debug_scores={"long": 0.6, "short": 0.4},
    )
    assert is_actionable(watch_signal, snapshot, settings) == (
        True,
        "WATCH",
        "LONG",
    )

    none_signal = _make_signal(snapshot, trade_confidence=0.3, edge_confidence=0.2)
    assert is_actionable(none_signal, snapshot, settings) == (
        False,
        "NONE",
        "NONE",
    )


def test_format_conditional_plan_line():
    snapshot = _make_snapshot()
    signal = _make_signal(snapshot)
    signal.conditional_plan = ConditionalPlan(
        execution_mode="PLACE_LIMIT_4H",
        direction="long",
        entry_price=100.0,
        valid_until_utc="2024-01-01T00:00:00",
        cancel_if={
            "invalidation_crossed": True,
            "regime_changed": True,
            "expired": True,
        },
        explain="Place 4h limit order at ideal entry",
    )

    line = format_conditional_plan_line(signal)

    assert "PLACE_LIMIT_4H" in line
    assert "LONG" in line
    assert "100.0000" in line


def test_settings_pick_up_telegram_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ACTION_BOT_TOKEN", "action-token")
    monkeypatch.setenv("TELEGRAM_ACTION_CHAT_ID", "action-chat")
    monkeypatch.setenv("TELEGRAM_SUMMARY_BOT_TOKEN", "summary-token")
    monkeypatch.setenv("TELEGRAM_SUMMARY_CHAT_ID", "summary-chat")

    settings = Settings()

    assert settings.telegram_action_token == "action-token"
    assert settings.telegram_action_chat_id == "action-chat"
    assert settings.telegram_summary_token == "summary-token"
    assert settings.telegram_summary_chat_id == "summary-chat"
