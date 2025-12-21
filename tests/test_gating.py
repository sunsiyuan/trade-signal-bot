from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from bot.config import Settings
from bot.execution.gating import DedupStore, PositionStore, apply_gate
from bot.signal_engine import TradeSignal
from bot.execution.gating import Position as GatePosition


def _signal(ts: datetime, **overrides) -> TradeSignal:
    signal = TradeSignal(
        symbol=overrides.get("symbol", "BTC/USDC:USDC"),
        direction=overrides.get("direction", "long"),
        signal_id=overrides.get("signal_id"),
        setup_type=overrides.get("setup_type", "mr"),
        entry=overrides.get("entry", 100.0),
        sl=overrides.get("sl", 95.0),
        tp1=overrides.get("tp1", 110.0),
    )
    signal.snapshot = SimpleNamespace(
        ts=ts, tf_15m=SimpleNamespace(timeframe="15m")
    )
    return signal


def _stores(tmp_path):
    db_path = tmp_path / "gating.sqlite"
    return DedupStore(str(db_path)), PositionStore(str(db_path))


def test_dedup_same_bucket_skips(tmp_path):
    settings = Settings()
    dedup_store, position_store = _stores(tmp_path)
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

    signal_a = _signal(ts, signal_id="a")
    _, _, decision_a = apply_gate(
        signal=signal_a,
        plan_type="EXECUTE_NOW",
        settings=settings,
        dedup_store=dedup_store,
        position_store=position_store,
        exchange_id="test",
        now_ts=ts,
    )
    assert decision_a.decision == "EXECUTE"

    signal_b = _signal(ts + timedelta(seconds=60), signal_id="b")
    _, _, decision_b = apply_gate(
        signal=signal_b,
        plan_type="EXECUTE_NOW",
        settings=settings,
        dedup_store=dedup_store,
        position_store=position_store,
        exchange_id="test",
        now_ts=ts + timedelta(seconds=60),
    )
    assert decision_b.decision == "SKIP_DUP"


def test_cooldown_skips_repeated_signal(tmp_path):
    settings = Settings()
    settings.GATE_COOLDOWN_SEC = 3600
    dedup_store, position_store = _stores(tmp_path)
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

    signal_a = _signal(ts, signal_id="a")
    _, _, decision_a = apply_gate(
        signal=signal_a,
        plan_type="EXECUTE_NOW",
        settings=settings,
        dedup_store=dedup_store,
        position_store=position_store,
        exchange_id="test",
        now_ts=ts,
    )
    assert decision_a.decision == "EXECUTE"
    dedup_store.set_last_exec_ts(decision_a.scope_key or "", ts.timestamp())

    signal_b = _signal(ts + timedelta(seconds=120), setup_type="mr2")
    _, _, decision_b = apply_gate(
        signal=signal_b,
        plan_type="EXECUTE_NOW",
        settings=settings,
        dedup_store=dedup_store,
        position_store=position_store,
        exchange_id="test",
        now_ts=ts + timedelta(seconds=120),
    )
    assert decision_b.decision == "SKIP_COOLDOWN"


def test_close_then_execute_after_cooldown(tmp_path):
    settings = Settings()
    settings.GATE_COOLDOWN_SEC = 60
    settings.GATE_FORCE_CLOSE_ON_NEW_SIGNAL = True
    settings.GATE_SCOPE = "symbol"
    dedup_store, position_store = _stores(tmp_path)
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

    scope_key = "test|BTC/USDC:USDC"
    position_store.set_open_position(
        scope_key,
        GatePosition(
            position_id="pos-1",
            open_ts=ts.timestamp(),
            direction="long",
            size=1.0,
            entry_price=100.0,
        ),
    )
    dedup_store.set_last_exec_ts(scope_key, ts.timestamp() - 120)

    signal = _signal(ts + timedelta(seconds=3600), direction="short", setup_type="tf")
    _, _, decision = apply_gate(
        signal=signal,
        plan_type="EXECUTE_NOW",
        settings=settings,
        dedup_store=dedup_store,
        position_store=position_store,
        exchange_id="test",
        now_ts=ts + timedelta(seconds=3600),
    )
    assert decision.decision == "CLOSE_THEN_EXEC"
    assert decision.forced_close is True


def test_scope_and_force_close_options(tmp_path):
    settings = Settings()
    settings.GATE_SCOPE = "symbol"
    settings.GATE_FORCE_CLOSE_ON_NEW_SIGNAL = False
    dedup_store, position_store = _stores(tmp_path)
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

    scope_key = "test|BTC/USDC:USDC"
    position_store.set_open_position(
        scope_key,
        GatePosition(
            position_id="pos-2",
            open_ts=ts.timestamp(),
            direction="long",
            size=1.0,
            entry_price=100.0,
        ),
    )

    signal = _signal(ts + timedelta(seconds=5000), direction="short", setup_type="tf2")
    _, _, decision = apply_gate(
        signal=signal,
        plan_type="EXECUTE_NOW",
        settings=settings,
        dedup_store=dedup_store,
        position_store=position_store,
        exchange_id="test",
        now_ts=ts + timedelta(seconds=5000),
    )
    assert decision.decision == "SKIP_IN_POSITION"
