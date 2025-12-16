from datetime import datetime, timedelta, timezone

from bot import state_store


class DummySignal:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_should_send_respects_ttl(tmp_path):
    base_dir = tmp_path / "state"
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)

    state = state_store.load_state("ETH/USDC", base_dir=str(base_dir))
    allowed, _ = state_store.should_send(
        state, "sig-1", "ETH/USDC", "WATCH", now, action_hash="hash-1"
    )
    assert allowed

    state_store.mark_sent(
        state,
        "sig-1",
        "ETH/USDC",
        "WATCH",
        now,
        action_hash="hash-1",
    )

    allowed_second, info_second = state_store.should_send(
        state,
        "sig-1",
        "ETH/USDC",
        "WATCH",
        now + timedelta(seconds=1),
        action_hash="hash-1",
    )
    assert not allowed_second
    assert info_second["reason"] == "action_not_expired"

    allowed_after_ttl, _ = state_store.should_send(
        state,
        "sig-1",
        "ETH/USDC",
        "WATCH",
        now + timedelta(seconds=state_store.ACTION_TTLS["WATCH"] + 5),
        action_hash="hash-1",
    )
    assert allowed_after_ttl


def test_upgrade_allows_next_action(tmp_path):
    base_dir = tmp_path / "state"
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    symbol = "BTC/USDC"
    signal_id = "sig-upgrade"

    state = state_store.load_state(symbol, base_dir=str(base_dir))
    state_store.mark_sent(state, signal_id, symbol, "WATCH", now)

    allowed_limit, _ = state_store.should_send(
        state,
        signal_id,
        symbol,
        "LIMIT_4H",
        now + timedelta(seconds=1),
    )
    assert allowed_limit
    valid_until = now + timedelta(hours=4)
    state_store.mark_sent(
        state,
        signal_id,
        symbol,
        "LIMIT_4H",
        now,
        valid_until=valid_until,
    )

    allowed_limit_again, _ = state_store.should_send(
        state,
        signal_id,
        symbol,
        "LIMIT_4H",
        now + timedelta(hours=1),
    )
    assert not allowed_limit_again

    allowed_execute, _ = state_store.should_send(
        state,
        signal_id,
        symbol,
        "EXECUTE_NOW",
        now + timedelta(seconds=2),
    )
    assert allowed_execute


def test_compute_signal_id_is_stable():
    payload = dict(
        symbol="ETH/USDC",
        regime="trending",
        strategy="tf",
        entry=1234.5,
        sl=1200.0,
        tp1=1300.0,
        tp2=1400.0,
        tp3=1500.0,
        main_tf="4h",
        thresholds_snapshot={"gate": "ok"},
    )
    sig1 = DummySignal(**payload)
    sig2 = DummySignal(**payload)

    assert state_store.compute_signal_id(sig1) == state_store.compute_signal_id(sig2)
