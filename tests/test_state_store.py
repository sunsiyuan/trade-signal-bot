from datetime import datetime, timedelta, timezone

from bot import state_store


class DummySignal:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class DummySettings:
    def __init__(self, price_quantization):
        self.price_quantization = price_quantization


def test_watch_signals_are_not_cached(tmp_path):
    base_dir = tmp_path / "state"
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)

    state = state_store.load_state("ETH/USDC", base_dir=str(base_dir))
    allowed, info = state_store.should_send(
        state, "sig-1", "ETH/USDC", "WATCH", now, action_hash="hash-1"
    )
    assert allowed
    assert info["reason"] == "watch_no_cache"

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
    assert allowed_second
    assert info_second["reason"] == "watch_no_cache"
    assert state.get("signals") == {}


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


def test_compute_signal_id_quantizes_prices():
    price_quantization = {"ETH": 10}
    settings = DummySettings(price_quantization)

    sig1 = DummySignal(
        symbol="ETH/USDC",
        entry=2010.1,
        sl=1900.4,
        tp1=2100.2,
        tp2=2200.6,
        tp3=2300.5,
        settings=settings,
    )
    sig2 = DummySignal(
        symbol="ETH/USDC",
        entry=2014.9,
        sl=1899.6,
        tp1=2104.6,
        tp2=2204.4,
        tp3=2304.4,
        settings=settings,
    )

    assert (
        state_store.compute_signal_id(sig1, price_quantization=price_quantization)
        == state_store.compute_signal_id(sig2, price_quantization=price_quantization)
    )
