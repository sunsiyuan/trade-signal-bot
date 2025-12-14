from datetime import datetime, timezone, timedelta

from bot.oi_history import OIHistoryStore, load_oi_history


def test_append_snapshot_skips_same_hour(tmp_path):
    store = OIHistoryStore(base_dir=tmp_path)
    ts = datetime(2024, 1, 1, 12, tzinfo=timezone.utc).isoformat()
    row = {
        "timestamp_utc": ts,
        "exchange": "hyperliquid",
        "symbol": "HYPE/USDC:USDC",
        "oi": 123.0,
        "funding_rate": 0.01,
        "mark_price": 1.23,
    }

    assert store.append_snapshot(row) is True
    assert store.append_snapshot(row) is False

    history = load_oi_history("HYPE/USDC:USDC", hours=48, base_dir=tmp_path, now=datetime(2024, 1, 2, tzinfo=timezone.utc))
    assert len(history) == 1
    assert history[0]["oi"] == 123.0


def test_load_orders_history_chronologically(tmp_path):
    store = OIHistoryStore(base_dir=tmp_path)
    base_ts = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
    symbols = "HYPE/USDC:USDC"

    for offset in [0, 1, 2]:
        ts = (base_ts + timedelta(hours=offset)).isoformat()
        store.append_snapshot(
            {
                "timestamp_utc": ts,
                "exchange": "hyperliquid",
                "symbol": symbols,
                "oi": float(offset),
                "funding_rate": "",
                "mark_price": "",
            }
        )

    history = load_oi_history(symbols, hours=48, base_dir=tmp_path, now=base_ts + timedelta(hours=3))
    assert [row["oi"] for row in history] == [0.0, 1.0, 2.0]
