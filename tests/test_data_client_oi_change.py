from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

import math

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot.config import Settings
from bot.data_client import HyperliquidDataClient
from bot.oi_history import OIHistoryStore


class DummyExchange:
    def market(self, symbol):
        return {"symbol": symbol, "id": symbol}

    def fetch_ticker(self, symbol):
        return {"info": {"openInterest": "200", "markPrice": "10"}}

    def fetch_funding_rates(self, *args, **kwargs):
        return []

    def fetch_open_interest_history(self, *args, **kwargs):
        return []

    def fetch_order_book(self, *args, **kwargs):
        return {"asks": [[10.0, 1.0]], "bids": [[9.5, 1.5]]}

    def price_to_precision(self, symbol, price):
        return price


def _freeze_datetime(monkeypatch, fixed_now: datetime):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz:
                return fixed_now.astimezone(tz)
            return fixed_now.replace(tzinfo=None)

    monkeypatch.setattr("bot.data_client.datetime", FrozenDatetime)


def _build_store(base_dir, symbol: str, fixed_now: datetime, entries):
    store = OIHistoryStore(base_dir=base_dir)
    for hours_ago, oi in entries:
        ts = (
            fixed_now
            - timedelta(hours=hours_ago)
        ).replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        store.append_snapshot(
            {
                "timestamp_utc": ts.isoformat(),
                "exchange": "hyperliquid",
                "symbol": symbol,
                "oi": oi,
                "funding_rate": "",
                "mark_price": "",
            }
        )
    return store


def test_compute_oi_change_uses_older_than_24h_snapshot(tmp_path, monkeypatch):
    symbol = "HYPE/USDC:USDC"
    fixed_now = datetime(2024, 1, 2, 12, tzinfo=timezone.utc)
    _freeze_datetime(monkeypatch, fixed_now)

    store = _build_store(tmp_path, symbol, fixed_now, [(25, 100.0), (5, 150.0)])

    def fake_load(symbol_arg, hours=30, base_dir=None, exchange="hyperliquid", now=None):
        return store.load(symbol_arg, hours=hours, now=now or fixed_now)

    monkeypatch.setattr("bot.data_client.load_oi_history", fake_load)

    client = HyperliquidDataClient(Settings(symbol=symbol), exchange=DummyExchange())

    change = client._compute_oi_change_24h(current_oi=200.0)

    assert change == 100.0


def test_compute_oi_change_skips_when_no_snapshot_24h_old(tmp_path, monkeypatch):
    symbol = "HYPE/USDC:USDC"
    fixed_now = datetime(2024, 1, 2, 12, tzinfo=timezone.utc)
    _freeze_datetime(monkeypatch, fixed_now)

    store = _build_store(tmp_path, symbol, fixed_now, [(5, 150.0)])

    def fake_load(symbol_arg, hours=30, base_dir=None, exchange="hyperliquid", now=None):
        return store.load(symbol_arg, hours=hours, now=now or fixed_now)

    monkeypatch.setattr("bot.data_client.load_oi_history", fake_load)

    client = HyperliquidDataClient(Settings(symbol=symbol), exchange=DummyExchange())

    change = client._compute_oi_change_24h(current_oi=200.0)

    assert change is None or math.isnan(change)


def test_fetch_derivative_indicators_sets_oi_change_24h(monkeypatch):
    symbol = "HYPE/USDC:USDC"
    client = HyperliquidDataClient(Settings(symbol=symbol), exchange=DummyExchange())

    monkeypatch.setattr(client, "_compute_oi_change_24h", lambda current_oi: 12.5)

    deriv = client.fetch_derivative_indicators()

    assert deriv.open_interest == 200.0
    assert deriv.oi_change_24h == 12.5
