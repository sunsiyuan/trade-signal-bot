from __future__ import annotations

import csv
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import ccxt

from .config import Settings


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "oi_history"


def _normalize_symbol(symbol: str) -> str:
    return symbol.replace(":", "/").upper()


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        # Some ISO strings may end with Z; handle gracefully.
        try:
            if value.endswith("Z"):
                return datetime.fromisoformat(value[:-1]).replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def _to_float(value: object) -> Optional[float]:
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except (TypeError, ValueError):
        return None
    return None


@dataclass
class OIHistoryStore:
    base_dir: Path = DEFAULT_DATA_DIR
    exchange: str = "hyperliquid"

    def _csv_path(self, symbol: str) -> Path:
        sanitized = symbol.replace("/", "_").replace(":", "_")
        return self.base_dir / f"{self.exchange}__{sanitized}.csv"

    def _read_last_timestamp(self, path: Path) -> Optional[datetime]:
        if not path.exists():
            return None

        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            last_row = None
            for row in reader:
                last_row = row

        if not last_row:
            return None

        return _parse_timestamp(last_row.get("timestamp_utc"))

    def append_snapshot(self, row: Dict[str, object]) -> bool:
        """
        Append a snapshot row if the timestamp is newer than the existing tail.

        Returns True when a new row was written, False when skipped.
        """

        self.base_dir.mkdir(parents=True, exist_ok=True)

        path = self._csv_path(str(row.get("symbol", "")))
        ts = _parse_timestamp(str(row.get("timestamp_utc", "")))
        if ts is None:
            return False

        last_ts = self._read_last_timestamp(path)
        if last_ts is not None and last_ts >= ts:
            return False

        file_exists = path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp_utc",
                    "exchange",
                    "symbol",
                    "oi",
                    "funding_rate",
                    "mark_price",
                ],
            )
            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

        return True

    def load(self, symbol: str, hours: int = 24, now: Optional[datetime] = None) -> List[Dict]:
        now_utc = now or datetime.now(timezone.utc)
        cutoff = now_utc - timedelta(hours=hours)
        path = self._csv_path(symbol)

        if not path.exists():
            return []

        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        def _row_ts(r: Dict[str, str]) -> Optional[datetime]:
            return _parse_timestamp(r.get("timestamp_utc"))

        filtered = [r for r in rows if (_row_ts(r) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff]
        for row in filtered:
            for field in ("oi", "funding_rate", "mark_price"):
                value = row.get(field)
                parsed = _to_float(value)
                row[field] = parsed if parsed is not None else value
        filtered.sort(key=lambda r: _row_ts(r) or datetime.min.replace(tzinfo=timezone.utc))
        return filtered

    def has_pending_changes(self) -> bool:
        result = subprocess.run(
            ["git", "status", "--porcelain", str(self.base_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        return bool(result.stdout.strip())

    def commit(self) -> None:
        csv_files = list(self.base_dir.glob("*.csv"))
        if not csv_files:
            return

        subprocess.run(["git", "add", *[str(f) for f in csv_files]], check=False)
        subprocess.run(
            ["git", "commit", "-m", "data: update hourly oi snapshot"],
            check=False,
        )


def _select_funding_entry(funding_rates, symbol: str) -> Optional[Dict]:
    if isinstance(funding_rates, list):
        for entry in funding_rates:
            if isinstance(entry, dict) and _normalize_symbol(entry.get("symbol", "")) == _normalize_symbol(symbol):
                return entry
    elif isinstance(funding_rates, dict) and funding_rates.get("symbol"):
        if _normalize_symbol(funding_rates.get("symbol")) == _normalize_symbol(symbol):
            return funding_rates
    return None


def _extract_funding_rate(entry: Dict[str, object]) -> Optional[float]:
    if not isinstance(entry, dict):
        return None

    candidates = [
        entry.get("fundingRate"),
        entry.get("nextFundingRate"),
        entry.get("predictedFundingRate"),
    ]

    info = entry.get("info", {}) if isinstance(entry.get("info"), dict) else {}
    candidates.extend([
        info.get("fundingRate"),
        info.get("nextFundingRate"),
        info.get("predictedFundingRate"),
        info.get("funding"),
    ])

    for candidate in candidates:
        value = _to_float(candidate)
        if value is not None:
            return value
    return None


def _extract_open_interest(entry: Dict[str, object]) -> Optional[float]:
    if not isinstance(entry, dict):
        return None

    info = entry.get("info", {}) if isinstance(entry.get("info"), dict) else {}
    candidates = [
        entry.get("openInterest"),
        entry.get("openInterestAmount"),
        entry.get("openInterestValue"),
        info.get("openInterest"),
        info.get("openInterestAmount"),
        info.get("openInterestValue"),
    ]

    for candidate in candidates:
        value = _to_float(candidate)
        if value is not None:
            return value
    return None


def _extract_mark_price(ticker: Dict[str, object]) -> Optional[float]:
    info = ticker.get("info", {}) if isinstance(ticker.get("info"), dict) else {}
    candidates = [
        ticker.get("mark"),
        ticker.get("last"),
        info.get("markPrice"),
        info.get("markPx"),
        info.get("last"),
        info.get("lastPrice"),
    ]

    for candidate in candidates:
        value = _to_float(candidate)
        if value is not None:
            return value
    return None


class HourlyOISnapshot:
    """Collect hourly OI / funding / mark price and persist to CSV."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        store: Optional[OIHistoryStore] = None,
        exchange: Optional[ccxt.hyperliquid] = None,
    ):
        self.settings = settings or Settings()
        self.store = store or OIHistoryStore()
        self.exchange = exchange or ccxt.hyperliquid({"enableRateLimit": True})
        if exchange is None:
            self.exchange.load_markets()

    def _current_hour(self) -> datetime:
        return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    def _fetch_ticker(self, symbol: str) -> Optional[Dict]:
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as exc:  # pragma: no cover - network failure fallback
            print(f"[warn] fetch_ticker failed for {symbol}: {exc}")
            return None

    def _fetch_funding_rates(self):
        try:
            return self.exchange.fetch_funding_rates()
        except Exception as exc:  # pragma: no cover - network failure fallback
            print(f"[warn] fetch_funding_rates failed: {exc}")
            return None

    def _build_snapshot(self, symbol: str, funding_rates) -> Optional[Dict[str, object]]:
        ticker = self._fetch_ticker(symbol)
        if not isinstance(ticker, dict):
            return None

        funding_entry = _select_funding_entry(funding_rates, symbol)
        funding_rate = _extract_funding_rate(funding_entry) if funding_entry else None
        open_interest = _extract_open_interest(funding_entry or ticker)
        mark_price = _extract_mark_price(ticker)

        if funding_rate is None:
            info = ticker.get("info", {}) if isinstance(ticker.get("info"), dict) else {}
            funding_rate = _to_float(info.get("funding") or info.get("fundingRate"))

        ts = self._current_hour().isoformat()

        return {
            "timestamp_utc": ts,
            "exchange": self.store.exchange,
            "symbol": symbol,
            "oi": open_interest if open_interest is not None else "",
            "funding_rate": funding_rate if funding_rate is not None else "",
            "mark_price": mark_price if mark_price is not None else "",
        }

    def run(self) -> bool:
        funding_rates = self._fetch_funding_rates()
        wrote = False
        for symbol in self.settings.tracked_symbols:
            snapshot = self._build_snapshot(symbol, funding_rates)
            if snapshot is None:
                continue
            wrote = self.store.append_snapshot(snapshot) or wrote
        return wrote


def load_oi_history(
    symbol: str,
    hours: int = 24,
    *,
    base_dir: Optional[Path] = None,
    exchange: str = "hyperliquid",
    now: Optional[datetime] = None,
) -> List[Dict]:
    """
    从 CSV 中读取最近 N 小时的 OI 历史（按时间升序）
    若数据不足 N 小时，返回可用部分
    """

    store = OIHistoryStore(base_dir=base_dir or DEFAULT_DATA_DIR, exchange=exchange)
    return store.load(symbol, hours=hours, now=now)


def main() -> None:  # pragma: no cover - CLI entry
    snapshotter = HourlyOISnapshot()
    changed = snapshotter.run()

    if changed and snapshotter.store.has_pending_changes():
        print("[info] New hourly OI snapshots collected, committing changes...")
        # Configure git identity if not already set (useful in CI).
        subprocess.run(["git", "config", "user.name", "github-actions"], check=False)
        subprocess.run(["git", "config", "user.email", "github-actions@users.noreply.github.com"], check=False)
        snapshotter.store.commit()
    else:
        print("[info] No new hourly data; skipping commit.")


if __name__ == "__main__":  # pragma: no cover
    main()
