from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .types import Candle, OptionalOI


class JSONLDataStore:
    def __init__(self, base_dir: str = "data/market_data"):
        self.base_dir = base_dir
        self.meta_path = os.path.join(self.base_dir, "meta.json")
        os.makedirs(self.base_dir, exist_ok=True)

    @staticmethod
    def _safe_symbol(symbol: str) -> str:
        return symbol.replace("/", "_").replace(":", "_")

    def _symbol_dir(self, symbol: str) -> str:
        path = os.path.join(self.base_dir, self._safe_symbol(symbol))
        os.makedirs(path, exist_ok=True)
        return path

    def _candles_path(self, symbol: str, timeframe: str) -> str:
        return os.path.join(self._symbol_dir(symbol), f"candles_{timeframe}.jsonl")

    def _oi_path(self, symbol: str) -> str:
        return os.path.join(self._symbol_dir(symbol), "open_interest_1h.jsonl")

    def _load_meta(self) -> Dict[str, Dict[str, int]]:
        if not os.path.exists(self.meta_path):
            return {}
        with open(self.meta_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save_meta(self, meta: Dict[str, Dict[str, int]]) -> None:
        with open(self.meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

    def get_max_ts(self, symbol: str, timeframe: str) -> Optional[int]:
        meta = self._load_meta()
        return meta.get(symbol, {}).get(timeframe)

    def update_max_ts(self, symbol: str, timeframe: str, ts_ms: int) -> None:
        meta = self._load_meta()
        meta.setdefault(symbol, {})[timeframe] = ts_ms
        self._save_meta(meta)

    def load_candles(
        self,
        symbol: str,
        timeframe: str,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> pd.DataFrame:
        path = self._candles_path(symbol, timeframe)
        if not os.path.exists(path):
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        records = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                entry = json.loads(line)
                ts = int(entry["ts_open_ms"])
                if start_ms is not None and ts < start_ms:
                    continue
                if end_ms is not None and ts > end_ms:
                    continue
                records.append(entry)

        if not records:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["ts_open_ms"], unit="ms", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def upsert_candles(self, symbol: str, timeframe: str, candles: Iterable[Candle]) -> int:
        path = self._candles_path(symbol, timeframe)
        existing = {}

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    existing[int(entry["ts_open_ms"])] = entry

        new_records = 0
        for candle in candles:
            data = asdict(candle)
            ts = int(candle.ts_open_ms)
            if ts not in existing:
                new_records += 1
            existing[ts] = data

        if existing:
            sorted_records = [existing[k] for k in sorted(existing.keys())]
            with open(path, "w", encoding="utf-8") as fh:
                for entry in sorted_records:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.update_max_ts(symbol, timeframe, sorted(existing.keys())[-1])

        return new_records

    def load_open_interest(
        self, symbol: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None
    ) -> pd.DataFrame:
        path = self._oi_path(symbol)
        if not os.path.exists(path):
            return pd.DataFrame(columns=["timestamp", "open_interest", "oi_change_24h"])

        records = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                entry = json.loads(line)
                ts = int(entry["ts_ms"])
                if start_ms is not None and ts < start_ms:
                    continue
                if end_ms is not None and ts > end_ms:
                    continue
                records.append(entry)

        if not records:
            return pd.DataFrame(columns=["timestamp", "open_interest", "oi_change_24h"])

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df[["timestamp", "open_interest", "oi_change_24h"]]

    def upsert_open_interest(self, symbol: str, rows: Iterable[OptionalOI]) -> int:
        path = self._oi_path(symbol)
        existing = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    existing[int(entry["ts_ms"])] = entry

        new_records = 0
        for row in rows:
            data = asdict(row)
            ts = int(row.ts_ms)
            if ts not in existing:
                new_records += 1
            existing[ts] = data

        if existing:
            sorted_records = [existing[k] for k in sorted(existing.keys())]
            with open(path, "w", encoding="utf-8") as fh:
                for entry in sorted_records:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return new_records
