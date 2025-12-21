from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from ..config import Settings
from ..signal_engine import TradeSignal


@dataclass
class GateDecision:
    decision: str
    reason: str
    dedup_key: Optional[str]
    cooldown_remaining_sec: Optional[float]
    in_position: bool
    forced_close: bool
    scope_key: Optional[str] = None


@dataclass
class Position:
    position_id: str
    open_ts: float
    direction: str
    size: float
    entry_price: float


@dataclass
class ClosePositionAction:
    scope_key: str
    method: str
    requested_ts: float


@dataclass
class ExecuteAction:
    scope_key: str
    requested_ts: float


class DedupStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or "data/gating.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dedup (
                dedup_key TEXT PRIMARY KEY,
                seen_ts REAL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cooldown (
                scope_key TEXT PRIMARY KEY,
                last_exec_ts REAL
            )
            """
        )

    def seen(self, dedup_key: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM dedup WHERE dedup_key = ? LIMIT 1", (dedup_key,)
        ).fetchone()
        return row is not None

    def mark_seen(self, dedup_key: str, ts: float) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO dedup (dedup_key, seen_ts) VALUES (?, ?)",
                (dedup_key, ts),
            )

    def get_last_exec_ts(self, scope_key: str) -> Optional[float]:
        row = self._conn.execute(
            "SELECT last_exec_ts FROM cooldown WHERE scope_key = ?",
            (scope_key,),
        ).fetchone()
        return row[0] if row else None

    def set_last_exec_ts(self, scope_key: str, ts: float) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO cooldown (scope_key, last_exec_ts) VALUES (?, ?)",
                (scope_key, ts),
            )


class PositionStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or "data/gating.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                scope_key TEXT PRIMARY KEY,
                position_id TEXT,
                open_ts REAL,
                direction TEXT,
                size REAL,
                entry_price REAL
            )
            """
        )

    def get_open_position(self, scope_key: str) -> Optional[Position]:
        row = self._conn.execute(
            """
            SELECT position_id, open_ts, direction, size, entry_price
            FROM positions WHERE scope_key = ?
            """,
            (scope_key,),
        ).fetchone()
        if not row:
            return None
        return Position(
            position_id=row[0],
            open_ts=row[1],
            direction=row[2],
            size=row[3],
            entry_price=row[4],
        )

    def set_open_position(self, scope_key: str, position: Position) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO positions
                (scope_key, position_id, open_ts, direction, size, entry_price)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    scope_key,
                    position.position_id,
                    position.open_ts,
                    position.direction,
                    position.size,
                    position.entry_price,
                ),
            )

    def clear_open_position(self, scope_key: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM positions WHERE scope_key = ?", (scope_key,))


def _timeframe_to_seconds(tf: Optional[str], fallback: int) -> int:
    if not tf:
        return fallback
    tf = tf.lower()
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600
    return fallback


def _resolve_signal_ts(signal: TradeSignal, now_ts: Optional[datetime]) -> datetime:
    if getattr(signal, "snapshot", None) and getattr(signal.snapshot, "ts", None):
        return signal.snapshot.ts
    if now_ts:
        return now_ts
    return datetime.now(timezone.utc)


def _round_price(price: Optional[float]) -> Optional[float]:
    if price is None:
        return None
    precision = 1 if abs(price) >= 1000 else 2
    return round(price, precision)


def _build_scope_key(
    exchange_id: str, symbol: str, direction: str, scope_mode: str
) -> str:
    if scope_mode == "symbol":
        return f"{exchange_id}|{symbol}"
    return f"{exchange_id}|{symbol}|{direction}"


def _build_dedup_key(
    signal: TradeSignal,
    exchange_id: str,
    plan_type: str,
    settings: Settings,
    signal_ts: datetime,
) -> str:
    key_mode = getattr(settings, "GATE_DEDUP_KEY_MODE", "content_hash")
    if key_mode == "signal_id" and signal.signal_id:
        return signal.signal_id

    tf = None
    if getattr(signal, "snapshot", None) and getattr(signal.snapshot, "tf_15m", None):
        tf = getattr(signal.snapshot.tf_15m, "timeframe", None)
    tf = tf or getattr(settings, "tf_15m", None)

    window = _timeframe_to_seconds(tf, settings.GATE_DEDUP_WINDOW_SEC)
    bucket = int(signal_ts.timestamp() // window)

    payload = {
        "exchange": exchange_id,
        "symbol": signal.symbol,
        "direction": signal.direction,
        "setup_type": signal.setup_type,
        "plan_type": plan_type,
        "tf": tf,
        "entry": _round_price(signal.entry),
        "sl": _round_price(signal.sl),
        "tp": _round_price(signal.tp1),
        "bucket": bucket,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return f"{digest[:16]}"


def should_execute(
    signal: TradeSignal,
    plan_type: str,
    settings: Settings,
    dedup_store: DedupStore,
    position_store: PositionStore,
    exchange_id: str = "",
    now_ts: Optional[datetime] = None,
) -> GateDecision:
    if not getattr(settings, "GATE_ENABLE", True):
        return GateDecision(
            decision="EXECUTE",
            reason="gate_disabled",
            dedup_key=None,
            cooldown_remaining_sec=None,
            in_position=False,
            forced_close=False,
            scope_key=None,
        )

    signal_ts = _resolve_signal_ts(signal, now_ts)
    dedup_key = _build_dedup_key(signal, exchange_id, plan_type, settings, signal_ts)
    scope_key = _build_scope_key(
        exchange_id, signal.symbol, signal.direction, settings.GATE_SCOPE
    )

    if dedup_store.seen(dedup_key):
        return GateDecision(
            decision="SKIP_DUP",
            reason="dedup_key_seen",
            dedup_key=dedup_key,
            cooldown_remaining_sec=None,
            in_position=False,
            forced_close=False,
            scope_key=scope_key,
        )

    dedup_store.mark_seen(dedup_key, signal_ts.timestamp())

    last_exec_ts = dedup_store.get_last_exec_ts(scope_key)
    if last_exec_ts is not None:
        cooldown_sec = getattr(settings, "GATE_COOLDOWN_SEC", 0)
        remaining = last_exec_ts + cooldown_sec - signal_ts.timestamp()
        if remaining > 0:
            return GateDecision(
                decision="SKIP_COOLDOWN",
                reason="cooldown_active",
                dedup_key=dedup_key,
                cooldown_remaining_sec=remaining,
                in_position=False,
                forced_close=False,
                scope_key=scope_key,
            )

    position = position_store.get_open_position(scope_key)
    in_position = position is not None

    if in_position:
        same_direction = position.direction == signal.direction if position else False
        refresh_same_direction = getattr(settings, "GATE_REFRESH_SAME_DIRECTION", False)
        allow_force_close = getattr(settings, "GATE_FORCE_CLOSE_ON_NEW_SIGNAL", True)
        if plan_type != "EXECUTE_NOW":
            return GateDecision(
                decision="SKIP_IN_POSITION",
                reason="in_position_non_execute",
                dedup_key=dedup_key,
                cooldown_remaining_sec=None,
                in_position=True,
                forced_close=False,
                scope_key=scope_key,
            )
        if same_direction and not refresh_same_direction:
            return GateDecision(
                decision="SKIP_IN_POSITION",
                reason="same_direction_skip",
                dedup_key=dedup_key,
                cooldown_remaining_sec=None,
                in_position=True,
                forced_close=False,
                scope_key=scope_key,
            )
        if allow_force_close:
            return GateDecision(
                decision="CLOSE_THEN_EXEC",
                reason="force_close_before_execute",
                dedup_key=dedup_key,
                cooldown_remaining_sec=None,
                in_position=True,
                forced_close=True,
                scope_key=scope_key,
            )
        return GateDecision(
            decision="SKIP_IN_POSITION",
            reason="in_position_skip",
            dedup_key=dedup_key,
            cooldown_remaining_sec=None,
            in_position=True,
            forced_close=False,
            scope_key=scope_key,
        )

    return GateDecision(
        decision="EXECUTE",
        reason="ok",
        dedup_key=dedup_key,
        cooldown_remaining_sec=None,
        in_position=False,
        forced_close=False,
        scope_key=scope_key,
    )


def apply_gate(
    signal: TradeSignal,
    plan_type: str,
    settings: Settings,
    dedup_store: DedupStore,
    position_store: PositionStore,
    exchange_id: str = "",
    now_ts: Optional[datetime] = None,
) -> Tuple[Optional[ClosePositionAction], Optional[ExecuteAction], GateDecision]:
    decision = should_execute(
        signal=signal,
        plan_type=plan_type,
        settings=settings,
        dedup_store=dedup_store,
        position_store=position_store,
        exchange_id=exchange_id,
        now_ts=now_ts,
    )

    if decision.decision == "EXECUTE":
        return (
            None,
            ExecuteAction(scope_key=decision.scope_key or "", requested_ts=_resolve_ts(now_ts)),
            decision,
        )
    if decision.decision == "CLOSE_THEN_EXEC":
        close_action = ClosePositionAction(
            scope_key=decision.scope_key or "",
            method=getattr(settings, "GATE_CLOSE_METHOD", "market"),
            requested_ts=_resolve_ts(now_ts),
        )
        execute_action = ExecuteAction(
            scope_key=decision.scope_key or "",
            requested_ts=_resolve_ts(now_ts),
        )
        return close_action, execute_action, decision

    return None, None, decision


def _resolve_ts(now_ts: Optional[datetime]) -> float:
    if now_ts:
        return now_ts.timestamp()
    return datetime.now(timezone.utc).timestamp()
