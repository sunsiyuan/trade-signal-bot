from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

from ..config import Settings
from ..execution.gating import DedupStore, PositionStore, apply_gate, Position as GatePosition
from ..models import ConditionalPlan, ExecutionIntent
from ..signal_engine import SignalEngine
from .data_store import JSONLDataStore
from .outcome import summarize_trades
from .snapshot_builder import DataWindow, SnapshotBuilder
from .types import Order, Position, TradeRecord, BacktestResult


@dataclass
class BacktestExecutionState:
    executed_signal_ids: Dict[str, int] = field(default_factory=dict)
    open_orders_by_signal_id: Dict[str, Order] = field(default_factory=dict)
    closed_signal_ids: Dict[str, int] = field(default_factory=dict)
    open_positions: Dict[str, Position] = field(default_factory=dict)


def _signal_valid_until_ms(snapshot_ts: datetime, intent: Optional[ExecutionIntent]) -> int:
    ttl_hours = getattr(intent, "ttl_hours", 4) if intent else 4
    return int((snapshot_ts + timedelta(hours=ttl_hours)).timestamp() * 1000)


def _entry_from_plan(plan: ConditionalPlan, fallback_price: float) -> float:
    return plan.entry_price if plan.entry_price is not None else fallback_price


def _direction_sign(direction: str) -> int:
    return 1 if direction == "long" else -1


def _pnl(entry: float, exit_price: float, direction: str) -> tuple[float, float]:
    sign = _direction_sign(direction)
    pnl_abs = (exit_price - entry) * sign
    pnl_pct = pnl_abs / entry if entry else 0.0
    return pnl_abs, pnl_pct


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _scope_key(settings: Settings, exchange_id: str, symbol: str, direction: str) -> str:
    if settings.GATE_SCOPE == "symbol":
        return f"{exchange_id}|{symbol}"
    return f"{exchange_id}|{symbol}|{direction}"


def _resample_from_15m(df_15m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df_15m.empty:
        return df_15m
    df = df_15m.copy().set_index("timestamp")
    resampled = df.resample(timeframe, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    resampled = resampled.dropna().reset_index()
    return resampled


def _update_trade_record(trades: List[TradeRecord], trade_id: str, **updates) -> None:
    for record in trades:
        if record.trade_id == trade_id:
            for key, value in updates.items():
                setattr(record, key, value)
            break


def run_backtest(
    symbol: str,
    data_store: JSONLDataStore,
    output_dir: str,
    mode: str,
    settings: Optional[Settings] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> BacktestResult:
    settings = settings or Settings()
    signal_engine = SignalEngine(settings=settings)

    os.makedirs(output_dir, exist_ok=True)
    gate_db_path = os.path.join(output_dir, "gating.sqlite")
    dedup_store = DedupStore(gate_db_path)
    position_store = PositionStore(gate_db_path)

    df_15m = data_store.load_candles(symbol, "15m")
    df_1h = data_store.load_candles(symbol, "1h")
    df_4h = data_store.load_candles(symbol, "4h")

    if df_15m.empty:
        raise ValueError(f"No 15m candles for {symbol}")

    if df_1h.empty:
        df_1h = _resample_from_15m(df_15m, "1h")
    if df_4h.empty:
        df_4h = _resample_from_15m(df_15m, "4h")

    oi_df = data_store.load_open_interest(symbol)

    if start:
        start_dt = datetime.fromisoformat(start)
        df_15m = df_15m[df_15m["timestamp"] >= start_dt]
    if end:
        end_dt = datetime.fromisoformat(end)
        df_15m = df_15m[df_15m["timestamp"] <= end_dt]

    window = DataWindow(tf_15m=df_15m, tf_1h=df_1h, tf_4h=df_4h, oi_1h=oi_df)
    builder = SnapshotBuilder(settings)
    state = BacktestExecutionState()

    trades: List[TradeRecord] = []

    for idx in range(len(df_15m)):
        candle = df_15m.iloc[idx]
        candle_open = candle["timestamp"].to_pydatetime()
        candle_close = candle_open + timedelta(minutes=15)

        snapshot = builder.build_snapshot(symbol, window, candle_close)
        if snapshot is None:
            continue

        signal = signal_engine.generate_signal(snapshot)
        plan = signal.conditional_plan
        intent = signal.execution_intent

        data_flags = getattr(snapshot, "data_flags", {"has_oi": False, "has_orderbook": False})

        if plan is None or plan.execution_mode == "WATCH_ONLY":
            continue

        plan_type = plan.execution_mode
        signal_id = signal.signal_id or ""
        valid_until_ms = _signal_valid_until_ms(snapshot.ts, intent)

        gate_decision = None
        should_gate = plan_type == "EXECUTE_NOW" or (
            plan_type == "PLACE_LIMIT_4H" and mode in {"execute_now_and_limit4h"}
        )
        if should_gate:
            _, _, gate_decision = apply_gate(
                signal=signal,
                plan_type=plan_type,
                settings=settings,
                dedup_store=dedup_store,
                position_store=position_store,
                exchange_id="backtest",
                now_ts=snapshot.ts,
            )

            if gate_decision.decision in {
                "SKIP_DUP",
                "SKIP_COOLDOWN",
                "SKIP_IN_POSITION",
            }:
                trades.append(
                    TradeRecord(
                        trade_id=f"skip-{signal_id}-{_to_ms(snapshot.ts)}",
                        symbol=symbol,
                        signal_id=signal_id,
                        plan_type=plan_type,
                        direction=signal.direction,
                        setup_type=signal.setup_type,
                        order_created_ts=None,
                        filled_ts=None,
                        filled_price=None,
                        expired=False,
                        exit_ts=None,
                        exit_price=None,
                        exit_reason=None,
                        pnl_abs=None,
                        pnl_pct=None,
                        decision_trace=signal.conditional_plan_debug,
                        data_coverage=data_flags,
                        duplicate_skipped=gate_decision.decision == "SKIP_DUP",
                        duplicate_reason=gate_decision.reason
                        if gate_decision.decision == "SKIP_DUP"
                        else None,
                        cooldown_skipped=gate_decision.decision == "SKIP_COOLDOWN",
                        cooldown_remaining_sec=gate_decision.cooldown_remaining_sec,
                        in_position=gate_decision.in_position,
                        forced_close=gate_decision.forced_close,
                        gate_decision=gate_decision.decision,
                        scope_key=gate_decision.scope_key,
                        dedup_key=gate_decision.dedup_key,
                    )
                )
                continue

        if plan_type == "EXECUTE_NOW":
            if idx + 1 >= len(df_15m):
                continue
            next_candle = df_15m.iloc[idx + 1]
            entry_price = float(next_candle["open"])
            entry_ts = _to_ms(next_candle["timestamp"].to_pydatetime())

            if gate_decision and gate_decision.decision == "CLOSE_THEN_EXEC":
                close_ts = entry_ts
                scope_key = gate_decision.scope_key or _scope_key(
                    settings, "backtest", symbol, signal.direction
                )
                position_info = position_store.get_open_position(scope_key)
                if position_info:
                    open_position = state.open_positions.get(position_info.position_id)
                    if open_position:
                        exit_price = float(next_candle["open"])
                        exit_ts = close_ts
                        pnl_abs, pnl_pct = _pnl(
                            open_position.entry_price, exit_price, open_position.direction
                        )
                        open_position.status = "closed"
                        open_position.exit_ts = exit_ts
                        open_position.exit_price = exit_price
                        open_position.exit_reason = "manual"
                        _update_trade_record(
                            trades,
                            open_position.trade_id,
                            exit_ts=exit_ts,
                            exit_price=exit_price,
                            exit_reason="manual",
                            pnl_abs=pnl_abs,
                            pnl_pct=pnl_pct,
                        )
                        del state.open_positions[open_position.trade_id]
                    if settings.GATE_ENABLE:
                        position_store.clear_open_position(scope_key)
                entry_ts = close_ts + settings.GATE_MIN_TIME_BETWEEN_CLOSE_AND_OPEN_MS

            trade_id = str(uuid.uuid4())
            position = Position(
                trade_id=trade_id,
                signal_id=signal_id,
                symbol=symbol,
                direction=signal.direction,
                entry_ts=entry_ts,
                entry_price=entry_price,
                sl=signal.sl,
                tp=signal.tp1,
            )
            state.open_positions[trade_id] = position
            scope_key = (
                gate_decision.scope_key
                if gate_decision and gate_decision.scope_key
                else _scope_key(settings, "backtest", symbol, signal.direction)
            )
            if settings.GATE_ENABLE:
                position_store.set_open_position(
                    scope_key,
                    position=GatePosition(
                        position_id=trade_id,
                        open_ts=entry_ts / 1000,
                        direction=signal.direction,
                        size=1.0,
                        entry_price=entry_price,
                    ),
                )
                dedup_store.set_last_exec_ts(scope_key, entry_ts / 1000)

            trades.append(
                TradeRecord(
                    trade_id=trade_id,
                    symbol=symbol,
                    signal_id=signal_id,
                    plan_type=plan_type,
                    direction=signal.direction,
                    setup_type=signal.setup_type,
                    order_created_ts=_to_ms(snapshot.ts),
                    filled_ts=entry_ts,
                    filled_price=entry_price,
                    expired=False,
                    exit_ts=None,
                    exit_price=None,
                    exit_reason=None,
                    pnl_abs=None,
                    pnl_pct=None,
                    decision_trace=signal.conditional_plan_debug,
                    data_coverage=data_flags,
                    in_position=gate_decision.in_position if gate_decision else False,
                    forced_close=gate_decision.forced_close if gate_decision else False,
                    gate_decision=gate_decision.decision if gate_decision else None,
                    scope_key=gate_decision.scope_key if gate_decision else None,
                    dedup_key=gate_decision.dedup_key if gate_decision else None,
                )
            )

        if plan_type == "PLACE_LIMIT_4H" and mode in {"execute_now_and_limit4h"}:
            entry_price = _entry_from_plan(plan, snapshot.tf_15m.close)
            order_id = str(uuid.uuid4())
            order = Order(
                order_id=order_id,
                signal_id=signal_id,
                symbol=symbol,
                plan_type=plan_type,
                direction=signal.direction,
                entry_price=entry_price,
                created_ts=_to_ms(snapshot.ts),
                valid_until_ts=valid_until_ms,
                sl=signal.sl,
                tp=signal.tp1,
                setup_type=signal.setup_type,
                decision_trace=signal.conditional_plan_debug,
                data_coverage=data_flags,
            )
            state.open_orders_by_signal_id[signal_id] = order
            trades.append(
                TradeRecord(
                    trade_id=order_id,
                    symbol=symbol,
                    signal_id=signal_id,
                    plan_type=plan_type,
                    direction=signal.direction,
                    setup_type=signal.setup_type,
                    order_created_ts=_to_ms(snapshot.ts),
                    filled_ts=None,
                    filled_price=None,
                    expired=False,
                    exit_ts=None,
                    exit_price=None,
                    exit_reason=None,
                    pnl_abs=None,
                    pnl_pct=None,
                    decision_trace=signal.conditional_plan_debug,
                    data_coverage=data_flags,
                    in_position=gate_decision.in_position if gate_decision else False,
                    forced_close=gate_decision.forced_close if gate_decision else False,
                    gate_decision=gate_decision.decision if gate_decision else None,
                    scope_key=gate_decision.scope_key if gate_decision else None,
                    dedup_key=gate_decision.dedup_key if gate_decision else None,
                )
            )

        for signal_key, order in list(state.open_orders_by_signal_id.items()):
            if order.status != "open":
                continue
            if candle_open <= datetime.fromtimestamp(order.created_ts / 1000, tz=timezone.utc):
                continue
            if _to_ms(candle_open) > order.valid_until_ts:
                order.status = "expired"
                order.expired_ts = _to_ms(candle_open)
                state.closed_signal_ids[signal_key] = order.expired_ts
                del state.open_orders_by_signal_id[signal_key]
                _update_trade_record(
                    trades,
                    order.order_id,
                    expired=True,
                    exit_ts=order.expired_ts,
                    exit_reason="expired",
                )
                continue

            if order.entry_price is None:
                continue

            triggered = False
            if order.direction == "long" and candle["low"] <= order.entry_price:
                triggered = True
            if order.direction == "short" and candle["high"] >= order.entry_price:
                triggered = True

            if triggered:
                order.status = "filled"
                order.filled_ts = _to_ms(candle_open)
                order.filled_price = order.entry_price
                position = Position(
                    trade_id=order.order_id,
                    signal_id=order.signal_id,
                    symbol=order.symbol,
                    direction=order.direction,
                    entry_ts=order.filled_ts,
                    entry_price=order.filled_price,
                    sl=order.sl,
                    tp=order.tp,
                )
                state.open_positions[position.trade_id] = position
                scope_key = _scope_key(settings, "backtest", order.symbol, order.direction)
                if settings.GATE_ENABLE:
                    position_store.set_open_position(
                        scope_key,
                        position=GatePosition(
                            position_id=order.order_id,
                            open_ts=order.filled_ts / 1000,
                            direction=order.direction,
                            size=1.0,
                            entry_price=order.filled_price or 0.0,
                        ),
                    )
                    dedup_store.set_last_exec_ts(scope_key, order.filled_ts / 1000)
                del state.open_orders_by_signal_id[signal_key]
                _update_trade_record(
                    trades,
                    order.order_id,
                    filled_ts=order.filled_ts,
                    filled_price=order.filled_price,
                )

        for trade_id, position in list(state.open_positions.items()):
            if position.status != "open":
                continue
            if candle_open <= datetime.fromtimestamp(position.entry_ts / 1000, tz=timezone.utc):
                continue

            exit_price = None
            exit_reason = None

            if position.direction == "long":
                if position.sl is not None and candle["low"] <= position.sl:
                    exit_price = position.sl
                    exit_reason = "sl"
                elif position.tp is not None and candle["high"] >= position.tp:
                    exit_price = position.tp
                    exit_reason = "tp"
            if position.direction == "short":
                if position.sl is not None and candle["high"] >= position.sl:
                    exit_price = position.sl
                    exit_reason = "sl"
                elif position.tp is not None and candle["low"] <= position.tp:
                    exit_price = position.tp
                    exit_reason = "tp"

            if exit_price is None:
                continue

            position.status = "closed"
            position.exit_ts = _to_ms(candle_open)
            position.exit_price = float(exit_price)
            position.exit_reason = exit_reason

            pnl_abs, pnl_pct = _pnl(position.entry_price, position.exit_price, position.direction)

            _update_trade_record(
                trades,
                trade_id,
                exit_ts=position.exit_ts,
                exit_price=position.exit_price,
                exit_reason=exit_reason,
                pnl_abs=pnl_abs,
                pnl_pct=pnl_pct,
            )

            del state.open_positions[trade_id]
            if settings.GATE_ENABLE:
                position_store.clear_open_position(
                    _scope_key(settings, "backtest", position.symbol, position.direction)
                )

    trades_path = os.path.join(output_dir, "trades.jsonl")
    with open(trades_path, "w", encoding="utf-8") as fh:
        for record in trades:
            fh.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")

    summary = summarize_trades(trades, mode=mode)
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    return BacktestResult(symbol=symbol, mode=mode, trades=trades, summary=summary)
