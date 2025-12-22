from __future__ import annotations

import pytest

from bot.backtest.outcome import summarize_trades
from bot.backtest.types import TradeRecord


def _trade(trade_id: str, pnl_abs: float) -> TradeRecord:
    return TradeRecord(
        trade_id=trade_id,
        symbol="SOL/USDC:USDC",
        signal_id=f"sig-{trade_id}",
        plan_type="EXECUTE_NOW",
        direction="long",
        setup_type="mr",
        order_created_ts=1,
        filled_ts=2,
        filled_price=100.0,
        expired=False,
        exit_ts=3,
        exit_price=101.0,
        exit_reason="tp",
        pnl_abs=pnl_abs,
        pnl_pct=pnl_abs / 100.0,
        decision_trace=None,
        data_coverage={"has_oi": True, "has_orderbook": True},
    )


def test_backtest_gate_rates_are_consistent():
    trades = [_trade("t1", 4.0), _trade("t2", -2.0)]
    gate_stats = {
        "signal_count": 100,
        "decision_count": 100,
        "execute_decision_count": 10,
        "order_created_count": 2,
        "filled_count": 2,
        "closed_trade_count": 2,
        "skipped_count": 8,
        "skipped_by_reason": {"SKIP_DUP": 5, "SKIP_COOLDOWN": 3},
        "forced_close_count": 0,
        "cooldown_blocked_count": 3,
        "dedup_blocked_count": 5,
        "in_position_blocked_count": 0,
    }

    summary = summarize_trades(trades, mode="execute_now_only", gate_stats=gate_stats)
    metrics = summary["metrics"]
    rates = summary["gate_stats"]["rates"]

    assert metrics["trade_count"] == 2
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["fill_rate"] == pytest.approx(1.0)
    assert rates["signal_fill_rate"] == pytest.approx(0.02)
    assert rates["execute_to_order_rate"] == pytest.approx(0.2)
    assert rates["order_fill_rate"] == pytest.approx(1.0)
    assert rates["signal_to_execute_rate"] == pytest.approx(0.1)
