"""Lightweight backtesting CLI for locally stored CSV data."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    pnl_abs: float
    pnl_pct: float


START_EQUITY = 10_000.0


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI indicator."""

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill")


def load_price_data(data_dir: str | Path, symbol: str, timeframe: str) -> pd.DataFrame:
    """Load OHLCV data for a symbol/timeframe from CSV."""

    csv_path = Path(data_dir) / symbol / f"{symbol}_{timeframe}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError("CSV missing required timestamp column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def run_backtest(df: pd.DataFrame, config: Dict[str, str]) -> Dict[str, object]:
    """Execute a simple RSI-based backtest.

    Returns a dict containing trades, equity curve, and metrics.
    """

    rsi = compute_rsi(df["close"], period=14)
    position = None
    entry_price = 0.0
    equity = START_EQUITY
    equity_curve: List[Tuple[pd.Timestamp, float]] = []
    trades: List[Trade] = []

    for idx, row in df.iterrows():
        ts = row["timestamp"]
        price = float(row["close"])
        signal = rsi.iloc[idx]

        equity_curve.append((ts, equity))

        if position is None:
            if signal < 30:
                position = "long"
                entry_price = price
                entry_time = ts
            elif signal > 70:
                position = "short"
                entry_price = price
                entry_time = ts
        elif position == "long":
            if signal > 70:
                pnl_abs = price - entry_price
                pnl_pct = pnl_abs / entry_price * 100
                equity += pnl_abs
                trades.append(
                    Trade(entry_time, ts, position, entry_price, price, pnl_abs, pnl_pct)
                )
                position = None
        elif position == "short":
            if signal < 30:
                pnl_abs = entry_price - price
                pnl_pct = pnl_abs / entry_price * 100
                equity += pnl_abs
                trades.append(
                    Trade(entry_time, ts, position, entry_price, price, pnl_abs, pnl_pct)
                )
                position = None

    # Close any open position at final price
    if position is not None:
        final_price = float(df.iloc[-1]["close"])
        final_ts = df.iloc[-1]["timestamp"]
        if position == "long":
            pnl_abs = final_price - entry_price
        else:
            pnl_abs = entry_price - final_price
        pnl_pct = pnl_abs / entry_price * 100
        equity += pnl_abs
        trades.append(Trade(entry_time, final_ts, position, entry_price, final_price, pnl_abs, pnl_pct))

    equity_curve.append((df.iloc[-1]["timestamp"], equity))

    trades_df = (
        pd.DataFrame([t.__dict__ for t in trades])
        if trades
        else pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "direction",
                "entry_price",
                "exit_price",
                "pnl_abs",
                "pnl_pct",
            ]
        )
    )
    metrics = compute_metrics(trades_df, equity_curve)
    return {"trades": trades_df, "equity_curve": equity_curve, "metrics": metrics}


def compute_metrics(trades: pd.DataFrame, equity_curve: List[Tuple[pd.Timestamp, float]]) -> Dict[str, float]:
    """Calculate summary backtest metrics."""

    total_trades = len(trades)
    wins = trades[trades["pnl_abs"] > 0] if total_trades else pd.DataFrame()
    losses = trades[trades["pnl_abs"] <= 0] if total_trades else pd.DataFrame()
    win_rate = float(len(wins) / total_trades * 100) if total_trades else 0.0
    avg_pnl_pct = float(trades["pnl_pct"].mean()) if total_trades else 0.0
    gross_profit = wins["pnl_abs"].sum() if total_trades else 0.0
    gross_loss = abs(losses["pnl_abs"].sum()) if total_trades else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss else 0.0
    expectancy = float(trades["pnl_abs"].mean()) if total_trades else 0.0

    equity_values = [val for _, val in equity_curve]
    max_equity = START_EQUITY
    max_drawdown = 0.0
    for value in equity_values:
        max_equity = max(max_equity, value)
        drawdown = (max_equity - value) / max_equity * 100 if max_equity else 0.0
        max_drawdown = max(max_drawdown, drawdown)

    end_equity = equity_values[-1] if equity_values else START_EQUITY

    metrics = {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_pnl_pct": avg_pnl_pct,
        "max_drawdown_pct": max_drawdown,
        "profit_factor": profit_factor,
        "expectancy_per_trade": expectancy,
        "start_equity": START_EQUITY,
        "end_equity": end_equity,
    }
    return metrics


def save_results(results: Dict[str, object], out_dir: Path) -> None:
    """Persist trades, metrics, and equity curve plot."""

    out_dir.mkdir(parents=True, exist_ok=True)

    trades: pd.DataFrame = results["trades"]
    trades_path = out_dir / "trades.csv"
    trades.to_csv(trades_path, index=False)

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results["metrics"], f, indent=2)

    equity_curve: List[Tuple[pd.Timestamp, float]] = results["equity_curve"]
    if equity_curve:
        times, equity_values = zip(*equity_curve)
        plt.figure(figsize=(8, 4))
        plt.plot(times, equity_values, label="Equity")
        plt.xlabel("Time")
        plt.ylabel("Equity")
        plt.title("Equity Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "equity_curve.png")
        plt.close()


def parse_args(args: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple RSI backtest")
    parser.add_argument("--symbol", required=True, help="Trading symbol")
    parser.add_argument("--timeframe", required=True, help="Timeframe of CSV data")
    parser.add_argument("--strategy", default="simple_rsi", help="Strategy name")
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing downloaded data")
    parser.add_argument("--out-dir", default="backtests", help="Directory to store backtest results")
    return parser.parse_args(args)


def main(argv: List[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

    try:
        df = load_price_data(args.data_dir, args.symbol, args.timeframe)
        results = run_backtest(df, {"symbol": args.symbol, "strategy": args.strategy})
        run_dir = Path(args.out_dir) / f"{args.symbol}_{args.strategy}_{args.timeframe}"
        save_results(results, run_dir)
        logger.info("Backtest completed: %s", run_dir)
        return 0
    except Exception as exc:  # pragma: no cover - CLI guard
        logger.error("Backtest failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
