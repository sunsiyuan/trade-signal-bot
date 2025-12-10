import math
from typing import Iterable, List

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _compute_rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss if avg_loss else 0.0
    rsi_value = 100.0 - (100.0 / (1.0 + rs))
    return max(0.0, min(100.0, rsi_value))


def _validate_period(period: int) -> int:
    if period <= 0:
        raise ValueError("RSI period must be positive")
    return period


def compute_rsi(closes: Iterable[float], period: int) -> pd.Series:
    """Compute Wilder RSI for a close price iterable.

    The implementation follows the classic Wilder smoothing:
    - The first RSI value is produced after ``period`` closes (period + 1 data
      points) using the simple average of the initial gains/losses.
    - Subsequent values apply Wilder's smoothing on each new change.
    """

    period = _validate_period(period)
    closes_list: List[float] = list(closes)
    n = len(closes_list)

    if n == 0:
        return pd.Series(dtype=float)

    rsi_values: List[float] = [math.nan] * n
    if n < period + 1:
        return pd.Series(rsi_values, dtype=float)

    # Price changes between consecutive closes
    changes = [closes_list[i] - closes_list[i - 1] for i in range(1, n)]

    gains = [max(change, 0.0) for change in changes[:period]]
    losses = [max(-change, 0.0) for change in changes[:period]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    rsi_values[period] = _compute_rsi_value(avg_gain, avg_loss)

    for i in range(period, len(changes)):
        change = changes[i]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        rsi_values[i + 1] = _compute_rsi_value(avg_gain, avg_loss)

    return pd.Series(rsi_values, dtype=float)


def rsi(series: pd.Series, period: int) -> pd.Series:
    """Compatibility wrapper to compute RSI from a pandas Series."""

    return compute_rsi(series.tolist(), period).set_axis(series.index)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    返回三个 Series：macd_line（DIF）, signal_line（DEA）, hist（柱子）
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_series = tr.rolling(window=period).mean()
    return atr_series


def detect_trend(close: pd.Series, ma7: pd.Series, ma25: pd.Series, ma99: pd.Series) -> str:
    """
    粗略趋势标记：
    - ma7 < ma25 < ma99 且都向下 → down
    - ma7 > ma25 > ma99 且都向上 → up
    - 否则 range
    """
    if len(close) < 3:
        return "range"

    ma7_0, ma25_0, ma99_0 = ma7.iloc[-1], ma25.iloc[-1], ma99.iloc[-1]
    ma7_1, ma25_1, ma99_1 = ma7.iloc[-2], ma25.iloc[-2], ma99.iloc[-2]

    # 下跌
    if ma7_0 < ma25_0 < ma99_0 and ma7_0 < ma7_1 and ma25_0 < ma25_1 and ma99_0 < ma99_1:
        return "down"

    # 上涨
    if ma7_0 > ma25_0 > ma99_0 and ma7_0 > ma7_1 and ma25_0 > ma25_1 and ma99_0 > ma99_1:
        return "up"

    return "range"
