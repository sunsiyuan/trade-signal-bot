import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI using smoothed gains/losses (matches most exchange UIs)."""

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing via EMA with alpha=1/period; min_periods ensures the
    # first valid RSI aligns with the configured window length.
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series


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
