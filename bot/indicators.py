import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)

    roll_up = pd.Series(up, index=series.index).rolling(window=period).mean()
    roll_down = pd.Series(down, index=series.index).rolling(window=period).mean()

    rs = roll_up / roll_down
    rsi_series = 100.0 - 100.0 / (1.0 + rs)
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
