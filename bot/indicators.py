import math
from typing import Iterable, List

import pandas as pd


# ============================================================
# EMA
# ============================================================
def ema(series: pd.Series, period: int) -> pd.Series:
    """
    工程约定：
    - 使用 pandas ewm(span=period, adjust=False)
    - 这是“交易系统里最常见、也是最稳定”的 EMA 实现
    - adjust=False = 使用递推公式，而不是全量回算
      → 好处：数值更贴近实盘、性能更好
    """
    return series.ewm(span=period, adjust=False).mean()


# ============================================================
# RSI（Wilder 版本）
# ============================================================

def _compute_rsi_value(avg_gain: float, avg_loss: float) -> float:
    """
    单点 RSI 的计算逻辑（已完成平滑后的 avg_gain / avg_loss）。

    工程注意点：
    - avg_loss == 0 时直接返回 100（极端单边上涨）
    - 最终 clamp 到 [0, 100]，避免数值漂移
    """
    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss if avg_loss else 0.0
    rsi_value = 100.0 - (100.0 / (1.0 + rs))
    return max(0.0, min(100.0, rsi_value))


def _validate_period(period: int) -> int:
    """
    RSI 周期的基本防御：
    - <=0 直接抛异常（宁可 crash，也不要 silently 出错）
    """
    if period <= 0:
        raise ValueError("RSI period must be positive")
    return period


def compute_rsi(closes: Iterable[float], period: int) -> pd.Series:
    """
    核心 RSI 实现（Wilder smoothing）。

    这里有几个**非常关键的工程假设**，以后你调策略一定要记得：

    1️⃣ RSI 不是 rolling window 算法
       - 第一根 RSI 在 period+1 根 close 之后才出现
       - 前 period 根全部是 NaN（这是“正确行为”）

    2️⃣ 初始值用 simple average（不是 EMA）
       - 这是 Wilder 原始定义
       - 很多库（TA-Lib / TradingView）也是这个逻辑

    3️⃣ 后续用递推平滑（Wilder smoothing）
       - avg_gain = (prev_avg_gain*(period-1) + gain) / period
       - avg_loss 同理

    工程层面含义：
    - RSI 在“刚启动 / 数据很短”时**不可靠**
    - 所以你在 data_client 里才会有 lookback_window 的概念
    """

    period = _validate_period(period)
    closes_list: List[float] = list(closes)
    n = len(closes_list)

    if n == 0:
        return pd.Series(dtype=float)

    # 默认全部 NaN，保证 index 对齐
    rsi_values: List[float] = [math.nan] * n

    # 数据不够时直接返回 NaN 序列（不上下游假信号）
    if n < period + 1:
        return pd.Series(rsi_values, dtype=float)

    # 价格变化（diff）
    changes = [closes_list[i] - closes_list[i - 1] for i in range(1, n)]

    # 初始 period 根的 gain / loss
    gains = [max(change, 0.0) for change in changes[:period]]
    losses = [max(-change, 0.0) for change in changes[:period]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # 第一根 RSI（index = period）
    rsi_values[period] = _compute_rsi_value(avg_gain, avg_loss)

    # 递推更新
    for i in range(period, len(changes)):
        change = changes[i]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        rsi_values[i + 1] = _compute_rsi_value(avg_gain, avg_loss)

    return pd.Series(rsi_values, dtype=float)


def rsi(series: pd.Series, period: int) -> pd.Series:
    """
    兼容接口：
    - 允许直接传 pandas Series
    - 内部仍然走 compute_rsi（保证逻辑一致）
    """
    return compute_rsi(series.tolist(), period).set_axis(series.index)


# ============================================================
# MACD
# ============================================================

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    返回：
    - macd_line（DIF）
    - signal_line（DEA）
    - hist（柱子）

    工程约定：
    - DIF = EMA(fast) - EMA(slow)
    - DEA = EMA(DIF, signal)
    - hist = DIF - DEA

    注意：
    - 这里不做任何 threshold / 方向判断
    - 纯粹输出“连续值特征”
    - 策略层（而不是指标层）负责解释 hist > 0 / < 0 的含义
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# ============================================================
# ATR
# ============================================================

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR 实现说明（工程视角）：

    - 使用经典 True Range 定义：
      TR = max(
        high - low,
        |high - prev_close|,
        |low - prev_close|
      )

    - ATR = TR 的 simple moving average（rolling mean）

    ⚠️ 注意：
    - 这里不是 Wilder ATR（EMA 版本），而是 SMA 版本
    - 对你现在的用途（波动尺度 / 相对阈值）来说：
      ✔ SMA 更平滑
      ✔ 不追求极致贴近 TradingView
    """
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


# ============================================================
# Trend detection（非常“刻意简化”的趋势判定）
# ============================================================

def detect_trend(close: pd.Series, ma7: pd.Series, ma25: pd.Series, ma99: pd.Series) -> str:
    """
    趋势标签：up / down / range

    ⚠️ 这是一个“极简 + 强约束”的趋势定义，而不是技术分析百科：

    判定条件（必须同时满足）：
    - 均线排列：ma7 > ma25 > ma99（或反之）
    - 均线方向一致：最近一根都在同向变化

    设计取舍：
    ✅ 优点：
       - 非常干净，趋势一旦成立，噪音很少
       - 适合作为 regime 的“主过滤器”

    ❌ 代价：
       - 很多你肉眼觉得“像趋势”的行情，这里会被判成 range
       - 这正是你之前体感 “趋势很明显但 no action” 的重要来源之一

    结论：
    - 这是一个**策略哲学选择**，不是 bug
    - 如果要松绑，应该在 regime 层或策略路由层，而不是在这里偷偷改
    """
    if len(close) < 3:
        return "range"

    ma7_0, ma25_0, ma99_0 = ma7.iloc[-1], ma25.iloc[-1], ma99.iloc[-1]
    ma7_1, ma25_1, ma99_1 = ma7.iloc[-2], ma25.iloc[-2], ma99.iloc[-2]

    # 下跌趋势：排列 + 同向向下
    if ma7_0 < ma25_0 < ma99_0 and ma7_0 < ma7_1 and ma25_0 < ma25_1 and ma99_0 < ma99_1:
        return "down"

    # 上涨趋势：排列 + 同向向上
    if ma7_0 > ma25_0 > ma99_0 and ma7_0 > ma7_1 and ma25_0 > ma25_1 and ma99_0 > ma99_1:
        return "up"

    return "range"
