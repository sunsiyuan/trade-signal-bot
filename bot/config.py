from dataclasses import dataclass


@dataclass
class Settings:
    # 交易标的
    symbol: str = "HYPE/USDC:USDC"

    # 时间周期
    tf_4h: str = "4h"
    tf_1h: str = "1h"
    tf_15m: str = "15m"

    candles_4h: int = 200
    candles_1h: int = 400
    candles_15m: int = 400

    # EMA / ATR
    ema_period: int = 21
    atr_period: int = 14

    # RSI 多周期
    rsi_fast: int = 6
    rsi_mid: int = 12
    rsi_slow: int = 24

    # MACD 参数 (12, 26, 9)
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # 判定阈值（你后面可以调）
    rsi_short_trigger: float = 60.0   # 做空触发：15m RSI-6/12/24 共振 >= 该区间
    rsi_long_trigger: float = 30.0    # 做多触发：15m RSI-6/12/24 共振 <= 该值

    # ATR 倍数用于止损/止盈
    atr_sl_mult: float = 1.5
    atr_tp1_mult: float = 1.0
    atr_tp2_mult: float = 2.0

    # 仓位建议
    core_position_pct: float = 0.3    # 核心仓 30%
    add_position_pct: float = 0.2     # 加仓 20%
