import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv


# Load environment variables from a local .env file if present so that
# configuration can be managed without exporting variables manually.
load_dotenv()


@dataclass
class Settings:
    # 交易标的
    symbol: str = "HYPE/USDC:USDC"

    # 趋势跟随策略基础阈值
    min_confidence: float = 0.3

    # 要同时跟踪的合约列表（默认为 HYPE、BTC、ETH、SOL、BNB、AAVE、ZEC）
    tracked_symbols: List[str] = field(
        default_factory=lambda: [
            "HYPE/USDC:USDC",
            "BTC/USDC:USDC",
            "ETH/USDC:USDC",
            "SOL/USDC:USDC",
            "BNB/USDC:USDC",
            "AAVE/USDC:USDC",
            "ZEC/USDC:USDC",
        ]
    )

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
    signal_confidence_threshold: float = 0.8  # 执行模式/方糖提醒的信心阈值

    # ATR 倍数用于止损/止盈
    atr_sl_mult: float = 1.5
    atr_tp1_mult: float = 1.0
    atr_tp2_mult: float = 2.0

    # 仓位建议
    core_position_pct: float = 0.3    # 核心仓 30%
    add_position_pct: float = 0.2     # 加仓 20%

    # 通知配置（通过环境变量注入）
    telegram_token: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN")
    )
    telegram_chat_id: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID")
    )
    telegram_action_token: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_ACTION_BOT_TOKEN")
    )
    telegram_action_chat_id: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_ACTION_CHAT_ID")
    )
    telegram_summary_token: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_SUMMARY_BOT_TOKEN")
        or os.getenv("TELEGRAM_BOT_TOKEN")
    )
    telegram_summary_chat_id: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_SUMMARY_CHAT_ID")
        or os.getenv("TELEGRAM_CHAT_ID")
    )
    ftqq_key: Optional[str] = field(default_factory=lambda: os.getenv("FT_SENDKEY"))
    webhook_url: Optional[str] = field(
        default_factory=lambda: os.getenv("WEBHOOK_URL")
    )

    # Debug flags
    debug_rsi: bool = field(
        default_factory=lambda: os.getenv("DEBUG_RSI", "").lower() in {"1", "true", "yes"}
    )

    # Regime detection
    regime: dict = field(
        default_factory=lambda: {
            "main_tf": "1h",
            "trend_ma_angle_min": 0.0008,
            "min_trend_ma_angle": 0.001,
            "high_vol_atr_rel": 0.015,
            "low_vol_atr_rel": 0.006,
            "ranging_rsi_band": 12,
            "slope_lookback": 5,
            "max_trend_osc": 1,
        }
    )

    # Mean reversion strategy
    mean_reversion: dict = field(
        default_factory=lambda: {
            "tf": "1h",
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "atr_dev_mult": 0.55,
            "min_oi_change_pct": 1.5,
            "tp_to_sl_ratio": 1.5,
            "core_position_pct": 0.5,
            "add_position_pct": 0.25,
            "sl_buffer_mult": 0.8,
            "require_oi": True,
            "allow_oi_missing_fallback": True,
            "fallback_confidence_mult": 0.75,
            "fallback_core_position_mult": 0.5,
            "fallback_add_position_mult": 0.0,
        }
    )

    # Liquidity hunt strategy
    liquidity_hunt: dict = field(
        default_factory=lambda: {
            "tf": "1h",
            "swing_lookback": 24,
            "swing_lookback_bars": 50,
            "swing_exclude_last": 1,
            "price_proximity_pct": 0.006,
            "min_wall_mult": 3.0,
            "min_oi_spike_pct": 5.0,
            "small_body_atr_mult": 0.35,
            "small_body_lookback": 8,
            "wall_depth_bps": 30,
            "wall_size_threshold": 3.0,
            "post_spike_candle_count": 2,
            "sl_buffer_pct": 0.002,
            "core_position_pct": 0.5,
            "add_position_pct": 0.25,
            "require_oi": True,
            "allow_oi_missing_fallback": False,
            "fallback_confidence": 0.65,
            "fallback_core_position_mult": 0.5,
            "fallback_add_position_mult": 0.5,
            "allow_fallback_when_missing": True,
            "missing_fallback_confidence": 0.35,
            "missing_fallback_core_mult": 0.5,
            "missing_fallback_add_mult": 0.0,
        }
    )

    # Trend following strategy
    trend_following: dict = field(
        default_factory=lambda: {
            "high_conf_bonus": 0.15,
            "low_conf_penalty": 0.10,
            "high_conf_core_mult": 1.0,
            "low_conf_core_mult": 0.5,
            "high_conf_add_mult": 1.0,
            "low_conf_add_mult": 0.0,
            "rsi_extreme_long": 25,
            "rsi_extreme_short": 75,
            "require_liquidity_prefix_for_high_conf": True,
        }
    )

    # Notification thresholds
    notification: dict = field(
        default_factory=lambda: {
            "execute_trade_conf": 0.70,
            "watch_trade_conf": 0.60,
            "watch_edge_conf": 0.80,
            "near_miss_delta": 0.05,
        }
    )

    # Price quantization for signal_id stability
    price_quantization: dict = field(
        default_factory=lambda: {
            "BTC": 100,
            "HYPE": 0.01,
            "SOL": 0.1,
            "ETH": 10,
            "AAVE": 0.1,
            "ZEC": 1,
        }
    )
