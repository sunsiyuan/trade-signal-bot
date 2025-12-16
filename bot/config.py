# config.py
# ============================================================
# 这个文件是 bot 的“核心调参面板”（Control Panel）。
# 你调参时建议按下面的层级去理解，而不是逐行乱调：
#
# 0) Universe（交易标的/多币种列表/时间周期/数据长度）
# 1) Indicators（EMA/ATR/RSI/MACD 的基础参数：决定信号的“观测方式”）
# 2) Execution（止损止盈、仓位、价格量化：决定“怎么下/怎么报/怎么去重”）
# 3) Regime（市场态势分类：trending vs ranging，及高/低波动）
# 4) Strategies（三种策略的阈值：MR / LH / TF；每个策略一组旋钮）
# 5) Notification（通知门槛：何时 WATCH，何时 EXECUTE_NOW / 方糖）
#
# 重要提醒：不要同时大幅改动多个分区。
# 建议一次只改一个“旋钮组”，并用固定回放区间/回测去对比效果。
# ============================================================

import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv


# ------------------------------------------------------------
# 环境变量加载：允许你用本地 .env 管理 token / chat_id 等敏感配置
# ------------------------------------------------------------
# 这行的意义：你不需要在 shell 里 export 一堆变量；项目根目录放一个 .env 即可。
# 注意：如果在 GitHub Actions 上跑，通常不会依赖 .env，而是用 Secrets 注入环境变量。
load_dotenv()


@dataclass
class Settings:
    # ============================================================
    # 0) Universe：交易标的、跟踪列表、时间周期、拉取的历史长度
    # ============================================================

    # 单一标的（在某些模块里仍可能用到；但你 main.py 已经主要走 tracked_symbols）
    # 你可以把它理解为 “默认 symbol / fallback symbol”。
    symbol: str = "HYPE/USDC:USDC"

    # 一个“全局最低信心”门槛（更像 legacy 时代的阈值）。
    # 如果你策略层现在主要用 trade_confidence/edge_confidence 分开管，
    # 那这个值通常只作为兜底过滤（或者某些旧逻辑仍引用它）。
    min_confidence: float = 0.3

    # 多币种监控列表（Universe 的核心开关）
    # 建议你把它当成“固定监控池”，不要频繁增删：
    # - 增加标的 = 增加噪音源 + 增加 API 压力 + 增加你注意力成本
    # - 减少标的 = 提升“高胜率策略”的专注度
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

    # 时间周期（TF Universe）
    # 你当前的体系：15m 用于微观触发/热度，1h 用于主判断，4h 用于大势过滤/确认。
    tf_4h: str = "4h"
    tf_1h: str = "1h"
    tf_15m: str = "15m"

    # 拉取的 K 线数量（决定指标“稳定性/滞后/计算成本”）
    # 原则：
    # - 过短：指标很抖（特别是 ATR/MA 角度）
    # - 过长：API 压力大、回放慢，但信号更平滑
    # 你现在的量级：4h=200（约 33 天），1h=400（约 16 天），15m=400（约 4 天）
    candles_4h: int = 200
    candles_1h: int = 400
    candles_15m: int = 400

    # ============================================================
    # 1) Indicators：基础指标参数（“看世界的镜头”）
    # ============================================================

    # EMA / ATR
    # - ema_period 影响趋势判断的灵敏度（越小越敏感，越容易“频繁趋势切换”）
    # - atr_period 影响波动率估计（越小越敏感，SL/TP 更容易被放大/缩小）
    ema_period: int = 21
    atr_period: int = 14

    # RSI 多周期（快/中/慢）
    # 你用 RSI6/12/24 的好处：能表达“短线超买超卖 + 中线确认 + 慢速过滤”
    # 但代价：共振条件会更苛刻（容易 no action）。
    rsi_fast: int = 6
    rsi_mid: int = 12
    rsi_slow: int = 24

    # MACD 参数 (12, 26, 9) —— 属于经典默认
    # 如果你要调 MACD，一般不建议在这里先动（先动阈值/路由更有效），
    # 因为调 MACD 相当于换了一套“趋势度量”。
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # ============================================================
    # 2) Execution（执行与仓位）：止损止盈、仓位、执行门槛、价格量化
    # ============================================================

    # 这些阈值看起来像“基于 15m RSI 多周期共振”的 legacy 触发条件：
    # - rsi_short_trigger：>= 该值时，认为短线偏热（倾向做空触发）
    # - rsi_long_trigger：<= 该值时，认为短线偏冷（倾向做多触发）
    #
    # 调参思路（如果你还在用这段触发）：
    # - 抬高 short_trigger / 降低 long_trigger：变得更“苛刻”，信号更少但可能更稳
    # - 降低 short_trigger / 抬高 long_trigger：更“宽松”，信号更多但噪音更大
    rsi_short_trigger: float = 60.0
    rsi_long_trigger: float = 30.0

    # “执行模式/方糖提醒”的单一信心阈值（更像旧版：confidence 一个数）。
    # 你现在体系已经分了 trade_confidence / edge_confidence，
    # 所以这个值若仍被使用，建议把它理解为“旧逻辑兜底阈值”。
    signal_confidence_threshold: float = 0.8

    # ATR 倍数用于止损/止盈（SL/TP 的“几倍波动”）
    # - atr_sl_mult 越大：止损更宽，胜率可能上升，但亏损幅度变大、回撤更深
    # - atr_tp1/tp2 越大：止盈更远，盈亏比更好，但触达率更低
    #
    # 注意：不同币 ATR 相对价格的比例差异巨大（HYPE vs BTC），
    # 所以如果你用统一倍数，体验会很“币种相关”。
    atr_sl_mult: float = 1.5
    atr_tp1_mult: float = 1.0
    atr_tp2_mult: float = 2.0

    # 仓位建议（只给“建议仓位/风险输出”使用，不一定自动下单）
    # core_position_pct / add_position_pct = 核心仓 + 加仓
    # 你当前是偏保守（0.3 + 0.2 = 0.5 总仓位建议）
    core_position_pct: float = 0.3
    add_position_pct: float = 0.2

    # ============================================================
    # 3) Notification（通知/渠道）：只通过 env 注入，不要写死在代码里
    # ============================================================

    # 这里采用 Optional[str] + default_factory 读取 env：
    # - 本地跑：.env 生效
    # - Actions 跑：Secrets 注入 env 生效

    # Action bot（用于“可执行动作”频道）
    telegram_action_token: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_ACTION_BOT_TOKEN")
    )
    telegram_action_chat_id: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_ACTION_CHAT_ID")
    )

    # Summary bot（用于“概览”频道）
    # 这里做了 fallback：没配 summary token/chat 就用默认 token/chat。
    telegram_summary_token: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_SUMMARY_BOT_TOKEN")
        or os.getenv("TELEGRAM_BOT_TOKEN")
    )
    telegram_summary_chat_id: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_SUMMARY_CHAT_ID")
        or os.getenv("TELEGRAM_CHAT_ID")
    )

    # 方糖（更强提醒渠道，通常只在 EXECUTE_NOW 用）
    ftqq_key: Optional[str] = field(default_factory=lambda: os.getenv("FT_SENDKEY"))

    # 其他 webhook（如果你有自建服务/飞书机器人等）
    webhook_url: Optional[str] = field(
        default_factory=lambda: os.getenv("WEBHOOK_URL")
    )

    # Debug flags：调参时可打开更多 RSI 调试输出
    debug_rsi: bool = field(
        default_factory=lambda: os.getenv("DEBUG_RSI", "").lower() in {"1", "true", "yes"}
    )

    # ============================================================
    # 4) Regime detection：市场态势分类（策略路由的“总闸门”）
    # ============================================================
    # 你现在的 regime 看起来有几个关键旋钮：
    # - main_tf：用哪个周期作为 regime 的主判断（你设 1h）
    # - trend_ma_angle_min / min_trend_ma_angle：趋势线斜率阈值（越大越苛刻，越容易判成震荡）
    # - high_vol_atr_rel / low_vol_atr_rel：用 ATR 相对价格（atr_rel）来区分高/低波动
    # - ranging_rsi_band：震荡时 RSI 的“允许区间宽度”（越小越容易认定“非震荡”）
    # - slope_lookback：斜率回看窗口（越大越平滑但滞后）
    # - max_trend_osc：趋势态允许的“震荡/摇摆次数”上限（越小越容易拒绝趋势）
    #
    # 调参思路（按优先级）：
    # 1) 先让 regime 分类“看起来像你肉眼看到的市场”
    # 2) 再调各策略的触发阈值
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

    # ============================================================
    # 5) Strategy: Mean Reversion（均值回归 MR）
    # ============================================================
    # 适用场景：ranging（尤其高波动震荡）里抓“超跌反弹/超涨回落”
    #
    # 关键旋钮 = 你手工调参最该盯的：
    # - rsi_oversold / rsi_overbought：超卖/超买阈值（越靠近 50 越频繁，越极端越少但更稳）
    # - atr_dev_mult：价格偏离“均值/中枢”的尺度（越大越苛刻，越小越宽松）
    # - min_oi_change_pct：OI 变化门槛（你现在把 OI 当作“参与度/能量”过滤器）
    # - tp_to_sl_ratio：期望盈亏比（越大越不容易触发 TP）
    #
    # 风险与降级（fallback）：
    # - require_oi=True：没有 OI 就不做（会导致大量 no action）
    # - allow_oi_missing_fallback=True：允许缺 OI 时降级执行，但要降低信心和仓位（见 fallback_*）
    #
    # 仓位（core/add）在策略内又单独配置：说明策略层会覆盖全局仓位建议。
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

    # ============================================================
    # 6) Strategy: Liquidity Hunt（流动性猎杀 LH）
    # ============================================================
    # 适用场景：临近 swing high/low、出现 orderbook wall / OI spike 后的“假突破/扫止损”
    #
    # 这个策略参数多，但你调参时不要逐个乱动，建议按 3 个子模块理解：
    #
    # A) swing 定位（定义“关键高低点在哪里”）
    # - swing_lookback / swing_lookback_bars / swing_exclude_last
    #
    # B) 触发条件（靠近关键位 + 墙体显著 + OI/成交结构配合）
    # - price_proximity_pct：离 swing 的距离（越小越苛刻）
    # - min_wall_mult / wall_depth_bps / wall_size_threshold：墙体强度与深度
    # - min_oi_spike_pct：OI 突变阈值（能量/参与度）
    # - small_body_atr_mult + small_body_lookback：是否出现“停顿/小实体K线”（吸筹/换手的信号）
    # - post_spike_candle_count：OI spike 后允许的确认 candle 数
    #
    # C) 风控与 fallback
    # - sl_buffer_pct：止损缓冲（越大越宽）
    # - require_oi / allow_oi_missing_fallback：OI 缺失如何处理
    #   这块你现在有两套字段（fallback_* 与 missing_fallback_*），
    #   手工调参时要认清“到底代码用哪套”。（建议后续统一命名）
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

    # ============================================================
    # 7) Strategy: Trend Following（趋势跟随 TF）
    # ============================================================
    # 适用场景：regime=trending，追随主方向（但你又要求“高信心时需要流动性前缀”）
    #
    # 这组参数的结构是“对信心的奖惩 -> 影响仓位/信心输出”：
    # - high_conf_bonus / low_conf_penalty：调整信心分（trade_confidence）或最终 confidence 的偏置
    # - high_conf_*_mult / low_conf_*_mult：当信心高/低时，仓位如何缩放
    # - rsi_extreme_long / rsi_extreme_short：极端 RSI 用于过滤追高/追低
    # - require_liquidity_prefix_for_high_conf：
    #   这是一个很关键的“苛刻闸门”：即使趋势很明显，你也可能因为缺少 LH 类证据而不给高信心
    #   （这正可能解释你之前“肉眼趋势很明显但 no action”的体感）
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

    # ============================================================
    # 8) Notification thresholds：把 trade_conf / edge_conf 映射成动作
    # ============================================================
    # 这块决定你系统“什么时候吵你”“什么时候给你可执行动作”。
    # 建议把它当成你产品的“通知产品策略”，不要为了提高胜率就胡乱调太高——
    # 你要的是：WATCH 让你预热注意力，EXECUTE_NOW 才强提醒。
    #
    # - execute_trade_conf：>= 这个值才允许 EXECUTE（强动作）
    # - watch_trade_conf：>= 这个值才允许 WATCH（弱动作）
    # - watch_edge_conf：机会出现概率（edge）>= 这个值才值得 watch
    # - near_miss_delta：接近 execute 门槛也允许 watch（防止“差一点就错过”）
    notification: dict = field(
        default_factory=lambda: {
            "execute_trade_conf": 0.70,
            "watch_trade_conf": 0.60,
            "watch_edge_conf": 0.80,
            "near_miss_delta": 0.05,
        }
    )

    # ============================================================
    # 9) Price quantization：用于 signal_id 稳定性（去重/缓存的地基）
    # ============================================================
    # 你 main.py 里 compute_signal_id / action_hash 的稳定性很依赖“价格离散化”。
    # 直觉：
    # - BTC 用 100：意思是价格变动 100 美元以内，signal_id 仍可能认为“同一档位”
    # - HYPE 用 0.01：更精细，避免小币波动导致过度合并
    #
    # 调参原则：
    # - 量化步长越大 -> 去重更强（更少重复提醒）但也更容易把“不同机会”合并成同一个 signal_id
    # - 量化步长越小 -> 更敏感（更多 signal_id）可能导致提醒更频繁
    #
    # 这块是“通知稳定性”的关键旋钮，通常不要和策略阈值一起改。
    price_quantization: dict = field(
        default_factory=lambda: {
            "BTC": 100,
            "HYPE": 0.01,
            "SOL": 0.1,
            "ETH": 10,
            "BNB": 1,
            "AAVE": 0.1,
            "ZEC": 1,
        }
    )
