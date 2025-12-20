# =========================
# regime_detector.py
# =========================
# 目标：根据 MA 斜率 + ATR 相对波动 + RSI 行为，判定市场 regime（行情状态）
#
# 输出 RegimeSignal：
# - regime: trending / high_vol_ranging / low_vol_ranging / unknown
# - reason: 一段可读字符串，方便你在 Telegram/日志里看“为什么这么判”
# - ma_angle: MA25 的相对斜率（代理趋势强弱）
# - atr_rel: ATR / close（代理波动强弱）
# - rsi_avg_dev: RSI 距离 50 的平均偏离（代理是否围绕中线震荡）
# - osc_count: RSI 穿越 50 的次数（代理来回摆动的“振荡频率”）
# - degraded/missing_fields: 数据缺失的降级标记（避免误判）
# =========================

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from .models import MarketSnapshot

# Regime 类型：强约束字符串集合（Literal）
Regime = Literal["trending", "high_vol_ranging", "low_vol_ranging", "unknown"]


# =========================
# RegimeSignal：detect_regime 的返回对象
# =========================
@dataclass
class RegimeSignal:
    # 最终判定的行情模式
    regime: Regime

    # 一段解释字符串（建议输出到日志/telegram）
    reason: str

    # 指标解释层：
    # MA 斜率（相对值），用于识别趋势
    ma_angle: float

    # ATR 相对波动（ATR/close），用于识别高/低波动区间
    atr_rel: float

    # RSI 围绕 50 的“平均偏离”，偏离越小越像震荡
    rsi_avg_dev: float

    # RSI 穿越 50 的次数：越多说明来回震荡越频繁
    osc_count: int

    # degraded 表示“数据不足 → 使用了降级逻辑”
    degraded: bool = False

    # 记录缺失了哪些字段，方便你回溯数据链路
    missing_fields: List[str] = field(default_factory=list)


# -------------------------
# 配置读取工具：兼容 settings 是对象或 dict
# -------------------------
def _get_nested(settings, group: str, key: str, default):
    """
    读取 settings[group][key] 或 settings.group.key

    支持：
    - settings 是 Settings 对象：
        settings.regime.main_tf
        settings.regime["main_tf"]
    - settings 是 dict：
        settings["regime"]["main_tf"]

    找不到就用 default
    """
    if hasattr(settings, group):
        container = getattr(settings, group)
        if isinstance(container, dict):
            return container.get(key, default)
        if hasattr(container, key):
            return getattr(container, key)
    if isinstance(settings, dict) and group in settings:
        return settings[group].get(key, default)
    return default


# =========================
# 主函数：detect_regime
# =========================
def detect_regime(snap: MarketSnapshot, settings) -> RegimeSignal:
    """Decide the current market regime using MA slope, ATR and RSI behavior."""

    # ---- 1) 读取判定所需参数（都来自 settings.regime） ----
    # 主周期：在哪个 TF 上做 regime 判定（默认 1h）
    main_tf = _get_nested(settings, "regime", "main_tf", "1h")

    # 趋势判定阈值：MA 相对斜率绝对值 >= 该值 → trending（默认 0.0008）
    trend_ma_angle_min = _get_nested(settings, "regime", "trend_ma_angle_min", 0.0008)

    # 弱趋势降级阈值：MA 斜率过小或振荡次数过少，认为趋势不足（默认 0.001 / 1）
    min_trend_ma_angle = _get_nested(settings, "regime", "min_trend_ma_angle", 0.001)
    max_trend_osc = _get_nested(settings, "regime", "max_trend_osc", 1)
    # weak_trend_osc_ge：震荡次数方向性开关（A/B 验证用，默认保持旧逻辑）
    weak_trend_osc_ge = _get_nested(settings, "regime", "weak_trend_osc_ge", False)

    # ATR 相对波动阈值：>= 0.015 认为“高波动震荡”
    high_vol_atr_rel = _get_nested(settings, "regime", "high_vol_atr_rel", 0.015)

    # ATR 相对波动阈值：<= 0.006 认为“低波动震荡”
    low_vol_atr_rel = _get_nested(settings, "regime", "low_vol_atr_rel", 0.006)

    # ranging_rsi_band：RSI 平均偏离 50 <= 该值，更像震荡（默认 12）
    rsi_band = _get_nested(settings, "regime", "ranging_rsi_band", 12)

    # slope_lookback：计算 MA 斜率、RSI 偏离与振荡的回看长度（默认 5）
    slope_lookback = _get_nested(settings, "regime", "slope_lookback", 5)

    # ---- 2) 取 main_tf 的 timeframe 指标对象 ----
    tf = snap.get_timeframe(main_tf)

    # missing/degraded：用于记录数据不足时的降级逻辑触发
    missing: List[str] = []
    degraded = False

    # MA25 历史与 RSI6 历史（用来计算斜率、偏离、振荡）
    ma_history = getattr(tf, "ma25_history", None) or []
    rsi_history = getattr(tf, "rsi6_history", None) or []

    # =========================
    # 3) 计算 ma_angle（MA 相对斜率）
    # =========================
    # 原理：用 MA25 历史数据，取一个 base（lookback 位置）与最新值做相对变化
    # ma_angle = (ma_now - ma_base) / |ma_base|
    #
    # 如果 ma_history 太短（<2），则 degraded=1，并让 ma_angle=0（等价“无趋势”）
    if len(ma_history) < 2:
        degraded = True
        missing.append("ma25_history")
        ma_angle = 0.0
    else:
        # 如果历史足够长，用 -slope_lookback 作为基准点，更像“近期斜率”
        if len(ma_history) > slope_lookback:
            base = ma_history[-slope_lookback]
            ma_angle = (ma_history[-1] - base) / max(abs(base), 1e-9)
        else:
            # 不够长就用第一条作为基准（更粗糙）
            base = ma_history[0]
            ma_angle = (ma_history[-1] - base) / max(abs(base), 1e-9)

    # =========================
    # 4) 计算 atr_rel（ATR 相对价格）
    # =========================
    # atr_rel = ATR / close，close 越小越容易放大，所以用 max(close,1e-6) 防止除零
    atr_rel = tf.atr / max(tf.close, 1e-6)

    # =========================
    # 5) 计算 rsi_avg_dev（RSI 平均偏离 50）
    # =========================
    # 原理：如果 RSI 大多围绕 50 摆动，|RSI-50| 的平均值会比较小（更像震荡）
    if rsi_history:
        deviations = [abs(v - 50) for v in rsi_history[-slope_lookback:]]
        rsi_avg_dev = sum(deviations) / max(len(deviations), 1)
    else:
        # RSI 历史缺失：降级逻辑
        degraded = True
        if "rsi6_history" not in missing:
            missing.append("rsi6_history")
        # 兜底：用当前 rsi6 与 50 的偏离当作平均偏离（非常粗糙）
        rsi_avg_dev = abs(tf.rsi6 - 50)

    # =========================
    # 6) 计算 osc_count（RSI 穿越 50 的次数）
    # =========================
    # 原理：用 rsi_history 统计相邻两点是否跨过 50（符号相乘 < 0）
    # 这衡量的是“震荡频率”，穿越越多越像区间震荡
    if len(rsi_history) < 3:
        # 历史太短：降级，认为 osc_count=0（等价“没怎么来回摆动”）
        degraded = True
        if "rsi6_history" not in missing:
            missing.append("rsi6_history")
        osc_count = 0
    else:
        osc_count = sum(
            1
            for i in range(1, len(rsi_history))
            if (rsi_history[i - 1] - 50) * (rsi_history[i] - 50) < 0
        )

    # =========================
    # 7) 第一阶段：用 MA 斜率判 trending，否则按 atr_rel 判 ranging（高/低）
    # =========================
    if abs(ma_angle) >= trend_ma_angle_min:
        # MA 斜率足够大 → trending
        regime: Regime = "trending"
    else:
        # MA 斜率不够 → 认为是 ranging，再用 atr_rel 区分高波动/低波动
        if atr_rel >= high_vol_atr_rel:
            regime = "high_vol_ranging"
        elif atr_rel <= low_vol_atr_rel:
            regime = "low_vol_ranging"
        else:
            # 中间地带：保持中性，判为 low_vol_ranging
            regime = "low_vol_ranging"

    # =========================
    # 8) 第二阶段：用 RSI 行为“修正” regime（非常关键）
    # =========================

    # 规则 A：
    # 如果 RSI 平均偏离很小（<= rsi_band）且穿越次数很多（>= max(2, lookback/2)）
    # → 强烈像震荡
    # 如果此时 regime 被判成 trending，则强制改回 high_vol_ranging
    if rsi_avg_dev <= rsi_band and osc_count >= max(2, slope_lookback // 2):
        if regime == "trending":
            regime = "high_vol_ranging"

    # 规则 B：
    # 如果 RSI 平均偏离很大（>= 1.5*rsi_band）且穿越次数很少（<=1）
    # → 强烈像单边（RSI 远离 50 且很少回穿）
    # 如果此时 regime 不是 trending，则强制改成 trending
    if rsi_avg_dev >= rsi_band * 1.5 and osc_count <= 1 and regime != "trending":
        regime = "trending"

    # =========================
    # 8.5) 弱趋势降级：trending 但趋势强度不足 → high_vol_ranging
    # =========================
    # osc_count 方向性可切换：默认 <=，开启时用 >=（便于 A/B 验证）
    if weak_trend_osc_ge:
        osc_is_weak = osc_count >= max_trend_osc
    else:
        osc_is_weak = osc_count <= max_trend_osc
    weak_trend = (abs(ma_angle) < min_trend_ma_angle) or osc_is_weak

    if regime == "trending" and weak_trend:
        regime = "high_vol_ranging"
        weak_trend_overridden = True
    else:
        weak_trend_overridden = False

    # =========================
    # 9) 生成 reason 字符串（用于输出/调试）
    # =========================
    # degraded 时会额外带 missing 字段，方便你定位“为什么 regime 不稳定”
    dq = "" if not degraded else f" degraded=1 missing={','.join(missing)}"
    reason = (
        f"tf={main_tf} ma_angle={ma_angle:.4f} atr_rel={atr_rel:.4f} "
        f"rsi_avg_dev={rsi_avg_dev:.2f} osc_count={osc_count}{dq}"
    )

    if weak_trend_overridden:
        reason += " | weak_trend_override=1"
    # 记录 A/B 开关状态，便于回放分析
    reason += f" | weak_trend_osc_ge={1 if weak_trend_osc_ge else 0}"

    # =========================
    # 10) 返回 RegimeSignal
    # =========================
    return RegimeSignal(
        regime=regime,
        reason=reason,
        ma_angle=ma_angle,
        atr_rel=atr_rel,
        rsi_avg_dev=rsi_avg_dev,
        osc_count=osc_count,
        degraded=degraded,
        missing_fields=missing,
    )
