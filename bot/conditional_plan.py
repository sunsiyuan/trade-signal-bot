"""Conditional plan compiler for execution intents (unified 4H TTL).

这个模块的定位：
- 输入：ExecutionIntent（“想怎么做”的执行意图，来自 SignalEngine/策略层）
- 输入：MarketSnapshot（当前市场数据快照，用于判断是否“够近/够合理”）
- 输出：ConditionalPlan（“怎么执行”的条件单计划，统一成 WATCH / PLACE_LIMIT_4H / EXECUTE_NOW 三种）

注意：这里“4H TTL”是统一策略：只要进入 PLACE_LIMIT_4H，就会写一个 valid_until_utc（默认 intent.ttl_hours）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from .models import ConditionalPlan, ExecutionIntent, MarketSnapshot


def now_plus_hours(hours: int) -> str:
    """Return UTC ISO string after given hours.

    作用：
    - 给计划单生成过期时间（valid_until_utc）
    - 返回 UTC 的 isoformat 字符串（带时区信息）

    约束/注意：
    - 这里返回的是字符串，不是 datetime；方便序列化进 state.json
    - 下游需要 _parse_dt 才能比较过期（你在 main.py 里做了）
    """
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def resolve_atr_4h(snapshot: MarketSnapshot) -> Optional[float]:
    """
    为策略/计划提供一个“4小时尺度 ATR”的估计值。

    优先级：
    1) snapshot.tf_4h.atr 直接用（最靠谱）
    2) snapshot.tf_1h.atr * 2.0 作为 4h 近似（粗略线性缩放）
    3) snapshot.tf_15m.atr * 4.0 作为 4h 近似（更粗略）
    4) 都没有 -> None

    设计意图：
    - Conditional plan 的“距离判断”（是否足够接近 entry）需要一个尺度量纲（atr）
    - 你在 build_conditional_plan_from_intent 里用 atr 来决定 EXECUTE_NOW / PLACE_LIMIT_4H

    风险点（你现在接受这种简化）：
    - ATR 严格来说不是线性随时间缩放（更常见的缩放是 sqrt(time)），
      但这里你用的是“工程近似”：只为了给“距离阈值”一个粗尺度。
    """
    if snapshot.tf_4h and snapshot.tf_4h.atr:
        return snapshot.tf_4h.atr
    if snapshot.tf_1h and snapshot.tf_1h.atr:
        return snapshot.tf_1h.atr * 2.0
    if snapshot.tf_15m and snapshot.tf_15m.atr:
        return snapshot.tf_15m.atr * 4.0
    return None


def _cancel_rules(valid_until: Optional[str]) -> dict:
    """
    生成 cancel_if 字段（计划单取消条件）。

    字段含义（约定式）：
    - invalidation_crossed: 价格触发失效位则取消（由 main.py reconcile 逻辑负责）
    - regime_changed: 市场状态变了则取消（同样由 main.py reconcile）
    - expired: 如果 valid_until 不为空，则启用“过期取消”

    注意：
    - 这里只是声明 cancel_if 规则，实际执行取消逻辑不在本模块，而在 main.py 的 reconcile loop。
    """
    return {
        "invalidation_crossed": True,
        "regime_changed": True,
        "expired": valid_until is not None,
    }


def build_conditional_plan_from_intent(
    intent: ExecutionIntent, snap: MarketSnapshot
) -> ConditionalPlan:
    """
    核心编译器：把 ExecutionIntent 编译成 ConditionalPlan。

    intent 关键字段（按你当前逻辑使用到的）：
    - direction: "long" / "short" / "none"
    - entry_price: 理想入场价（limit 的挂单价）
    - atr_4h: 4h 尺度波动（用于判断“离 entry 近不近”）
    - ttl_hours: 限价计划有效期（通常 4h）
    - allow_execute_now: 是否允许“立即执行”模式

    snap 用到的字段：
    - snap.price_last（如果 snapshot 上有）
    - 或 snap.tf_15m.close（兜底当前价）

    输出 plan 的三类：
    - WATCH_ONLY：只关注，不创建限价/不立即交易
    - PLACE_LIMIT_4H：创建一个 4小时有效的限价计划
    - EXECUTE_NOW：认为 entry 已经到达/足够近，直接按现价执行
    """

    # --- 0) 方向为 none：直接返回 WATCH_ONLY ---
    # 说明策略层当前没有任何执行意图（无论 edge/trade_conf 怎么样）
    if intent.direction == "none":
        return ConditionalPlan(
            execution_mode="WATCH_ONLY",
            direction="none",
            entry_price=None,
            valid_until_utc=None,
            cancel_if=_cancel_rules(None),
            explain="No execution intent",
        )

    # --- 1) 当前价 current 的选择 ---
    # 优先 snap.price_last（如果你的 snapshot 顶层有）
    # 否则用 15m close 兜底
    # 注意：这跟 main.py 的 _extract_mark_price 逻辑不同：
    # - main.py 更偏向 mark 价
    # - 这里偏向 last/close（对“距离 entry”判断可能会有轻微差异）
    current = getattr(snap, "price_last", None) or getattr(snap.tf_15m, "close", None)

    # ATR 与 entry_price 从 intent 直接取（说明：策略层需要负责给足这些值）
    atr = intent.atr_4h
    entry_price = intent.entry_price

    def _direction_allows_entry(current_price: float, ideal_entry: float) -> bool:
        """
        方向与入场价的“相对位置”校验（防止挂反方向的单）：

        - long：理想入场价应当 <= 当前价
          否则说明 entry 在“当前价上方”（你等于要追涨买入），不符合你“ideal entry=回踩买入”的假设
        - short：理想入场价应当 >= 当前价
          否则说明 entry 在“当前价下方”（你等于要追跌卖出），不符合你“ideal entry=反弹做空”的假设
        - 其他：默认允许

        这一步的设计意图是：保证 entry 的含义始终是“更好的价位”（buy lower / sell higher）。
        """
        if intent.direction == "long":
            return ideal_entry <= current_price
        if intent.direction == "short":
            return ideal_entry >= current_price
        return True

    # --- 2) EXECUTE_NOW 分支：entry 已经到达或非常接近 ---
    # 条件解释：
    # - allow_execute_now: 策略层允许立即执行（有些策略可能不允许）
    # - current/entry/atr 都必须有值
    # - 方向校验通过（entry 在正确的一侧）
    # - abs(current-entry) <= 0.35 * atr：距离小于 0.35 ATR，则认为“已经到位”，可以用现价执行
    #
    # 输出：
    # - execution_mode=EXECUTE_NOW
    # - entry_price 用 current（注意：这里把 plan.entry_price 设成当前价，表示“以当前价成交”）
    if (
        intent.allow_execute_now
        and current is not None
        and entry_price is not None
        and atr
        and _direction_allows_entry(current, entry_price)
        and abs(current - entry_price) <= 0.35 * atr
    ):
        return ConditionalPlan(
            execution_mode="EXECUTE_NOW",
            direction=intent.direction,
            entry_price=current,
            valid_until_utc=None,
            cancel_if=_cancel_rules(None),
            explain="Entry reached, execute now",
        )

    # --- 3) PLACE_LIMIT_4H / WATCH_ONLY 分支（需要 atr + current + entry）---
    # 如果缺数据（atr/current/entry 任一缺失），会跳到最后的 WATCH_ONLY
    if atr and current is not None and entry_price is not None:

        # 3.1) 如果 entry 在“方向不该出现”的一侧：直接 WATCH_ONLY
        # 例：long 的 entry 在 current 上方（追涨），short 的 entry 在 current 下方（追跌）
        if not _direction_allows_entry(current, entry_price):
            return ConditionalPlan(
                execution_mode="WATCH_ONLY",
                direction=intent.direction,
                entry_price=None,
                valid_until_utc=None,
                cancel_if=_cancel_rules(None),
                explain="Entry on wrong side of current price for direction",
            )

        # 3.2) 如果离 entry 不远：允许下 4H 限价单
        # abs(current-entry) <= 1.5*atr：距离阈值更宽（比 execute_now 更宽）
        # 设计含义：
        # - 0.35 ATR：足够近 -> 直接打
        # - 1.5 ATR：还算近 -> 挂单等
        if abs(current - entry_price) <= 1.5 * atr:
            valid_until = now_plus_hours(intent.ttl_hours)
            return ConditionalPlan(
                execution_mode="PLACE_LIMIT_4H",
                direction=intent.direction,
                entry_price=entry_price,
                valid_until_utc=valid_until,
                cancel_if=_cancel_rules(valid_until),
                explain="Place 4h limit order at ideal entry",
            )

    # --- 4) 兜底：太远/缺数据 -> WATCH_ONLY ---
    # 典型情况：
    # - ATR 缺失（indicator 没算出来，或数据缺口）
    # - current 缺失（snapshot 没有 last/close）
    # - entry_price 缺失（策略层没给）
    # - 或者距离 entry > 1.5 ATR（“现在挂单意义不大”，先看着）
    return ConditionalPlan(
        execution_mode="WATCH_ONLY",
        direction=intent.direction,
        entry_price=None,
        valid_until_utc=None,
        cancel_if=_cancel_rules(None),
        explain="Too far from entry, watch only",
    )


__all__ = [
    "build_conditional_plan_from_intent",
    "now_plus_hours",
    "resolve_atr_4h",
]
