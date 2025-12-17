import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional, Tuple

# ============================================================
# State store 的定位（非常重要）
#
# 这个模块做两件事：
# 1) 每个 symbol 维护一个持久化 state 文件：.state/<symbol>.json
#    - 里面记录 “每个 signal_id 对应的 action_type 是否已经发过、什么时候过期”
# 2) 提供 should_send / mark_sent：
#    - should_send：在发通知/执行动作之前判断是否需要 dedupe（避免重复）
#    - mark_sent：实际发送后写回记录（产生 TTL）
#
# 你目前的系统里，把动作粗分成：
# - WATCH：只做关注提醒（这里明确：WATCH 不走缓存 dedupe）
# - LIMIT_4H：4小时有效的限价计划/条件单
# - EXECUTE_NOW：立即执行动作（强提醒）
#
# 这层不关心策略，不关心行情，只关心：
# “同一件事（signal/action）是不是刚发过？”
# ============================================================

STATE_DIR = os.getenv("STATE_DIR", ".state")

# state 文件格式版本号（用于未来升级迁移）
STATE_VERSION = 1

# 各 action_type 的 TTL（秒）：决定 “多久内算重复”
# - WATCH: 1h  （但注意：WATCH 被特殊处理，完全不dedupe）
# - LIMIT_4H: 6h（与 4h 条件单接近，但稍宽）
# - EXECUTE_NOW: 30m（立即执行的强提醒，短TTL，避免刷屏）
ACTION_TTLS = {
    "WATCH": 3600,
    "LIMIT_4H": 21600,
    "EXECUTE_NOW": 1800,
}


# ============================================================
# 文件命名与路径：symbol -> 安全文件名 -> .state/<symbol>.json
# ============================================================
def sanitize_symbol(symbol: str) -> str:
    """
    将 symbol 转换为安全的文件名片段：
    - 只允许 [A-Za-z0-9._-]
    - 其它字符统一替换为 "_"
    - 空值/全被替换后兜底 "unknown"

    目的：避免路径注入、避免 Windows/macOS 文件名不兼容。
    """
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", symbol or "unknown")
    return sanitized or "unknown"


def _state_path(symbol: str, base_dir: str = STATE_DIR) -> str:
    """
    给定 symbol + base_dir，生成对应 state 文件路径：
    - 默认 base_dir = .state
    - 文件名 = sanitize_symbol(symbol) + ".json"
    """
    return os.path.join(base_dir, f"{sanitize_symbol(symbol)}.json")


def _default_state() -> Dict[str, Any]:
    """
    每个 symbol 的 state 文件默认结构：
    - version: 版本号
    - updated_at_utc: 最近写入时间
    - signals: dict，key 为 "<symbol>|<signal_id>"，value 为该 signal 的记录
    """
    return {"version": STATE_VERSION, "updated_at_utc": None, "signals": {}}


# ============================================================
# 时间序列工具：解析与序列化 UTC 时间
# ============================================================
def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    """
    从 isoformat 字符串解析 datetime。
    - 如果字符串无时区，则默认按 UTC 处理
    - 失败返回 None（state 文件损坏/字段异常时不阻断主流程）
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def _isoformat(dt: datetime) -> str:
    """
    将 datetime 规范化成 UTC isoformat 字符串。
    - 如果 dt 无 tzinfo，补 UTC
    - 统一转为 UTC 输出
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


# ============================================================
# Canonical JSON：用于 hash 计算，保证“同内容 -> 同 hash”
# ============================================================
def _canonical_json(payload: Dict[str, Any]) -> str:
    """
    将 dict 序列化成稳定字符串（用于 sha1）：
    - sort_keys=True：key 顺序稳定
    - separators=(",", ":")：紧凑输出，避免空格差异
    - ensure_ascii=False：保留 unicode（中文 reason 不会被转义）
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ============================================================
# signal_id：把 TradeSignal（或类似对象）映射成一个稳定 id
# ============================================================
def compute_signal_id(
    signal: Any, price_quantization: Optional[Dict[str, float]] = None
) -> str:
    """
    生成 signal_id 的策略：

    1) 如果 signal 对象本身已有 signal.signal_id（你现在已经在加这个字段）
       -> 直接用它（最稳定、也是你未来应该坚持的主路径）

    2) 否则用一组“关键字段”拼成 payload，再对 canonical_json 做 sha1。
       目的：同一个策略机会（相同 symbol/regime/entry/sl/tp/...）得到同一个 signal_id。
       - 如果提供 price_quantization（或 signal.settings.price_quantization），
         会对 entry/sl/tp 做价格离散化，减少浮点抖动导致的 id 变化。

    ⚠️ 风险点（你排查“同一个 signal id 重复输出 create”时必须看这里）：
    - payload 里包含 entry/sl/tp_list：如果这些值每次计算有微小浮动（尤其 entry）
      就会导致 signal_id 不稳定 -> 无法 dedupe -> 每次都是新 signal。
    - payload 里 gate_reason 也可能变化（不同 run 的 gate_tag 文案/字段）
      也会导致 signal_id 变化。
    - 解决方向通常是：
      a) 在策略层稳定量化 entry（如按 price_quantization）后再写入 signal
      b) signal.signal_id 由策略层直接给出（避免这里拼装）
    """
    base_signal_id = getattr(signal, "signal_id", None)
    if base_signal_id:
        return str(base_signal_id)

    symbol = getattr(signal, "symbol", None)

    def _get_price_step() -> Decimal:
        base = symbol.split("/")[0] if symbol else ""
        mapping = price_quantization or getattr(
            getattr(signal, "settings", None), "price_quantization", None
        )
        step = (mapping or {}).get(base)
        if step is None:
            return Decimal("0.01")
        return Decimal(str(step))

    def _quantize_price(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        step = _get_price_step()
        quantized = (Decimal(value) / step).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return float(quantized * step)

    payload = {
        "symbol": symbol,
        "regime": getattr(signal, "regime", None)
        or getattr(getattr(signal, "snapshot", None), "regime", None),
        "strategy": getattr(signal, "strategy", None) or getattr(signal, "setup_type", None),
        "entry": _quantize_price(getattr(signal, "entry", None)),
        "sl": _quantize_price(getattr(signal, "sl", None)),
        "tp_list": [
            _quantize_price(getattr(signal, "tp1", None)),
            _quantize_price(getattr(signal, "tp2", None)),
            _quantize_price(getattr(signal, "tp3", None)),
        ],
        "main_tf": getattr(signal, "main_tf", None)
        or getattr(signal, "timeframe", None),
        "gate_reason": (getattr(signal, "thresholds_snapshot", {}) or {}).get("gate")
        or (getattr(signal, "debug_scores", {}) or {}).get("gate_tag"),
    }
    raw = _canonical_json(payload)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return digest


# ============================================================
# action_hash：同一个 action_type 下的“内容是否变化”的指纹
# ============================================================
def compute_action_hash(action_type: str, action_payload: Dict[str, Any]) -> str:
    """
    action_hash 的用途：
    - should_send 在 dedupe 的时候，会把 “changed_but_deduped” 标出来：
      即：虽然内容变了（hash 不同），但 TTL 未过期，仍然不发送。

    relevant 的选择：
    - entry/sl/tp_list/reason 等“用户可见&执行相关”的字段
    - 但不包含 timestamp 等易变字段（避免每次都变 hash）

    ⚠️ 注意：
    - 这里 entry 字段写了重复的 or action_payload.get("entry_price")（写两次）
      行为不受影响，但属于“可清理的小瑕疵”。
    """
    relevant = {
        "action_type": action_type,
        "entry": action_payload.get("entry_price")
        or action_payload.get("entry")
        or action_payload.get("entry_price"),
        "sl": action_payload.get("sl")
        or action_payload.get("invalidation_price")
        or action_payload.get("sl_price"),
        "tp_list": [
            action_payload.get("tp1"),
            action_payload.get("tp2"),
            action_payload.get("tp3"),
        ],
        "reason": action_payload.get("reason")
        or action_payload.get("explain")
        or action_payload.get("message"),
    }
    raw = _canonical_json(relevant)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# ============================================================
# per-symbol state：load / save（带损坏兜底）
# ============================================================
def load_state(symbol: str, base_dir: str = STATE_DIR) -> Dict[str, Any]:
    """
    读取某个 symbol 的 state 文件：
    - 若不存在：返回 _default_state()
    - 若解析失败/内容非法：
      - 尝试复制为 .bak 备份（不保证成功）
      - 返回 _default_state()

    设计哲学：
    - state 是“缓存”，坏了就丢，不影响主流程
    - 允许偶发重复提醒，但不允许 bot 因 state 损坏而 crash
    """
    path = _state_path(symbol, base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        return _default_state()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("state file root is not dict")
        data.setdefault("signals", {})
        data.setdefault("version", STATE_VERSION)
        data.setdefault("updated_at_utc", None)
        return data
    except Exception:
        backup_path = f"{path}.bak"
        try:
            shutil.copy(path, backup_path)
        except Exception:
            pass
        return _default_state()


def save_state(symbol: str, state: Dict[str, Any], base_dir: str = STATE_DIR) -> None:
    """
    将 state 写回对应的 per-symbol 文件。
    - 写入时会强制更新 version + updated_at_utc
    - 使用 indent=2，方便你手工读文件排查
    """
    path = _state_path(symbol, base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state["version"] = STATE_VERSION
    state["updated_at_utc"] = _isoformat(datetime.now(timezone.utc))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ============================================================
# signals 结构：key 与 entry 的组织方式
# ============================================================
def _signal_key(symbol: str, signal_id: str) -> str:
    """
    signals dict 的 key 统一格式： "<symbol>|<signal_id>"
    - 这里 symbol 是“原 symbol 字符串”（不是 sanitize_symbol）
    - 因此如果 symbol 字符串未来有不同表现形式（大小写/冒号），也会影响 key 一致性
    """
    return f"{symbol}|{signal_id}"


def _ensure_signal_entry(state: Dict[str, Any], signal_id: str, symbol: str, now: datetime) -> Dict[str, Any]:
    """
    获取或创建某个 signal 的 entry。
    entry 结构：
    - signal_id / symbol：冗余存储，方便人工读
    - created_at_utc：首次看到该 signal 的时间
    - last_seen_utc：最近一次看到该 signal 的时间（即本轮 run 更新）
    - actions_sent：按 action_type 记录发送历史（含 expires_at_utc + hash）
    """
    signals = state.setdefault("signals", {})
    key = _signal_key(symbol, signal_id)
    entry = signals.get(key)
    if not entry:
        entry = {
            "signal_id": signal_id,
            "symbol": symbol,
            "created_at_utc": _isoformat(now),
            "last_seen_utc": _isoformat(now),
            "actions_sent": {},
        }
        signals[key] = entry
    else:
        entry.setdefault("actions_sent", {})
        entry["last_seen_utc"] = _isoformat(now)
    return entry


# ============================================================
# should_send：是否需要发通知/动作（dedupe 核心）
# ============================================================
def should_send(
    state: Dict[str, Any],
    signal_id: str,
    symbol: str,
    action_type: str,
    now: datetime,
    *,
    action_hash: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    判断本次 action 是否应该发送。

    返回：
    - bool：True=发送，False=dedupe
    - info：调试信息（你可以打日志帮助排查“为何重复/为何被dedupe”）

    ⚠️ WATCH 特殊规则：
    - 你这里明确写死：WATCH 永远 SEND，且 reason="watch_no_cache"
    - 所以如果你觉得“WATCH 也太吵”，应该改这里（或在上游减少 WATCH 产出）
    - 但你现在的需求里：WATCH 走 summary bot，频率可接受，所以这样设计是合理的
    """
    if action_type == "WATCH":
        info = {
            "dedupe_key": _signal_key(symbol, signal_id),
            "symbol": symbol,
            "signal_id": signal_id,
            "action_type": action_type,
            "existing": False,
            "result": "SEND",
            "reason": "watch_no_cache",
        }
        return True, info

    # 对非 WATCH：确保 signal entry 存在
    entry = _ensure_signal_entry(state, signal_id, symbol, now)
    dedupe_key = f"{symbol}|{signal_id}|{action_type}"

    # record 结构：{sent_at_utc, expires_at_utc, hash?}
    record = entry.get("actions_sent", {}).get(action_type)

    info = {
        "dedupe_key": dedupe_key,
        "symbol": symbol,
        "signal_id": signal_id,
        "action_type": action_type,
        "existing": record is not None,
    }

    # 如果 record 存在且未过期：dedupe
    if record:
        expires_at = _parse_dt(record.get("expires_at_utc"))
        if expires_at and expires_at > now:
            info.update(
                {
                    "result": "DEDUPED",
                    "reason": "action_not_expired",
                    "expires_at_utc": record.get("expires_at_utc"),
                    # changed_but_deduped 用于排查：
                    # 内容变了（hash 不同），但 TTL 未到，仍不发送
                    "changed_but_deduped": bool(action_hash and action_hash != record.get("hash")),
                }
            )
            return False, info

    # 没 record 或 record 过期：允许发送
    info.update({"result": "SEND", "reason": "new_or_expired"})
    return True, info


# ============================================================
# TTL 与 expires_at：怎么计算“多长时间内算重复”
# ============================================================
def _compute_expires_at(action_type: str, now: datetime, valid_until: Optional[datetime]) -> datetime:
    """
    计算 expires_at（dedupe 过期时间）。

    基本规则：
    - expires_at = now + ACTION_TTLS[action_type]
    - 默认 TTL：若 action_type 不在 ACTION_TTLS，兜底 30m

    LIMIT_4H 的特殊规则：
    - 如果上游给了 valid_until（例如条件单本身的有效期）
      则 dedupe 过期时间取 min(now+TTL, valid_until)
      → 防止 dedupe 比计划本身还长（计划都失效了却仍 dedupe）
    """
    ttl_seconds = ACTION_TTLS.get(action_type, 1800)
    expires_at = now + timedelta(seconds=ttl_seconds)
    if action_type == "LIMIT_4H" and valid_until:
        return min(expires_at, valid_until)
    return expires_at


# ============================================================
# mark_sent：发送后写入 state（产生 dedupe 记录）
# ============================================================
def mark_sent(
    state: Dict[str, Any],
    signal_id: str,
    symbol: str,
    action_type: str,
    now: datetime,
    *,
    valid_until: Optional[datetime] = None,
    action_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """
    在你“真正发送动作/通知之后”调用，把 record 写回 state。

    WATCH 特殊规则：
    - 直接 return {}，完全不写 state
    - 这与 should_send 的“WATCH 不走缓存”一一对应

    非 WATCH：
    - 创建/更新 signal entry
    - 计算 expires_at
    - 写入 actions_sent[action_type] = {sent_at_utc, expires_at_utc, hash?}
    """
    if action_type == "WATCH":
        return {}

    entry = _ensure_signal_entry(state, signal_id, symbol, now)
    expires_at = _compute_expires_at(action_type, now, valid_until)

    record = {
        "sent_at_utc": _isoformat(now),
        "expires_at_utc": _isoformat(expires_at),
    }
    if action_hash:
        record["hash"] = action_hash

    entry.setdefault("actions_sent", {})[action_type] = record
    return record


# ============================================================
# 全局 state（global_state）：给“跨 symbol 的计划”用
# ============================================================
def load_global_state(path: str) -> Dict[str, Any]:
    """
    global state 与 per-symbol state 的区别：
    - per-symbol state：每个币一个文件，主要做 action 去重
    - global state：一个文件，记录跨币/跨 signal 的全局信息

    default_state：
    - active_plans：当前活跃的条件单计划（例如每个币的 LIMIT_4H 计划）
    - sent_events：一些“全局事件”的去重/记录

    同样：坏了就备份 .bak 并重置。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    default_state = {"active_plans": {}, "sent_events": {}}
    if not os.path.exists(path):
        return default_state
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("global state root is not dict")
        data.setdefault("active_plans", {})
        data.setdefault("sent_events", {})
        return data
    except Exception:
        backup_path = f"{path}.bak"
        try:
            shutil.copy(path, backup_path)
        except Exception:
            pass
        return default_state


def save_global_state(path: str, state: Dict[str, Any]) -> None:
    """
    保存 global state。
    - 不加 version（目前 global_state 结构简单）
    - indent=2 便于手工排查
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# 对外导出 API
__all__ = [
    "ACTION_TTLS",
    "compute_action_hash",
    "compute_signal_id",
    "load_global_state",
    "load_state",
    "mark_sent",
    "sanitize_symbol",
    "save_global_state",
    "save_state",
    "should_send",
]
