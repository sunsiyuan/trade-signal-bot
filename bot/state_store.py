import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

STATE_DIR = os.getenv("STATE_DIR", ".state")
STATE_VERSION = 1
ACTION_TTLS = {
    "WATCH": 3600,
    "LIMIT_4H": 21600,
    "EXECUTE_NOW": 1800,
}


def sanitize_symbol(symbol: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", symbol or "unknown")
    return sanitized or "unknown"


def _state_path(symbol: str, base_dir: str = STATE_DIR) -> str:
    return os.path.join(base_dir, f"{sanitize_symbol(symbol)}.json")


def _default_state() -> Dict[str, Any]:
    return {"version": STATE_VERSION, "updated_at_utc": None, "signals": {}}


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
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
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_signal_id(signal: Any) -> str:
    base_signal_id = getattr(signal, "signal_id", None)
    if base_signal_id:
        return str(base_signal_id)

    payload = {
        "symbol": getattr(signal, "symbol", None),
        "regime": getattr(signal, "regime", None)
        or getattr(getattr(signal, "snapshot", None), "regime", None),
        "strategy": getattr(signal, "strategy", None) or getattr(signal, "setup_type", None),
        "entry": getattr(signal, "entry", None),
        "sl": getattr(signal, "sl", None),
        "tp_list": [
            getattr(signal, "tp1", None),
            getattr(signal, "tp2", None),
            getattr(signal, "tp3", None),
        ],
        "main_tf": getattr(signal, "main_tf", None)
        or getattr(signal, "timeframe", None),
        "gate_reason": (getattr(signal, "thresholds_snapshot", {}) or {}).get("gate")
        or (getattr(signal, "debug_scores", {}) or {}).get("gate_tag"),
    }
    raw = _canonical_json(payload)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return digest


def compute_action_hash(action_type: str, action_payload: Dict[str, Any]) -> str:
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


def load_state(symbol: str, base_dir: str = STATE_DIR) -> Dict[str, Any]:
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
    path = _state_path(symbol, base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state["version"] = STATE_VERSION
    state["updated_at_utc"] = _isoformat(datetime.now(timezone.utc))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _signal_key(symbol: str, signal_id: str) -> str:
    return f"{symbol}|{signal_id}"


def _ensure_signal_entry(state: Dict[str, Any], signal_id: str, symbol: str, now: datetime) -> Dict[str, Any]:
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


def should_send(
    state: Dict[str, Any],
    signal_id: str,
    symbol: str,
    action_type: str,
    now: datetime,
    *,
    action_hash: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    entry = _ensure_signal_entry(state, signal_id, symbol, now)
    dedupe_key = f"{symbol}|{signal_id}|{action_type}"
    record = entry.get("actions_sent", {}).get(action_type)
    info = {
        "dedupe_key": dedupe_key,
        "symbol": symbol,
        "signal_id": signal_id,
        "action_type": action_type,
        "existing": record is not None,
    }
    if record:
        expires_at = _parse_dt(record.get("expires_at_utc"))
        if expires_at and expires_at > now:
            info.update(
                {
                    "result": "DEDUPED",
                    "reason": "action_not_expired",
                    "expires_at_utc": record.get("expires_at_utc"),
                    "changed_but_deduped": bool(action_hash and action_hash != record.get("hash")),
                }
            )
            return False, info
    info.update({"result": "SEND", "reason": "new_or_expired"})
    return True, info


def _compute_expires_at(action_type: str, now: datetime, valid_until: Optional[datetime]) -> datetime:
    ttl_seconds = ACTION_TTLS.get(action_type, 1800)
    expires_at = now + timedelta(seconds=ttl_seconds)
    if action_type == "LIMIT_4H" and valid_until:
        return min(expires_at, valid_until)
    return expires_at


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


def load_global_state(path: str) -> Dict[str, Any]:
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


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
