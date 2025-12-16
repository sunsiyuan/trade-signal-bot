import json
import os
from typing import Any, Dict


DEFAULT_STATE = {"active_plans": {}, "sent_events": {}, "dedupe_store": {}}


def load_state(path: str) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        return DEFAULT_STATE.copy()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return DEFAULT_STATE.copy()
            data.setdefault("active_plans", {})
            data.setdefault("sent_events", {})
            data.setdefault("dedupe_store", {})
            return data
    except Exception:
        return DEFAULT_STATE.copy()


def save_state(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


__all__ = ["load_state", "save_state", "DEFAULT_STATE"]
