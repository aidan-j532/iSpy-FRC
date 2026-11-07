
from __future__ import annotations

from typing import Any

CORE_ARTIFACT_TYPES = frozenset({"model", "source", "browse"})

MODEL_PATH_KEYS = ("file_path", "source_pt", "path", "name")

def _defn(schema: dict, key: str) -> dict:
    value = schema.get(key)
    return value if isinstance(value, dict) else {}

def _effective_value(schema: dict, settings: dict, key: str) -> Any:
    if isinstance(settings, dict) and key in settings:
        return settings[key]
    return _defn(schema, key).get("default")

def gate_is_active(schema: dict, settings: dict, defn: dict) -> bool:
    gate = defn.get("gated_by") if isinstance(defn, dict) else None
    if not gate:
        return True
    gates = gate if isinstance(gate, (list, tuple, set)) else (gate,)
    for name in gates:
        if not isinstance(name, str):
            continue
        if not bool(_effective_value(schema, settings or {}, name)):
            return False
    return True

def setting_is_required(schema: dict, key: str, settings: dict | None = None) -> bool:
    defn = _defn(schema, key)
    if not gate_is_active(schema, settings or {}, defn):
        return False
    explicit = defn.get("required")
    if explicit is not None:
        return bool(explicit)
    if defn.get("nullable"):
        return False
    if defn.get("type") in CORE_ARTIFACT_TYPES:
        return True
    if "default" not in defn:
        return True
    return not value_is_blank(defn.get("default"), defn)

def required_setting_keys(schema: dict, settings: dict | None = None) -> list[str]:
    if not isinstance(schema, dict):
        return []
    settings = settings if isinstance(settings, dict) else {}
    return [
        key
        for key in schema
        if setting_is_required(schema, key, settings)
    ]

def value_is_blank(value: Any, defn: dict | None = None) -> bool:
    if value is None:
        return True
    stype = (defn or {}).get("type")
    if stype == "model":
        if isinstance(value, dict):
            return not any(
                str(value.get(key) or "").strip() for key in MODEL_PATH_KEYS
            )
        if isinstance(value, (list, tuple)):
            return not any(str(part or "").strip() for part in value)
        return not str(value).strip()
    if stype in ("source", "browse", "text", "number", "select"):
        return not str(value).strip()
    if stype == "list":
        return isinstance(value, (list, tuple)) and len(value) == 0
    if stype == "range":
        return not isinstance(value, (list, tuple)) or len(value) != 2
    return False

def setting_is_complete(
    schema: dict, key: str, settings: dict | None = None
) -> bool:
    if not isinstance(settings, dict) or key not in settings:
        return False
    return not value_is_blank(settings[key], _defn(schema, key))

def completion_for_schema(schema: dict, settings: dict | None = None) -> dict:
    settings = settings if isinstance(settings, dict) else {}
    required = required_setting_keys(schema, settings)
    missing = [
        key for key in required if not setting_is_complete(schema, key, settings)
    ]
    total = len(required)
    done = total - len(missing)
    percent = 100 if total == 0 else int(round((done / total) * 100))
    labels = {
        key: str(_defn(schema, key).get("label") or key) for key in required
    }
    return {
        "required": required,
        "missing": missing,
        "missing_labels": [labels[key] for key in missing],
        "done": done,
        "total": total,
        "percent": percent,
        "complete": not missing,
    }
