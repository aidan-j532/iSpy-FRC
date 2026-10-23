"""Shared schema helpers for required settings and completion progress.

The editor sends this module's result to the browser, and the same result is
used by the backend save/validate routes.  Keeping the rule here means a field
cannot look required in the canvas and optional to the API (or the other way
around).

Rule (unless a schema explicitly overrides it with ``required``):

* ``nullable: true`` is optional.
* ``model``, ``source`` and ``browse`` are core artifact fields and are
  required, even when their default is an empty string.
* Any other field without a ``default`` is required.
* A field with a default is optional because it already has a usable value.
* A field hidden behind an inactive ``gated_by`` is not counted.
"""

from __future__ import annotations

from typing import Any

# Types whose whole job is to point at a real artifact.  These stay required
# even when a schema gives them an empty default so the canvas cannot report a
# camera/model node as complete before the user has chosen something.
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
    """Return False when a conditional field's controller is off."""
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
    """Whether ``key`` counts toward completion for this schema/settings pair."""
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
    return "default" not in defn


def required_setting_keys(schema: dict, settings: dict | None = None) -> list[str]:
    """Active required keys, in schema order."""
    if not isinstance(schema, dict):
        return []
    settings = settings if isinstance(settings, dict) else {}
    return [
        key
        for key in schema
        if setting_is_required(schema, key, settings)
    ]


def value_is_blank(value: Any, defn: dict | None = None) -> bool:
    """True for values that should not satisfy a required field."""
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
    """Whether a required field has a usable value (defaults included)."""
    defn = _defn(schema, key)
    value = _effective_value(schema, settings or {}, key)
    return not value_is_blank(value, defn)


def completion_for_schema(schema: dict, settings: dict | None = None) -> dict:
    """Progress payload shared by the editor canvas and backend validation."""
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
