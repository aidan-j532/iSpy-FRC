#AI DISCLAMER: This file was largely written by AI for time saving, while the framework was developed by me. 
"""Unit packs: JSON-defined measurement/output conventions.

A unit pack controls how user measurements are interpreted (input_unit),
how they are converted internally (input_to_inches), and how coordinates are
published (inches_to_output, output_unit, axis). Packs live as JSON files in
the top-level ``Units/`` directory and are selectable from the Add-ons UI.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path.cwd()
_UNITS_DIR = _REPO_ROOT / "Units"

# Built-in fallback packs (used if Units/ is missing).
_BUILTIN = {
    "frc": {
        "key": "frc",
        "name": "FRC (inches in, meters out)",
        "description": "FRC convention: measure in inches, output meters.",
        "input_unit": "inch",
        "output_unit": "meter",
        "input_to_inches": 1.0,
        "inches_to_output": 0.0254,
        "axis": {"x": 1.0, "y": 1.0, "z": 1.0, "yaw": 1.0},
    },
    "metric": {
        "key": "metric",
        "name": "Metric (meters in, meters out)",
        "description": "Metric convention: measure and output in meters.",
        "input_unit": "meter",
        "output_unit": "meter",
        "input_to_inches": 39.37007874,
        "inches_to_output": 0.0254,
        "axis": {"x": 1.0, "y": 1.0, "z": 1.0, "yaw": 1.0},
    },
    "imperial": {
        "key": "imperial",
        "name": "Imperial (inches in, inches out)",
        "description": "Imperial convention: measure and output in inches.",
        "input_unit": "inch",
        "output_unit": "inch",
        "input_to_inches": 1.0,
        "inches_to_output": 1.0,
        "axis": {"x": 1.0, "y": 1.0, "z": 1.0, "yaw": 1.0},
    },
}


def _load_packs() -> dict:
    packs = dict(_BUILTIN)
    try:
        if _UNITS_DIR.exists():
            for path in sorted(_UNITS_DIR.glob("*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    key = str(data.get("key") or path.stem)
                    data.setdefault("key", key)
                    packs[key] = data
                except Exception as e:
                    logger.warning("Could not load unit pack %s: %s", path, e)
    except Exception as e:
        logger.warning("Unit pack discovery failed: %s", e)
    return packs


_PACKS = _load_packs()


def reload_packs() -> None:
    global _PACKS
    _PACKS = _load_packs()


def list_unit_packs() -> list[dict]:
    return list(_PACKS.values())


def get_unit_pack(key: str) -> dict | None:
    return _PACKS.get(str(key).lower().strip())


def input_to_inches(unit: str) -> float:
    pack = get_unit_pack(unit)
    if pack:
        return float(pack.get("input_to_inches", 1.0))
    # legacy fallback to built-in tables
    from iSpy.config.iSpyConfig import _UNIT_TO_INCHES
    key = str(unit).lower().strip()
    return float(_UNIT_TO_INCHES.get(key, 1.0))


def inches_to_output(unit: str) -> float:
    pack = get_unit_pack(unit)
    if pack:
        return float(pack.get("inches_to_output", 1.0))
    # legacy fallback to built-in tables
    from iSpy.config.iSpyConfig import _INCHES_TO_OUTPUT_UNIT
    key = str(unit).lower().strip()
    return float(_INCHES_TO_OUTPUT_UNIT.get(key, _INCHES_TO_OUTPUT_UNIT.get("frc", 0.0254)))


def unit_axis(unit: str) -> dict:
    pack = get_unit_pack(unit)
    if pack:
        ax = pack.get("axis") or {}
        return {
            "x": float(ax.get("x", 1.0)),
            "y": float(ax.get("y", 1.0)),
            "z": float(ax.get("z", 1.0)),
            "yaw": float(ax.get("yaw", 1.0)),
        }
    return {"x": 1.0, "y": 1.0, "z": 1.0, "yaw": 1.0}


def unit_input_label(unit: str) -> str:
    pack = get_unit_pack(unit)
    if pack:
        return str(pack.get("input_unit", unit))
    return str(unit)


def unit_output_label(unit: str) -> str:
    pack = get_unit_pack(unit)
    if pack:
        return str(pack.get("output_unit", unit))
    return str(unit)


def unit_name(unit: str) -> str:
    pack = get_unit_pack(unit)
    if pack:
        return str(pack.get("name", unit))
    return str(unit)
