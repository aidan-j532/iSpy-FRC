#AI DISCLAMER: This file was largely written by AI for time saving, while the framework was developed by me.
"""Unit packs: JSON-defined measurement, output, and viewer conventions.

A unit pack is the single source of truth for:

- how user-entered measurements become internal inches (``input_to_inches``)
- how internal inches become published values (``inches_to_output``)
- sign/axis conventions for published positions and angles (``axis``)
- labels shown in the UI (``labels``)
- default camera mount angles (``defaults``)
- 3D viewer axis mapping, bounds, scale sliders, grid, and camera (``viewer``)

Packs live as JSON files in the top-level ``Units/`` directory.  See
``Units/README.md`` for the full schema.
"""

import copy
import json
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

_MODULE_ROOT = Path(__file__).resolve().parents[2]
if (_MODULE_ROOT / "Units").is_dir():
    _REPO_ROOT = _MODULE_ROOT
else:
    _REPO_ROOT = Path.cwd()
_UNITS_DIR = _REPO_ROOT / "Units"

# Model/PnP output is meters; the rest of the code works in inches internally.
METERS_TO_INCHES = 1.0 / 0.0254

_AXIS_DEFAULTS = {
    "x": 1.0,
    "y": 1.0,
    "z": 1.0,
    "roll": 1.0,
    "pitch": 1.0,
    "yaw": 1.0,
}

_LABEL_DEFAULTS = {
    "input": "",
    "output": "",
    "speed": "",
}

_DEFAULTS_DEFAULTS = {
    "camera_pitch_deg": 0.0,
}

_VIEWER_DEFAULTS = {
    # Each entry says which robot axis feeds that viewer axis.  Only signed
    # axis permutations with a +1 determinant are valid rotations.
    "robot_to_viewer": ["x", "z", "-y"],
    # Optional extra rotation applied to camera overlays.  Normally zero:
    # the physical mount angles already define the camera orientation.
    "camera_roll_offset_deg": 0.0,
    "camera_pitch_offset_deg": 0.0,
    "camera_yaw_offset_deg": 0.0,
    # Scene-scale rendering values.  They must grow with the pack's output
    # scale or the fog/axes disappear (or swallow the scene) in imperial mode.
    "fog_density": 0.04,
    "axes_size": 8.0,
    "bounds": {
        "x": {"min": -10.0, "max": 10.0},
        "y": {"min": -10.0, "max": 10.0},
        "z": {"min": 0.0, "max": 10.0},
    },
    "scale": {
        "x": {"min": -10.0, "max": 10.0, "step": 0.1, "default": 1.0},
        "y": {"min": -10.0, "max": 10.0, "step": 0.1, "default": 1.0},
        "z": {"min": -10.0, "max": 10.0, "step": 0.1, "default": 1.0},
    },
    "grid": {
        "extent": None,  # derived from bounds when omitted
        "divisions": 80,
        "marker_size": 0.25,
    },
    "camera": {
        "fov": 60.0,
        "near": 0.1,
        "far": 200.0,
        "position": [9.0, 9.0, 9.0],
        "target": [0.0, 0.0, 0.0],
    },
}

_BASE_DEFAULTS = {
    "schema_version": 2,
    "key": "",
    "name": "",
    "description": "",
    "input_unit": "inch",
    "output_unit": "inch",
    "input_to_inches": 1.0,
    "inches_to_output": 1.0,
    "aliases": [],
    "axis": dict(_AXIS_DEFAULTS),
    "labels": dict(_LABEL_DEFAULTS),
    "defaults": dict(_DEFAULTS_DEFAULTS),
    "viewer": copy.deepcopy(_VIEWER_DEFAULTS),
}

# Built-in fallback packs (used only if Units/ is missing or unreadable).
_BUILTIN = {
    "frc": {
        "schema_version": 2,
        "key": "frc",
        "name": "FRC (inches in, meters out)",
        "description": "FRC convention: measure in inches, output meters (WPILib).",
        "input_unit": "inch",
        "output_unit": "meter",
        "input_to_inches": 1.0,
        "inches_to_output": 0.0254,
        "aliases": ["wpilib"],
        "labels": {"input": "in", "output": "m", "speed": "m/s"},
        "defaults": {"camera_pitch_deg": 0.0},
        "axis": dict(_AXIS_DEFAULTS),
        "viewer": copy.deepcopy(_VIEWER_DEFAULTS),
    },
    "metric": {
        "schema_version": 2,
        "key": "metric",
        "name": "Metric (meters in, meters out)",
        "description": "Metric convention: measure and output in meters.",
        "input_unit": "meter",
        "output_unit": "meter",
        "input_to_inches": 39.37007874015748,
        "inches_to_output": 0.0254,
        "aliases": ["m", "meter", "meters", "cm", "centimeter", "centimeters", "si"],
        "labels": {"input": "m", "output": "m", "speed": "m/s"},
        "defaults": {"camera_pitch_deg": 0.0},
        "axis": dict(_AXIS_DEFAULTS),
        "viewer": copy.deepcopy(_VIEWER_DEFAULTS),
    },
    "imperial": {
        "schema_version": 2,
        "key": "imperial",
        "name": "Imperial (inches in, inches out)",
        "description": "Imperial convention: measure and output in inches.",
        "input_unit": "inch",
        "output_unit": "inch",
        "input_to_inches": 1.0,
        "inches_to_output": 1.0,
        "aliases": ["in", "inch", "inches", "ft", "foot", "feet"],
        "labels": {"input": "in", "output": "in", "speed": "in/s"},
        "defaults": {"camera_pitch_deg": 0.0},
        "axis": dict(_AXIS_DEFAULTS),
        "viewer": {
            "robot_to_viewer": ["x", "z", "-y"],
            "camera_roll_offset_deg": 0.0,
            "camera_pitch_offset_deg": 0.0,
            "camera_yaw_offset_deg": 0.0,
            "fog_density": 0.001,
            "axes_size": 320.0,
            "bounds": {
                "x": {"min": -400.0, "max": 400.0},
                "y": {"min": -400.0, "max": 400.0},
                "z": {"min": 0.0, "max": 200.0},
            },
            "scale": copy.deepcopy(_VIEWER_DEFAULTS["scale"]),
            "grid": {
                "extent": 800.0,
                "divisions": 80,
                "marker_size": 10.0,
            },
            "camera": {
                "fov": 60.0,
                "near": 0.1,
                "far": 5000.0,
                "position": [350.0, 350.0, 350.0],
                "target": [0.0, 0.0, 0.0],
            },
        },
    },
}

_SHORT_LABELS = {
    "inch": "in",
    "inches": "in",
    "foot": "ft",
    "feet": "ft",
    "meter": "m",
    "meters": "m",
    "centimeter": "cm",
    "centimeters": "cm",
    "millimeter": "mm",
    "millimeters": "mm",
    "frc": "in",
}


def _as_float(value, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def _as_positive_float(value, default: float) -> float:
    result = _as_float(value, default)
    return result if result > 0 else float(default)


def _short_label(unit: str) -> str:
    key = str(unit or "").strip().lower()
    return _SHORT_LABELS.get(key, str(unit or ""))


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _axis_expression(expr) -> tuple[float, float, float] | None:
    """Parse ``x``, ``-y``, ``+z`` into a signed unit vector."""
    if not isinstance(expr, str):
        return None
    text = expr.strip().lower().replace(" ", "")
    if not text:
        return None
    sign = 1.0
    if text[0] in "+-":
        sign = -1.0 if text[0] == "-" else 1.0
        text = text[1:]
    vectors = {
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
    }
    vec = vectors.get(text)
    if vec is None:
        return None
    return (sign * vec[0], sign * vec[1], sign * vec[2])


def axis_map_matrix(axis_map) -> list[list[float]] | None:
    """Return the 3x3 rotation matrix for a signed axis permutation.

    The determinant must be +1 so orientation quaternions stay valid.
    """
    if not isinstance(axis_map, (list, tuple)) or len(axis_map) != 3:
        return None
    rows = [_axis_expression(item) for item in axis_map]
    if any(row is None for row in rows):
        return None
    matrix = [list(row) for row in rows]
    det = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    return matrix if abs(det - 1.0) < 1e-9 else None


def _normalize_range(value, default: dict) -> dict:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = _as_float(value[0], default["min"]), _as_float(
            value[1], default["max"]
        )
    elif isinstance(value, dict):
        lo = _as_float(value.get("min", default["min"]), default["min"])
        hi = _as_float(value.get("max", default["max"]), default["max"])
    else:
        lo, hi = default["min"], default["max"]
    return {"min": min(lo, hi), "max": max(lo, hi)}


def _normalize_scale(value, default: dict) -> dict:
    if not isinstance(value, dict):
        return copy.deepcopy(default)
    lo = _as_float(value.get("min", default["min"]), default["min"])
    hi = _as_float(value.get("max", default["max"]), default["max"])
    if lo > hi:
        lo, hi = hi, lo
    step = _as_positive_float(value.get("step", default["step"]), default["step"])
    start = _as_float(value.get("default", default["default"]), default["default"])
    start = min(max(start, lo), hi)
    return {"min": lo, "max": hi, "step": step, "default": start}


def _normalize_viewer(raw: dict, pack: dict) -> dict:
    viewer = _deep_merge(_VIEWER_DEFAULTS, raw or {})

    if axis_map_matrix(viewer.get("robot_to_viewer")) is None:
        logger.warning(
            "Unit pack %s has an invalid viewer.robot_to_viewer value %r; "
            "falling back to x,z,-y",
            pack.get("key") or "?",
            viewer.get("robot_to_viewer"),
        )
        viewer["robot_to_viewer"] = list(_VIEWER_DEFAULTS["robot_to_viewer"])

    bounds = {}
    for axis in ("x", "y", "z"):
        bounds[axis] = _normalize_range(
            (viewer.get("bounds") or {}).get(axis),
            _VIEWER_DEFAULTS["bounds"][axis],
        )
    viewer["bounds"] = bounds

    scale = {}
    for axis in ("x", "y", "z"):
        scale[axis] = _normalize_scale(
            (viewer.get("scale") or {}).get(axis),
            _VIEWER_DEFAULTS["scale"][axis],
        )
    viewer["scale"] = scale

    grid = _deep_merge(_VIEWER_DEFAULTS["grid"], viewer.get("grid") or {})
    grid["divisions"] = max(1, int(_as_positive_float(grid.get("divisions"), 80)))
    grid["marker_size"] = _as_positive_float(grid.get("marker_size"), 0.25)
    if grid.get("extent") is not None:
        grid["extent"] = _as_positive_float(grid.get("extent"), 20.0)
    viewer["grid"] = grid

    camera = _deep_merge(_VIEWER_DEFAULTS["camera"], viewer.get("camera") or {})
    camera["fov"] = min(
        179.0, max(1.0, _as_float(camera.get("fov"), 60.0))
    )
    camera["near"] = _as_positive_float(camera.get("near"), 0.1)
    camera["far"] = _as_positive_float(camera.get("far"), 200.0)
    if camera["far"] <= camera["near"]:
        camera["far"] = camera["near"] * 1000.0
    for key, default in (
        ("position", [9.0, 9.0, 9.0]),
        ("target", [0.0, 0.0, 0.0]),
    ):
        value = camera.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            value = default
        camera[key] = [_as_float(item, default[index]) for index, item in enumerate(value)]
    viewer["camera"] = camera

    for key in (
        "camera_roll_offset_deg",
        "camera_pitch_offset_deg",
        "camera_yaw_offset_deg",
    ):
        viewer[key] = _as_float(viewer.get(key), 0.0)
    viewer["fog_density"] = max(
        0.0, _as_float(viewer.get("fog_density"), 0.04)
    )
    viewer["axes_size"] = _as_positive_float(viewer.get("axes_size"), 8.0)
    return viewer


def _normalize_pack(raw: dict, key: str | None = None) -> dict:
    data = _deep_merge(_BASE_DEFAULTS, raw or {})
    pack_key = str(key or data.get("key") or "").strip().lower()
    if not pack_key:
        pack_key = "custom"
    data["key"] = pack_key
    data["name"] = str(data.get("name") or pack_key)
    data["description"] = str(data.get("description") or "")
    data["input_unit"] = str(data.get("input_unit") or "inch")
    data["output_unit"] = str(data.get("output_unit") or "inch")
    data["input_to_inches"] = _as_positive_float(data.get("input_to_inches"), 1.0)
    data["inches_to_output"] = _as_positive_float(data.get("inches_to_output"), 1.0)

    aliases = data.get("aliases")
    if not isinstance(aliases, list):
        aliases = []
    data["aliases"] = [str(item).strip().lower() for item in aliases if str(item).strip()]

    axis = data.get("axis")
    if not isinstance(axis, dict):
        axis = {}
    data["axis"] = {
        name: _as_float(axis.get(name), 1.0)
        for name in _AXIS_DEFAULTS
    }

    labels = _deep_merge(_LABEL_DEFAULTS, data.get("labels") or {})
    if not labels.get("input"):
        labels["input"] = _short_label(data["input_unit"])
    if not labels.get("output"):
        labels["output"] = _short_label(data["output_unit"])
    if not labels.get("speed"):
        labels["speed"] = f"{labels['output']}/s"
    data["labels"] = {
        "input": str(labels["input"]),
        "output": str(labels["output"]),
        "speed": str(labels["speed"]),
    }

    defaults = _deep_merge(_DEFAULTS_DEFAULTS, data.get("defaults") or {})
    defaults["camera_pitch_deg"] = _as_float(
        defaults.get("camera_pitch_deg"), 0.0
    )
    data["defaults"] = defaults

    data["viewer"] = _normalize_viewer(data.get("viewer") or {}, data)
    return data


def _load_packs() -> dict:
    packs = {}
    for key, raw in _BUILTIN.items():
        packs[key] = _normalize_pack(raw, key)
    try:
        if _UNITS_DIR.exists():
            for path in sorted(_UNITS_DIR.glob("*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(data, dict):
                        raise ValueError("unit pack must be a JSON object")
                    key = str(data.get("key") or path.stem).strip().lower()
                    merged = _deep_merge(_BUILTIN.get(key, {}), data)
                    packs[key] = _normalize_pack(merged, key)
                except Exception as exc:
                    logger.warning("Could not load unit pack %s: %s", path, exc)
    except Exception as exc:
        logger.warning("Unit pack discovery failed: %s", exc)
    return packs


_PACKS = _load_packs()


def reload_packs() -> None:
    global _PACKS
    _PACKS = _load_packs()


def list_unit_packs() -> list[dict]:
    return [copy.deepcopy(pack) for pack in _PACKS.values()]


def resolve_unit_pack(unit: str) -> dict | None:
    """Resolve a pack by key, alias, input unit, or output unit."""
    if unit is None:
        return None
    key = str(unit).strip().lower()
    if not key:
        return None
    pack = _PACKS.get(key)
    if pack is not None:
        return pack
    for candidate in _PACKS.values():
        if key in candidate.get("aliases", []):
            return candidate
    for candidate in _PACKS.values():
        if key == str(candidate.get("input_unit", "")).strip().lower():
            return candidate
    for candidate in _PACKS.values():
        if key == str(candidate.get("output_unit", "")).strip().lower():
            return candidate
    return None


def get_unit_pack(key: str) -> dict | None:
    return resolve_unit_pack(key)


def _pack_or_default(unit: str | None) -> dict:
    pack = resolve_unit_pack(unit)
    if pack is not None:
        return pack
    return _PACKS.get("frc") or _normalize_pack(_BUILTIN["frc"], "frc")


def input_to_inches(unit: str) -> float:
    return float(_pack_or_default(unit)["input_to_inches"])


def inches_to_output(unit: str) -> float:
    return float(_pack_or_default(unit)["inches_to_output"])


def meters_to_output(unit: str) -> float:
    return METERS_TO_INCHES * inches_to_output(unit)


def unit_axis(unit: str) -> dict:
    axis = _pack_or_default(unit)["axis"]
    return {name: float(axis.get(name, 1.0)) for name in _AXIS_DEFAULTS}


def unit_input_label(unit: str) -> str:
    return str(_pack_or_default(unit)["labels"]["input"])


def unit_output_label(unit: str) -> str:
    return str(_pack_or_default(unit)["labels"]["output"])


def unit_speed_label(unit: str) -> str:
    return str(_pack_or_default(unit)["labels"]["speed"])


def unit_name(unit: str) -> str:
    return str(_pack_or_default(unit)["name"])


def unit_default_camera_pitch(unit: str) -> float:
    return float(_pack_or_default(unit)["defaults"]["camera_pitch_deg"])


def output_conversions() -> dict[str, float]:
    """Map every pack key/alias/label to an inches -> output factor."""
    result: dict[str, float] = {}
    for pack in _PACKS.values():
        result[str(pack["key"]).lower()] = float(pack["inches_to_output"])
    for pack in _PACKS.values():
        candidates = list(pack.get("aliases", []))
        candidates.append(str(pack.get("output_unit", "")).strip().lower())
        candidates.append(str(pack["labels"].get("output", "")).strip().lower())
        for alias in candidates:
            if alias:
                result.setdefault(alias, float(pack["inches_to_output"]))
    return result


def input_conversions() -> dict[str, float]:
    """Map every pack key/alias/label to an input -> inches factor."""
    result: dict[str, float] = {}
    for pack in _PACKS.values():
        result[str(pack["key"]).lower()] = float(pack["input_to_inches"])
    for pack in _PACKS.values():
        candidates = list(pack.get("aliases", []))
        candidates.append(str(pack.get("input_unit", "")).strip().lower())
        candidates.append(str(pack["labels"].get("input", "")).strip().lower())
        for alias in candidates:
            if alias:
                result.setdefault(alias, float(pack["input_to_inches"]))
    return result


def output_point_from_inches(point, unit: str, *, apply_axis: bool = True):
    """Convert an internal inch point into the pack's published convention."""
    scale = inches_to_output(unit)
    axis = unit_axis(unit) if apply_axis else _AXIS_DEFAULTS
    return (
        float(point[0]) * scale * axis["x"],
        float(point[1]) * scale * axis["y"],
        float(point[2]) * scale * axis["z"],
    )


def output_point_from_meters(point, unit: str, *, apply_axis: bool = True):
    return output_point_from_inches(
        (
            float(point[0]) * METERS_TO_INCHES,
            float(point[1]) * METERS_TO_INCHES,
            float(point[2]) * METERS_TO_INCHES,
        ),
        unit,
        apply_axis=apply_axis,
    )


def input_point_to_output(point, unit: str, *, apply_axis: bool = True):
    """Convert a point typed in the pack's input unit to published units."""
    scale = input_to_inches(unit) * inches_to_output(unit)
    axis = unit_axis(unit) if apply_axis else _AXIS_DEFAULTS
    return (
        float(point[0]) * scale * axis["x"],
        float(point[1]) * scale * axis["y"],
        float(point[2]) * scale * axis["z"],
    )


def output_length(value_inches: float, unit: str, axis: str = "z") -> float:
    factors = unit_axis(unit)
    return float(value_inches) * inches_to_output(unit) * factors.get(axis, 1.0)


def output_angle(value_rad: float, unit: str, axis: str) -> float:
    return float(value_rad) * unit_axis(unit).get(axis, 1.0)


def output_euler(roll: float, pitch: float, yaw: float, unit: str):
    """Apply the pack's axis convention to an orientation.

    A coordinate convention with an odd number of axis flips is a reflection,
    not a rotation.  In that case orientations cannot be represented by Euler
    angles, so their components are sign-scaled instead of silently producing
    a wrong quaternion.
    """
    axis = unit_axis(unit)
    if all(abs(axis[name] - 1.0) < 1e-12 for name in "xyz"):
        return (roll, pitch, yaw)
    import numpy as np

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    rotation = rz @ ry @ rx
    signs = np.diag([axis["x"], axis["y"], axis["z"]])
    rotation = signs @ rotation @ signs
    if np.linalg.det(rotation) < 0:
        return (
            roll * axis["roll"],
            pitch * axis["pitch"],
            yaw * axis["yaw"],
        )

    sy_ = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    if sy_ > 1e-6:
        return (
            math.atan2(rotation[2, 1], rotation[2, 2]),
            math.atan2(-rotation[2, 0], sy_),
            math.atan2(rotation[1, 0], rotation[0, 0]),
        )
    return (
        math.atan2(-rotation[1, 2], rotation[1, 1]),
        math.atan2(-rotation[2, 0], sy_),
        0.0,
    )


def _viewer_bounds_viewer(viewer: dict) -> dict:
    matrix = axis_map_matrix(viewer["robot_to_viewer"])
    if matrix is None:
        matrix = axis_map_matrix(_VIEWER_DEFAULTS["robot_to_viewer"])
    bounds = viewer["bounds"]
    corners = []
    for x in (bounds["x"]["min"], bounds["x"]["max"]):
        for y in (bounds["y"]["min"], bounds["y"]["max"]):
            for z in (bounds["z"]["min"], bounds["z"]["max"]):
                corners.append(
                    (
                        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
                        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
                        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z,
                    )
                )
    out = {}
    for index, axis in enumerate(("x", "y", "z")):
        values = [corner[index] for corner in corners]
        out[axis] = {"min": min(values), "max": max(values)}
    return out


def viewer_settings(unit: str) -> dict:
    """Normalized viewer settings for a unit pack (JSON-serializable)."""
    pack = _pack_or_default(unit)
    viewer = copy.deepcopy(pack["viewer"])
    bounds_viewer = _viewer_bounds_viewer(viewer)
    grid = viewer["grid"]
    if not grid.get("extent"):
        grid["extent"] = max(
            bounds_viewer["x"]["max"] - bounds_viewer["x"]["min"],
            bounds_viewer["z"]["max"] - bounds_viewer["z"]["min"],
            1.0,
        )
    return {
        "unit": pack["key"],
        "unit_name": pack["name"],
        "input_label": pack["labels"]["input"],
        "output_label": pack["labels"]["output"],
        "speed_label": pack["labels"]["speed"],
        "axis": pack["axis"],
        "defaults": pack["defaults"],
        "robot_to_viewer": viewer["robot_to_viewer"],
        "bounds": viewer["bounds"],
        "bounds_viewer": bounds_viewer,
        "scale": viewer["scale"],
        "grid": grid,
        "camera": viewer["camera"],
        "camera_roll_offset_deg": viewer["camera_roll_offset_deg"],
        "camera_pitch_offset_deg": viewer["camera_pitch_offset_deg"],
        "camera_yaw_offset_deg": viewer["camera_yaw_offset_deg"],
        "fog_density": viewer["fog_density"],
        "axes_size": viewer["axes_size"],
    }


def validate_unit_pack(data) -> list[str]:
    """Return human-readable validation errors for an uploaded/written pack."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["unit pack must be a JSON object"]

    key = str(data.get("key") or "").strip()
    if not key:
        errors.append("missing key")

    for field in ("input_to_inches", "inches_to_output"):
        try:
            value = float(data.get(field))
        except (TypeError, ValueError):
            errors.append(f"{field} must be a number")
            continue
        if not math.isfinite(value) or value <= 0:
            errors.append(f"{field} must be positive")

    if not str(data.get("input_unit") or "").strip():
        errors.append("missing input_unit")
    if not str(data.get("output_unit") or "").strip():
        errors.append("missing output_unit")

    axis = data.get("axis")
    if axis is not None:
        if not isinstance(axis, dict):
            errors.append("axis must be an object")
        else:
            for name in _AXIS_DEFAULTS:
                if name in axis:
                    try:
                        value = float(axis[name])
                    except (TypeError, ValueError):
                        errors.append(f"axis.{name} must be a number")
                        continue
                    if not math.isfinite(value):
                        errors.append(f"axis.{name} must be finite")

    viewer = data.get("viewer")
    if viewer is not None:
        if not isinstance(viewer, dict):
            errors.append("viewer must be an object")
        else:
            axis_map = viewer.get("robot_to_viewer")
            if axis_map is not None and axis_map_matrix(axis_map) is None:
                errors.append(
                    "viewer.robot_to_viewer must be a signed x/y/z permutation "
                    "with determinant +1 (for example [\"x\", \"z\", \"-y\"])"
                )
            bounds = viewer.get("bounds")
            if bounds is not None and not isinstance(bounds, dict):
                errors.append("viewer.bounds must be an object")
            scale = viewer.get("scale")
            if scale is not None and not isinstance(scale, dict):
                errors.append("viewer.scale must be an object")
            camera = viewer.get("camera")
            if camera is not None and not isinstance(camera, dict):
                errors.append("viewer.camera must be an object")
            for field, minimum in (
                ("fog_density", 0.0),
                ("axes_size", 0.0),
            ):
                if field not in viewer:
                    continue
                try:
                    value = float(viewer[field])
                except (TypeError, ValueError):
                    errors.append(f"viewer.{field} must be a number")
                    continue
                if not math.isfinite(value) or value < minimum:
                    errors.append(f"viewer.{field} must be >= {minimum:g}")
                elif field == "axes_size" and value == minimum:
                    errors.append("viewer.axes_size must be positive")
    return errors
