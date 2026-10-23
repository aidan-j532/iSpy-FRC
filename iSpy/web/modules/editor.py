"""Node editor backend.

The editor is deliberately additive: it reads and writes the existing
``camera_configs`` / ``plugins`` structures, and only keeps canvas metadata in
the namespaced ``editor_graph`` block.  Runtime objects are never touched, so
every save reports that a restart is required.
"""

from __future__ import annotations

import copy
import logging
import math
import re
import threading
from pathlib import Path

from flask import jsonify, render_template, request

from iSpy.config.iSpyConfig import (
    _CAMERA_CORE_KEYS,
    default_vision_model,
    get_pipeline_name,
    get_pipeline_settings,
    iSpyCameraConfig,
    is_model_backed_pipeline,
    normalize_camera_entry,
)
from iSpy.config.schema_fields import completion_for_schema
from iSpy.plugins._loader import load_plugins
from iSpy.plugins.bases import FrameProcessorBase, TrackerBase, UtilityBase
from iSpy.web.Backend.PluginStatus import (
    _PLUGIN_ROOT,
    _build_vision_pipeline_payloads,
    _coerce_setting_value,
    require_local_or_token,
)
from iSpy.web.Backend.WebModule import WebModule
from iSpy.vision.Cameras import CAMERA_TYPE_LABELS, get_camera_classes
from iSpy.vision.pipelines import get_pipeline_classes

logger = logging.getLogger(__name__)

# Fixed category rows.  The frontend draws the same values; the backend keeps
# them authoritative so a hand-written layout cannot put a camera in the
# tracker row.
ROW_Y = {
    "camera": 40,
    "frame_processor": 220,
    "pipeline": 400,
    "tracker": 580,
    "utility": 760,
}
ROW_LABELS = {
    "camera": "Camera",
    "frame_processor": "Frame Processors",
    "pipeline": "Pipeline",
    "tracker": "Trackers",
    "utility": "Utilities",
}

_EDGE_TARGETS = {
    "camera": {"frame_processor", "pipeline"},
    "frame_processor": {"frame_processor", "pipeline"},
    "pipeline": {"tracker", "utility"},
    "tracker": {"tracker", "utility"},
    "utility": set(),
}
_PLUGIN_TYPES = {
    "frame_processor": ("frame_processors", FrameProcessorBase),
    "tracker": ("trackers", TrackerBase),
    "utility": ("utilities", UtilityBase),
}
_NODE_TYPES = {"camera", "frame_processor", "pipeline", "tracker", "utility"}
_CAMERA_NAME_RE = re.compile(r"^[^\x00-\x1f]{1,80}$")
_MODEL_PATH_KEYS = ("file_path", "source_pt", "path")
_LOCK = threading.RLock()
_RESTART_REQUIRED = False


def _finite(value, default=0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _clamp(value, low, high):
    return max(low, min(high, value))


def _deep(value):
    return copy.deepcopy(value)


def _defaults_from_schema(schema: dict, *, pipeline_name: str | None = None) -> dict:
    defaults = {}
    for key, defn in (schema or {}).items():
        if not isinstance(defn, dict) or "default" not in defn:
            continue
        value = _deep(defn["default"])
        # The model picker has no useful empty default.  Give new nodes the
        # same default model the rest of iSpy uses so the first pipeline node
        # is immediately usable.
        if (
            pipeline_name
            and key == "vision_model"
            and defn.get("type") == "model"
            and not value
        ):
            value = default_vision_model()
        defaults[key] = value
    return defaults


def _merge_defaults(schema: dict, settings: dict | None, *, pipeline_name=None) -> dict:
    out = _deep(settings) if isinstance(settings, dict) else {}
    for key, value in _defaults_from_schema(
        schema, pipeline_name=pipeline_name
    ).items():
        out.setdefault(key, _deep(value))
    return out


def _label_for(cls, name: str) -> str:
    return str(
        getattr(cls, "display_name", None)
        or getattr(cls, "label", None)
        or name
    )


def _doc_for(cls) -> str:
    return (getattr(cls, "__doc__", None) or "").strip()[:400]


def _plugin_catalog() -> dict[str, dict[str, dict]]:
    catalog = {}
    for ptype, (subdir, base_cls) in _PLUGIN_TYPES.items():
        try:
            classes = load_plugins(_PLUGIN_ROOT / subdir, base_cls)
        except Exception:
            logger.exception("Failed to discover %s plugins for editor", ptype)
            classes = {}
        entries = {}
        for name, cls in sorted(classes.items()):
            if ptype in {"tracker", "utility"} and name.lower() in {
                "rollback",
                "roll_back",
            }:
                continue
            try:
                schema = cls.config_schema() or {}
            except Exception:
                logger.exception("Failed to read %s schema for '%s'", ptype, name)
                schema = {}
            if not isinstance(schema, dict):
                schema = {}
            supported = getattr(cls, "supported_pipelines", None)
            entries[name] = {
                "name": name,
                "type": ptype,
                "label": _label_for(cls, name),
                "doc": _doc_for(cls),
                "schema": schema,
                "defaults": _defaults_from_schema(schema),
                "supported_pipelines": list(supported) if supported else [],
                "template": bool(getattr(cls, "template", False)),
                "needs_internet": bool(getattr(cls, "needs_internet", False)),
            }
        catalog[ptype] = entries
    return catalog


def _pipeline_catalog(config) -> dict[str, dict]:
    admin = config.get("admin") if config is not None else None
    entries = {}
    try:
        payloads = _build_vision_pipeline_payloads(admin=admin)
    except Exception:
        logger.exception("Failed to list pipelines for editor")
        payloads = []
    pipeline_classes = get_pipeline_classes()
    for payload in payloads:
        name = payload.get("name")
        if not name:
            continue
        schema = payload.get("config_schema") or {}
        if not isinstance(schema, dict):
            schema = {}
        item = _deep(payload)
        item["schema"] = schema
        item["type"] = "pipeline"
        item["label"] = item.get("label") or name
        item["defaults"] = _defaults_from_schema(schema, pipeline_name=name)
        cls = pipeline_classes.get(name)
        item["needs_internet"] = bool(getattr(cls, "needs_internet", False))
        entries[name] = item
    return entries


def _camera_catalog() -> dict[str, dict]:
    entries = {}
    for cam_type, cls in get_camera_classes().items():
        try:
            schema = cls.config_schema() or {}
        except Exception:
            logger.exception("Failed to read camera schema for '%s'", cam_type)
            schema = {}
        if not isinstance(schema, dict):
            schema = {}
        entries[cam_type] = {
            "name": cam_type,
            "type": "camera",
            "label": CAMERA_TYPE_LABELS.get(cam_type, cam_type),
            "schema": schema,
            "defaults": _defaults_from_schema(schema),
        }
    return entries


def _palette(config) -> dict:
    pipelines = _pipeline_catalog(config)
    cameras = _camera_catalog()
    catalog = _plugin_catalog()
    raw_plugins = (
        config.get("plugins", {}) if hasattr(config, "get") else {}
    )
    raw_plugins = raw_plugins if isinstance(raw_plugins, dict) else {}
    for ptype, bucket in (
        ("frame_processor", "frame_processors"),
        ("tracker", "trackers"),
        ("utility", "utilities"),
    ):
        current_bucket = raw_plugins.get(bucket)
        current_bucket = (
            current_bucket if isinstance(current_bucket, dict) else {}
        )
        for name, item in catalog.get(ptype, {}).items():
            current = current_bucket.get(name)
            if isinstance(current, dict):
                merged = _deep(item.get("defaults", {}))
                merged.update(_deep(current))
                item["defaults"] = merged
    return {
        "cameras": list(cameras.values()),
        "pipelines": list(pipelines.values()),
        "frame_processors": list(catalog["frame_processor"].values()),
        "trackers": list(catalog["tracker"].values()),
        "utilities": list(catalog["utility"].values()),
        "rows": ROW_LABELS,
        "grid": 20,
    }


def _camera_calibrated(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    calib = entry.get("calibration") or {}
    return bool(
        (calib.get("camera_matrix") and calib.get("dist_coeffs") is not None)
        or calib.get("focal_length_pixels")
        or _finite(calib.get("fov"), 0) > 0
    )


def _clean_layout(raw, fallback_nodes: dict | None = None) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    grid = int(_clamp(_finite(raw.get("grid"), 20), 8, 80))
    viewport = raw.get("viewport") if isinstance(raw.get("viewport"), dict) else {}
    layout = {
        "version": 1,
        "grid": grid,
        "viewport": {
            "x": _finite(viewport.get("x"), 0),
            "y": _finite(viewport.get("y"), 0),
            "zoom": _clamp(_finite(viewport.get("zoom"), 1), 0.25, 2.5),
        },
        "nodes": {},
    }
    raw_nodes = raw.get("nodes") if isinstance(raw.get("nodes"), dict) else {}
    source = fallback_nodes if fallback_nodes is not None else raw_nodes
    for node_id, pos in source.items():
        if not isinstance(node_id, str) or not isinstance(pos, dict):
            continue
        layout["nodes"][node_id] = {
            "x": _finite(pos.get("x"), 0),
            "y": _finite(pos.get("y"), 0),
        }
    return layout


def _layout_from_config(config) -> dict:
    return _clean_layout(config.get("editor_graph", {}) if config else {})


def _node_pos(layout: dict, node_id: str, default_x: float, row: str) -> tuple[float, float]:
    saved = (layout.get("nodes") or {}).get(node_id) or {}
    x = _finite(saved.get("x"), default_x)
    return x, float(ROW_Y[row])


def _completion(schema: dict, settings: dict) -> dict:
    return completion_for_schema(schema, settings)


def _make_node(
    *,
    node_id: str,
    node_type: str,
    label: str,
    camera: str | None,
    plugin: str | None,
    schema: dict,
    settings: dict,
    default_x: float,
    layout: dict,
    order: int | None = None,
    subtitle: str = "",
    doc: str = "",
    supported_pipelines=None,
    builtin: bool = False,
    warnings=None,
    extra=None,
) -> dict:
    x, y = _node_pos(layout, node_id, default_x, node_type)
    progress = _completion(schema, settings)
    node = {
        "id": node_id,
        "type": node_type,
        "category": node_type,
        "label": label,
        "subtitle": subtitle,
        "camera": camera,
        "plugin": plugin,
        "schema": schema,
        "settings": settings,
        "x": x,
        "y": y,
        "row": node_type,
        "order": order,
        "completion": progress,
        "required": progress["required"],
        "doc": doc,
        "supported_pipelines": list(supported_pipelines or []),
        "builtin": bool(builtin),
        "warnings": list(warnings or []),
    }
    if extra:
        node.update(extra)
    return node


def _pipeline_supported_ok(pipeline_name: str, supported: list[str] | None) -> bool:
    if not supported:
        return True
    return pipeline_name in set(supported)


def _edge_state(source: dict | None, target: dict | None) -> str:
    if source is None or target is None:
        return "invalid"
    if not source.get("completion", {}).get("complete"):
        return "dashed"
    if not target.get("completion", {}).get("complete"):
        return "half"
    return "complete"


def _make_edge(source: dict, target: dict, kind: str, warning: str | None = None) -> dict:
    return {
        "id": f"e:{source['id']}->{target['id']}",
        "from": source["id"],
        "to": target["id"],
        "kind": kind,
        "state": _edge_state(source, target),
        "warning": warning,
    }


def _build_edges(nodes: list[dict]) -> list[dict]:
    by_id = {node["id"]: node for node in nodes}
    edges = []
    seen = set()

    def add(source, target, kind="flow", warning=None):
        if source is None or target is None:
            return
        edge = _make_edge(source, target, kind, warning)
        if edge["id"] in seen:
            return
        seen.add(edge["id"])
        edges.append(edge)

    cameras = [node for node in nodes if node["type"] == "camera"]
    pipelines = [node for node in nodes if node["type"] == "pipeline"]
    trackers = [node for node in nodes if node["type"] == "tracker"]
    utilities = [node for node in nodes if node["type"] == "utility"]

    for camera in cameras:
        chain = [camera]
        chain.extend(
            sorted(
                [
                    node
                    for node in nodes
                    if node["type"] == "frame_processor"
                    and node.get("camera") == camera.get("camera")
                ],
                key=lambda node: node.get("order") or 0,
            )
        )
        pipeline = next(
            (
                node
                for node in pipelines
                if node.get("camera") == camera.get("camera")
            ),
            None,
        )
        if pipeline is not None:
            chain.append(pipeline)
        for index in range(len(chain) - 1):
            add(chain[index], chain[index + 1])

    active_pipelines = {
        node.get("plugin") for node in pipelines if node.get("plugin")
    }
    if trackers:
        for pipeline in pipelines:
            add(
                pipeline,
                trackers[0],
                warning=(
                    None
                    if _pipeline_supported_ok(
                        pipeline.get("plugin"), trackers[0].get("supported_pipelines")
                    )
                    else f"Tracker '{trackers[0]['label']}' does not list "
                    f"pipeline '{pipeline.get('plugin')}' as supported."
                ),
            )
        for index in range(len(trackers) - 1):
            target = trackers[index + 1]
            bad = sorted(
                name
                for name in active_pipelines
                if not _pipeline_supported_ok(
                    name, target.get("supported_pipelines")
                )
            )
            add(
                trackers[index],
                target,
                kind="order",
                warning=(
                    f"Tracker '{target['label']}' does not list "
                    f"pipeline(s) {', '.join(bad)} as supported."
                    if bad
                    else None
                ),
            )
        utility_source = trackers[-1]
    else:
        utility_source = None

    if utilities:
        sources = [utility_source] if utility_source is not None else pipelines
        for source in sources:
            for utility in utilities:
                bad = sorted(
                    name
                    for name in active_pipelines
                    if not _pipeline_supported_ok(
                        name, utility.get("supported_pipelines")
                    )
                )
                add(
                    source,
                    utility,
                    warning=(
                        f"Utility '{utility['label']}' does not list "
                        f"pipeline(s) {', '.join(bad)} as supported."
                        if bad
                        else None
                    ),
                )
    return edges


def build_graph(config) -> dict:
    """Build the canonical graph from executable config, never from layout."""
    raw_cams = config.get("camera_configs", {}) if config is not None else {}
    raw_plugins = config.get("plugins", {}) if config is not None else {}
    raw_cams = raw_cams if isinstance(raw_cams, dict) else {}
    raw_plugins = raw_plugins if isinstance(raw_plugins, dict) else {}
    layout = _layout_from_config(config)
    pipelines_cat = _pipeline_catalog(config)
    cameras_cat = _camera_catalog()
    addon_cat = _plugin_catalog()

    nodes = []
    camera_rows = []
    for index, (cam_name, raw_entry) in enumerate(raw_cams.items()):
        if not isinstance(raw_entry, dict):
            continue
        entry = _deep(raw_entry)
        normalize_camera_entry(entry)
        cam_type = str(entry.get("camera_type") or "opencv")
        cam_schema = (cameras_cat.get(cam_type) or {}).get("schema", {})
        cam_settings = {
            key: _deep(entry[key])
            for key in cam_schema
            if key in entry
        }
        cam_settings = _merge_defaults(cam_schema, cam_settings)
        cam_settings["camera_type"] = cam_type
        camera_node = _make_node(
            node_id=f"camera:{cam_name}",
            node_type="camera",
            label=cam_name,
            camera=cam_name,
            plugin=None,
            schema=cam_schema,
            settings=cam_settings,
            default_x=80 + index * 420,
            layout=layout,
            subtitle=f"{CAMERA_TYPE_LABELS.get(cam_type, cam_type)} · {entry.get('source', '')}",
            extra={
                "camera_type": cam_type,
                "calibrated": _camera_calibrated(entry),
            },
        )
        nodes.append(camera_node)
        camera_rows.append((cam_name, entry, camera_node))

        enabled_fps = raw_plugins.get("frame_processors")
        enabled_fps = enabled_fps if isinstance(enabled_fps, dict) else {}
        assigned = entry.get("frame_processors")
        if isinstance(assigned, list):
            fp_names = [name for name in assigned if isinstance(name, str)]
        else:
            fp_names = list(enabled_fps)
        for order, fp_name in enumerate(fp_names):
            fp_cat = addon_cat["frame_processor"].get(fp_name)
            schema = (fp_cat or {}).get("schema", {})
            stored = (
                enabled_fps.get(fp_name)
                if isinstance(enabled_fps.get(fp_name), dict)
                else {}
            )
            settings = _merge_defaults(schema, stored)
            nodes.append(
                _make_node(
                    node_id=f"frame_processor:{cam_name}:{fp_name}",
                    node_type="frame_processor",
                    label=(fp_cat or {}).get("label", fp_name),
                    camera=cam_name,
                    plugin=fp_name,
                    schema=schema,
                    settings=settings,
                    default_x=400 + index * 420 + order * 240,
                    layout=layout,
                    order=order,
                    subtitle=(fp_cat or {}).get("doc", "")[:90],
                    doc=(fp_cat or {}).get("doc", ""),
                    supported_pipelines=(fp_cat or {}).get("supported_pipelines"),
                    builtin=(fp_cat or {}).get("template", False),
                    warnings=(
                        [f"Frame processor '{fp_name}' was not found."]
                        if fp_cat is None
                        else []
                    ),
                    extra={
                        "needs_internet": bool(
                            (fp_cat or {}).get("needs_internet", False)
                        )
                    },
                )
            )

        pipeline_name = get_pipeline_name(entry)
        pipeline_cat = pipelines_cat.get(pipeline_name)
        pipeline_schema = (pipeline_cat or {}).get("schema", {})
        pipeline_settings = _merge_defaults(
            pipeline_schema,
            get_pipeline_settings(entry),
            pipeline_name=pipeline_name,
        )
        requires_calibration = bool(
            (pipeline_cat or {}).get("requires_calibration", False)
        )
        calibration_warning = None
        if requires_calibration and not _camera_calibrated(entry):
            calibration_warning = (
                f"Camera '{cam_name}' has no usable calibration for this "
                "pipeline. Calibrate it on the Cameras page."
            )
        nodes.append(
            _make_node(
                node_id=f"pipeline:{cam_name}",
                node_type="pipeline",
                label=(pipeline_cat or {}).get("label", pipeline_name),
                camera=cam_name,
                plugin=pipeline_name,
                schema=pipeline_schema,
                settings=pipeline_settings,
                default_x=820 + index * 420,
                layout=layout,
                subtitle=pipeline_name,
                doc=(pipeline_cat or {}).get("class_name", ""),
                supported_pipelines=None,
                warnings=(
                    [f"Unknown pipeline '{pipeline_name}'."]
                    if pipeline_cat is None
                    else ([calibration_warning] if calibration_warning else [])
                ),
                extra={
                    "calibration_sections": list(
                        (pipeline_cat or {}).get("calibration_sections") or []
                    ),
                    "requires_calibration": requires_calibration,
                    "calibrated": _camera_calibrated(entry),
                    "beta": bool((pipeline_cat or {}).get("beta", False)),
                    "show_common_fields": bool(
                        (pipeline_cat or {}).get("show_common_fields", True)
                    ),
                    "needs_internet": bool(
                        (pipeline_cat or {}).get("needs_internet", False)
                    ),
                },
            )
        )

    tracker_raw = raw_plugins.get("trackers")
    tracker_raw = tracker_raw if isinstance(tracker_raw, dict) else {}
    for order, (name, stored) in enumerate(tracker_raw.items()):
        if not isinstance(name, str) or name.lower() in {"rollback", "roll_back"}:
            continue
        cat = addon_cat["tracker"].get(name)
        schema = (cat or {}).get("schema", {})
        settings = _merge_defaults(
            schema, stored if isinstance(stored, dict) else {}
        )
        nodes.append(
            _make_node(
                node_id=f"tracker:{name}",
                node_type="tracker",
                label=(cat or {}).get("label", name),
                camera=None,
                plugin=name,
                schema=schema,
                settings=settings,
                default_x=80 + order * 280,
                layout=layout,
                order=order,
                subtitle=(cat or {}).get("doc", "")[:90],
                doc=(cat or {}).get("doc", ""),
                supported_pipelines=(cat or {}).get("supported_pipelines"),
                builtin=(cat or {}).get("template", False),
                warnings=([f"Tracker '{name}' was not found."] if cat is None else []),
                extra={
                    "needs_internet": bool(
                        (cat or {}).get("needs_internet", False)
                    )
                },
            )
        )

    utility_raw = raw_plugins.get("utilities")
    utility_raw = utility_raw if isinstance(utility_raw, dict) else {}
    for order, (name, stored) in enumerate(utility_raw.items()):
        if not isinstance(name, str) or name.lower() in {"rollback", "roll_back"}:
            continue
        cat = addon_cat["utility"].get(name)
        schema = (cat or {}).get("schema", {})
        settings = _merge_defaults(
            schema, stored if isinstance(stored, dict) else {}
        )
        nodes.append(
            _make_node(
                node_id=f"utility:{name}",
                node_type="utility",
                label=(cat or {}).get("label", name),
                camera=None,
                plugin=name,
                schema=schema,
                settings=settings,
                default_x=80 + order * 280,
                layout=layout,
                order=order,
                subtitle=(cat or {}).get("doc", "")[:90],
                doc=(cat or {}).get("doc", ""),
                supported_pipelines=(cat or {}).get("supported_pipelines"),
                builtin=(cat or {}).get("template", False),
                warnings=([f"Utility '{name}' was not found."] if cat is None else []),
                extra={
                    "needs_internet": bool(
                        (cat or {}).get("needs_internet", False)
                    )
                },
            )
        )

    edges = _build_edges(nodes)
    saved_layout = (
        config.get("editor_graph", {}) if hasattr(config, "get") else {}
    )
    return {
        "version": 1,
        "nodes": nodes,
        "edges": edges,
        "layout": {
            "grid": layout["grid"],
            "viewport": layout["viewport"],
        },
        "layout_saved": bool(
            isinstance(saved_layout, dict) and saved_layout
        ),
        "rows": ROW_LABELS,
    }


def _coerce_settings(
    schema: dict,
    provided: dict | None,
    existing: dict | None = None,
) -> dict:
    out = _deep(existing) if isinstance(existing, dict) else {}
    provided = provided if isinstance(provided, dict) else {}
    for key, defn in (schema or {}).items():
        if key not in provided or not isinstance(defn, dict):
            continue
        out[key] = _coerce_setting_value(provided[key], defn)
    return out


def _normalize_model(settings: dict) -> None:
    value = settings.get("vision_model")
    if isinstance(value, str) and value.strip():
        settings["vision_model"] = {
            "file_path": value.strip(),
            "source_pt": value.strip(),
        }
    elif isinstance(value, dict):
        path = value.get("file_path") or value.get("source_pt") or value.get("path")
        if isinstance(path, str) and path.strip():
            value.setdefault("file_path", path.strip())
            value.setdefault("source_pt", path.strip())


def _model_paths(settings: dict) -> list[str]:
    value = settings.get("vision_model")
    if not isinstance(value, dict):
        return []
    paths = []
    for key in _MODEL_PATH_KEYS:
        path = value.get(key)
        if isinstance(path, str) and path.strip():
            paths.append(path.strip())
    return paths


def _edge_allowed(source: dict, target: dict) -> bool:
    return target["type"] in _EDGE_TARGETS.get(source["type"], set())


def _validate_edges(edges, nodes_by_id: dict, active_pipelines: set) -> list[dict]:
    errors = []
    seen = set()
    adjacency = {node_id: [] for node_id in nodes_by_id}
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(
                {"code": "edge_invalid", "message": f"Connection {index + 1} is invalid."}
            )
            continue
        source_id = edge.get("from")
        target_id = edge.get("to")
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)
        if source is None or target is None:
            errors.append(
                {
                    "code": "edge_endpoint",
                    "message": "A connection points at a node that is not in the graph.",
                    "edge": {"from": source_id, "to": target_id},
                }
            )
            continue
        key = (source_id, target_id)
        if key in seen:
            errors.append(
                {
                    "code": "edge_duplicate",
                    "message": f"Duplicate connection {source['label']} → {target['label']}.",
                    "edge": {"from": source_id, "to": target_id},
                }
            )
            continue
        seen.add(key)
        if not _edge_allowed(source, target):
            errors.append(
                {
                    "code": "edge_type",
                    "message": (
                        f"Can't connect {source['type'].replace('_', ' ')} "
                        f"'{source['label']}' to {target['type'].replace('_', ' ')} "
                        f"'{target['label']}'. Only camera → frame processor/pipeline, "
                        "frame processor → frame processor/pipeline, pipeline → "
                        "tracker/utility, and tracker → tracker/utility are valid."
                    ),
                    "edge": {"from": source_id, "to": target_id},
                }
            )
            continue
        if source["type"] in ("camera", "frame_processor") and target["type"] in (
            "frame_processor",
            "pipeline",
        ):
            if source.get("camera") != target.get("camera"):
                errors.append(
                    {
                        "code": "edge_camera",
                        "message": (
                            f"'{source['label']}' and '{target['label']}' belong to "
                            "different cameras. A frame-processor chain cannot cross cameras."
                        ),
                        "edge": {"from": source_id, "to": target_id},
                    }
                )
                continue
        if source["type"] == "frame_processor" and target["type"] == "pipeline":
            supported = source.get("supported_pipelines") or []
            if supported and target.get("plugin") not in set(supported):
                errors.append(
                    {
                        "code": "edge_pipeline_unsupported",
                        "message": (
                            f"Frame processor '{source['label']}' does not support "
                            f"pipeline '{target.get('plugin')}'."
                        ),
                        "edge": {"from": source_id, "to": target_id},
                    }
                )
                continue
        if target["type"] in ("tracker", "utility"):
            supported = target.get("supported_pipelines") or []
            if supported:
                names = (
                    {source.get("plugin")}
                    if source["type"] == "pipeline"
                    else active_pipelines
                )
                bad = sorted(
                    name for name in names if name and name not in set(supported)
                )
                if bad:
                    errors.append(
                        {
                            "code": "edge_pipeline_unsupported",
                            "message": (
                                f"{target['type'].title()} '{target['label']}' does not "
                                f"support pipeline(s) {', '.join(bad)}."
                            ),
                            "edge": {"from": source_id, "to": target_id},
                        }
                    )
                    continue
        adjacency.setdefault(source_id, []).append(target_id)

    # A cycle would make the visual flow impossible to execute as drawn.
    state = {}

    def visit(node_id, trail):
        state[node_id] = 1
        for nxt in adjacency.get(node_id, []):
            if state.get(nxt) == 1:
                cycle = " → ".join(trail + [nxt])
                errors.append(
                    {
                        "code": "edge_cycle",
                        "message": f"Connections form a cycle: {cycle}.",
                    }
                )
                return True
            if state.get(nxt, 0) == 0 and visit(nxt, trail + [nxt]):
                return True
        state[node_id] = 2
        return False

    for node_id in list(adjacency):
        if state.get(node_id, 0) == 0 and visit(node_id, [node_id]):
            break
    return errors


def _normalize_graph(payload: dict, config) -> tuple[dict, list[dict], list[dict]]:
    """Validate and turn a graph payload into executable config structures."""
    errors = []
    warnings = []
    payload = payload if isinstance(payload, dict) else {}
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list):
        return {}, [{"code": "nodes_invalid", "message": "nodes must be a list."}], []

    cameras_cat = _camera_catalog()
    pipelines_cat = _pipeline_catalog(config)
    addon_cat = _plugin_catalog()
    existing_cams = config.get("camera_configs", {}) if config else {}
    existing_cams = existing_cams if isinstance(existing_cams, dict) else {}
    existing_plugins = config.get("plugins", {}) if config else {}
    existing_plugins = existing_plugins if isinstance(existing_plugins, dict) else {}
    existing_fps = existing_plugins.get("frame_processors")
    existing_fps = existing_fps if isinstance(existing_fps, dict) else {}

    declared_cameras = set()
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("type") or raw.get("category") or "") == "camera":
            name = str(raw.get("camera") or raw.get("name") or "").strip()
            if name:
                declared_cameras.add(name)

    nodes = []
    nodes_by_id = {}
    camera_nodes = {}
    pipeline_nodes = {}
    fp_by_camera = {}
    fp_plugins = {}
    tracker_nodes = []
    utility_nodes = []

    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            errors.append(
                {"code": "node_invalid", "message": f"Node {index + 1} is not an object."}
            )
            continue
        node_type = str(raw.get("type") or raw.get("category") or "")
        if node_type not in _NODE_TYPES:
            errors.append(
                {
                    "code": "node_type",
                    "message": f"Node {index + 1} has unknown type '{node_type}'.",
                }
            )
            continue
        node_id = str(raw.get("id") or "")
        if not node_id:
            errors.append(
                {"code": "node_id", "message": f"Node {index + 1} has no id."}
            )
            continue
        if node_id in nodes_by_id:
            errors.append(
                {
                    "code": "node_duplicate",
                    "message": f"Duplicate node id '{node_id}'.",
                    "node": node_id,
                }
            )
            continue

        node = {
            "id": node_id,
            "type": node_type,
            "label": str(raw.get("label") or node_type),
            "camera": raw.get("camera"),
            "plugin": raw.get("plugin"),
            "settings": raw.get("settings")
            if isinstance(raw.get("settings"), dict)
            else {},
            "x": raw.get("x"),
            "y": raw.get("y"),
            "order": (
                raw.get("order")
                if isinstance(raw.get("order"), int)
                else index
            ),
        }
        if node_type == "camera":
            name = str(raw.get("camera") or raw.get("name") or "").strip()
            if not name or not _CAMERA_NAME_RE.match(name):
                errors.append(
                    {
                        "code": "camera_name",
                        "message": f"Camera node '{node_id}' needs a valid name.",
                        "node": node_id,
                    }
                )
                continue
            cam_type = str(
                raw.get("camera_type")
                or node["settings"].get("camera_type")
                or (existing_cams.get(name) or {}).get("camera_type")
                or "opencv"
            )
            cam_cat = cameras_cat.get(cam_type)
            if cam_cat is None:
                errors.append(
                    {
                        "code": "camera_type",
                        "message": f"Camera '{name}' has unknown type '{cam_type}'.",
                        "node": node_id,
                    }
                )
                continue
            if name in camera_nodes:
                errors.append(
                    {
                        "code": "camera_duplicate",
                        "message": f"Camera '{name}' appears more than once.",
                        "node": node_id,
                    }
                )
                continue
            existing = _deep(existing_cams.get(name) or {})
            settings = _coerce_settings(
                cam_cat["schema"],
                node["settings"],
                {
                    key: existing[key]
                    for key in cam_cat["schema"]
                    if key in existing
                },
            )
            settings = _merge_defaults(cam_cat["schema"], settings)
            settings["camera_type"] = cam_type
            node["camera"] = name
            node["camera_type"] = cam_type
            node["schema"] = cam_cat["schema"]
            node["settings"] = settings
            progress = _completion(cam_cat["schema"], settings)
            if not progress["complete"]:
                errors.append(
                    {
                        "code": "settings_incomplete",
                        "message": (
                            f"Camera '{name}' is missing required field(s): "
                            f"{', '.join(progress['missing_labels'])}."
                        ),
                        "node": node_id,
                        "missing": progress["missing"],
                    }
                )
            camera_nodes[name] = node
        elif node_type in ("frame_processor", "pipeline"):
            camera = str(raw.get("camera") or "").strip()
            plugin = str(raw.get("plugin") or "").strip()
            if (
                not camera
                or camera not in existing_cams
                and camera not in declared_cameras
            ):
                errors.append(
                    {
                        "code": "camera_missing",
                        "message": f"{node_type.replace('_', ' ').title()} '{node_id}' references unknown camera '{camera}'.",
                        "node": node_id,
                    }
                )
                continue
            node["camera"] = camera
            if node_type == "pipeline":
                cat = pipelines_cat.get(plugin)
                if cat is None:
                    errors.append(
                        {
                            "code": "plugin_unknown",
                            "message": f"Unknown pipeline '{plugin}'.",
                            "node": node_id,
                        }
                    )
                    continue
                if camera in pipeline_nodes:
                    errors.append(
                        {
                            "code": "pipeline_duplicate",
                            "message": f"Camera '{camera}' already has a pipeline.",
                            "node": node_id,
                        }
                    )
                    continue
                existing_pipeline = (
                    existing_cams.get(camera) or {}
                )
                existing_name = get_pipeline_name(existing_pipeline)
                existing_settings = (
                    get_pipeline_settings(existing_pipeline)
                    if existing_name == plugin
                    else {}
                )
                schema = cat["schema"]
                settings = _coerce_settings(
                    schema, node["settings"], existing_settings
                )
                settings = _merge_defaults(
                    schema, settings, pipeline_name=plugin
                )
                _normalize_model(settings)
                node["plugin"] = plugin
                node["schema"] = schema
                node["settings"] = settings
                progress = _completion(schema, settings)
                if not progress["complete"]:
                    errors.append(
                        {
                            "code": "settings_incomplete",
                            "message": (
                                f"Pipeline '{cat.get('label', plugin)}' on camera "
                                f"'{camera}' is missing required field(s): "
                                f"{', '.join(progress['missing_labels'])}."
                            ),
                            "node": node_id,
                            "missing": progress["missing"],
                        }
                    )
                for path in _model_paths(settings):
                    try:
                        if not Path(path).expanduser().exists():
                            warnings.append(
                                {
                                    "code": "model_path_missing",
                                    "message": (
                                        f"Pipeline '{plugin}' references model '{path}', "
                                        "which does not exist on this machine yet."
                                    ),
                                    "node": node_id,
                                }
                            )
                    except OSError:
                        pass
                pipeline_nodes[camera] = node
            else:
                cat = addon_cat["frame_processor"].get(plugin)
                if cat is None:
                    errors.append(
                        {
                            "code": "plugin_unknown",
                            "message": f"Unknown frame processor '{plugin}'.",
                            "node": node_id,
                        }
                    )
                    continue
                schema = cat["schema"]
                existing = (
                    existing_fps.get(plugin)
                    if isinstance(existing_fps.get(plugin), dict)
                    else {}
                )
                settings = _coerce_settings(schema, node["settings"], existing)
                settings = _merge_defaults(schema, settings)
                node["plugin"] = plugin
                node["schema"] = schema
                node["settings"] = settings
                node["supported_pipelines"] = list(
                    cat.get("supported_pipelines") or []
                )
                node["needs_internet"] = bool(cat.get("needs_internet"))
                progress = _completion(schema, settings)
                if not progress["complete"]:
                    errors.append(
                        {
                            "code": "settings_incomplete",
                            "message": (
                                f"Frame processor '{cat.get('label', plugin)}' is missing "
                                f"required field(s): {', '.join(progress['missing_labels'])}."
                            ),
                            "node": node_id,
                            "missing": progress["missing"],
                        }
                    )
                if any(
                    previous["plugin"] == plugin
                    for previous in fp_by_camera.get(camera, [])
                ):
                    errors.append(
                        {
                            "code": "frame_processor_duplicate",
                            "message": (
                                f"Frame processor '{plugin}' is assigned to camera "
                                f"'{camera}' more than once."
                            ),
                            "node": node_id,
                        }
                    )
                fp_by_camera.setdefault(camera, []).append(node)
                if plugin in fp_plugins:
                    if fp_plugins[plugin]["settings"] != settings:
                        errors.append(
                            {
                                "code": "frame_processor_shared_settings",
                                "message": (
                                    f"Frame processor '{plugin}' is shared by multiple "
                                    "cameras and must use the same settings."
                                ),
                                "node": node_id,
                            }
                        )
                else:
                    fp_plugins[plugin] = node
        elif node_type in ("tracker", "utility"):
            plugin = str(raw.get("plugin") or "").strip()
            if plugin.lower() in {"rollback", "roll_back"}:
                errors.append(
                    {
                        "code": "rollback_node",
                        "message": "RollBack is a core service and can never be a node.",
                        "node": node_id,
                    }
                )
                continue
            cat = addon_cat[node_type].get(plugin)
            if cat is None:
                errors.append(
                    {
                        "code": "plugin_unknown",
                        "message": f"Unknown {node_type} '{plugin}'.",
                        "node": node_id,
                    }
                )
                continue
            schema = cat["schema"]
            existing_group = existing_plugins.get(
                "trackers" if node_type == "tracker" else "utilities"
            )
            existing_group = (
                existing_group if isinstance(existing_group, dict) else {}
            )
            existing = (
                existing_group.get(plugin)
                if isinstance(existing_group.get(plugin), dict)
                else {}
            )
            settings = _coerce_settings(schema, node["settings"], existing)
            settings = _merge_defaults(schema, settings)
            node["plugin"] = plugin
            node["schema"] = schema
            node["settings"] = settings
            node["supported_pipelines"] = list(
                cat.get("supported_pipelines") or []
            )
            node["needs_internet"] = bool(cat.get("needs_internet"))
            progress = _completion(schema, settings)
            if not progress["complete"]:
                errors.append(
                    {
                        "code": "settings_incomplete",
                        "message": (
                            f"{node_type.title()} '{cat.get('label', plugin)}' is missing "
                            f"required field(s): {', '.join(progress['missing_labels'])}."
                        ),
                        "node": node_id,
                        "missing": progress["missing"],
                    }
                )
            group = tracker_nodes if node_type == "tracker" else utility_nodes
            if any(previous["plugin"] == plugin for previous in group):
                errors.append(
                    {
                        "code": "node_duplicate",
                        "message": f"{node_type.title()} '{plugin}' appears more than once.",
                        "node": node_id,
                    }
                )
                continue
            group.append(node)
        nodes.append(node)
        nodes_by_id[node_id] = node

    if not camera_nodes:
        errors.append(
            {"code": "no_cameras", "message": "Add at least one camera to the graph."}
        )
    for camera in camera_nodes:
        if camera not in pipeline_nodes:
            errors.append(
                {
                    "code": "pipeline_missing",
                    "message": f"Camera '{camera}' needs exactly one pipeline.",
                    "node": f"camera:{camera}",
                }
            )
    for camera, fp_nodes in fp_by_camera.items():
        if camera not in camera_nodes:
            errors.append(
                {
                    "code": "camera_missing",
                    "message": f"Frame processors reference unknown camera '{camera}'.",
                }
            )
        orders = [fp.get("order", 0) for fp in fp_nodes]
        if len(set(orders)) != len(orders):
            errors.append(
                {
                    "code": "frame_processor_order",
                    "message": f"Camera '{camera}' has duplicate frame-processor order values.",
                }
            )

    edges = payload.get("edges")
    if edges is None:
        edges = []
        for camera in camera_nodes.values():
            chain = [camera]
            chain.extend(fp_by_camera.get(camera["camera"], []))
            if camera["camera"] in pipeline_nodes:
                chain.append(pipeline_nodes[camera["camera"]])
            for index in range(len(chain) - 1):
                edges.append(
                    {
                        "from": chain[index]["id"],
                        "to": chain[index + 1]["id"],
                    }
                )
        active_pipelines = {
            node["plugin"] for node in pipeline_nodes.values() if node.get("plugin")
        }
        if tracker_nodes:
            for pipeline in pipeline_nodes.values():
                edges.append(
                    {"from": pipeline["id"], "to": tracker_nodes[0]["id"]}
                )
            for index in range(len(tracker_nodes) - 1):
                edges.append(
                    {
                        "from": tracker_nodes[index]["id"],
                        "to": tracker_nodes[index + 1]["id"],
                    }
                )
            source_nodes = [tracker_nodes[-1]]
        else:
            source_nodes = list(pipeline_nodes.values())
        for source in source_nodes:
            for utility in utility_nodes:
                edges.append({"from": source["id"], "to": utility["id"]})
    else:
        active_pipelines = {
            node["plugin"] for node in pipeline_nodes.values() if node.get("plugin")
        }

    if not isinstance(edges, list):
        errors.append(
            {"code": "edges_invalid", "message": "edges must be a list."}
        )
        edges = []
    errors.extend(_validate_edges(edges, nodes_by_id, active_pipelines))

    # Build executable structures.  Camera/pipeline/frame-processor nodes are
    # keyed by camera; trackers/utilities are global and keep node order.
    new_cams = {}
    for camera, node in camera_nodes.items():
        existing = _deep(existing_cams.get(camera) or {})
        if not existing:
            existing = {
                "name": camera,
                "source": node["settings"].get("source", 0),
                "camera_type": node.get("camera_type", "opencv"),
                "calibration": {
                    "distance": 0,
                    "game_piece_size": 0,
                    "size": 0,
                    "fov": 0,
                },
            }
        for key in _CAMERA_CORE_KEYS:
            if key in node["settings"] and key not in ("name", "frame_processors"):
                existing[key] = _deep(node["settings"][key])
        existing["name"] = camera
        existing["camera_type"] = node.get("camera_type", "opencv")
        existing.pop("pipeline", None)
        new_cams[camera] = existing

    for camera, node in pipeline_nodes.items():
        entry = new_cams.setdefault(camera, {})
        entry["pipeline"] = {
            "name": node["plugin"],
            "settings": _deep(node["settings"]),
        }

    # The editor graph is authoritative for camera assignment.  Writing an
    # explicit list (including an empty one) is what makes "remove every frame
    # processor from this camera" survive a reload; leaving the key absent
    # would expand every global frame processor again.
    for camera in camera_nodes:
        entry = new_cams.setdefault(camera, {})
        fp_nodes = fp_by_camera.get(camera, [])
        entry["frame_processors"] = [
            fp["plugin"]
            for fp in sorted(fp_nodes, key=lambda item: item.get("order", 0))
        ]

    new_plugins = _deep(existing_plugins)
    if not isinstance(new_plugins, dict):
        new_plugins = {}
    # A frame processor can exist in plugins.frame_processors without being
    # assigned to a camera.  The palette and Add-ons page still need its saved
    # settings, so merge the graph's assignments over the existing bucket
    # rather than replacing it.
    new_fps = new_plugins.get("frame_processors")
    new_fps = _deep(new_fps) if isinstance(new_fps, dict) else {}
    for plugin, node in fp_plugins.items():
        new_fps[plugin] = _deep(node["settings"])
    new_plugins["frame_processors"] = new_fps
    new_plugins["trackers"] = {
        node["plugin"]: _deep(node["settings"])
        for node in sorted(tracker_nodes, key=lambda item: item.get("order", 0))
    }
    new_plugins["utilities"] = {
        node["plugin"]: _deep(node["settings"])
        for node in sorted(utility_nodes, key=lambda item: item.get("order", 0))
    }
    # RollBack is a core service, not a graph node.  If an older config still
    # carries a plugin entry under either legacy name, keep it untouched.
    for bucket_name, target in (
        ("trackers", new_plugins["trackers"]),
        ("utilities", new_plugins["utilities"]),
    ):
        existing_bucket = existing_plugins.get(bucket_name)
        if not isinstance(existing_bucket, dict):
            continue
        for name, settings in existing_bucket.items():
            if str(name).lower() in {"rollback", "roll_back"}:
                target.setdefault(name, _deep(settings))

    for camera, entry in new_cams.items():
        normalize_camera_entry(entry)
        # The graph already validated required fields.  Normalize model aliases
        # but never silently invent a model the user did not choose.
        _normalize_model(get_pipeline_settings(entry))

    # Layout is editor-only metadata.  Keep only known node ids and fixed rows.
    raw_layout = payload.get("layout") if isinstance(payload.get("layout"), dict) else {}
    layout = _clean_layout(raw_layout)
    layout["nodes"] = {}
    for node in nodes:
        raw_pos = (
            (raw_layout.get("nodes") or {}).get(node["id"])
            if isinstance(raw_layout.get("nodes"), dict)
            else None
        )
        if not isinstance(raw_pos, dict):
            raw_pos = {"x": node.get("x", 0), "y": node.get("y", 0)}
        layout["nodes"][node["id"]] = {
            "x": _clamp(_finite(raw_pos.get("x"), 0), -5000, 30000),
            "y": float(ROW_Y[node["type"]]),
        }

    return (
        {
            "cameras": new_cams,
            "plugins": new_plugins,
            "layout": layout,
            "nodes": nodes,
            "edges": edges,
        },
        errors,
        warnings,
    )


def _validation_payload(payload: dict, config) -> tuple[dict, int]:
    normalized, errors, warnings = _normalize_graph(payload, config)
    completion = {
        node["id"]: _completion(node.get("schema", {}), node.get("settings", {}))
        for node in normalized.get("nodes", [])
    }
    return (
        {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "completion": completion,
            "restart_required": _RESTART_REQUIRED,
        },
        200 if not errors else 400,
    )


class EditorModule(WebModule):
    plugin_name = "editor"

    def register_routes(self, flask_app):
        flask_app.add_url_rule(
            "/editor", "editor_page", lambda: render_template("editor.html")
        )
        flask_app.add_url_rule(
            "/api/editor/graph", "api_editor_graph_get", self._get_graph
        )
        flask_app.add_url_rule(
            "/api/editor/graph",
            "api_editor_graph_save",
            require_local_or_token(self._save_graph),
            methods=["POST"],
        )
        flask_app.add_url_rule(
            "/api/editor/validate",
            "api_editor_validate",
            require_local_or_token(self._validate),
            methods=["POST"],
        )

    def _config(self):
        return self.context.get("config")

    def _get_graph(self):
        config = self._config()
        if config is None:
            return jsonify(error="No config available"), 500
        with _LOCK:
            graph = build_graph(config)
        return jsonify(
            ok=True,
            graph=graph,
            palette=_palette(config),
            restart_required=_RESTART_REQUIRED,
            admin=bool(config.get("admin")),
            config_file=str(getattr(config, "file_path", "") or ""),
        )

    def _validate(self):
        config = self._config()
        if config is None:
            return jsonify(error="No config available"), 500
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(ok=False, errors=[{"code": "payload", "message": "JSON object required."}]), 400
        result, status = _validation_payload(payload, config)
        return jsonify(result), status

    def _save_graph(self):
        global _RESTART_REQUIRED
        config = self._config()
        if config is None:
            return jsonify(error="No config available"), 500
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(ok=False, errors=[{"code": "payload", "message": "JSON object required."}]), 400
        with _LOCK:
            normalized, errors, warnings = _normalize_graph(payload, config)
            if errors:
                return (
                    jsonify(
                        ok=False,
                        errors=errors,
                        warnings=warnings,
                        restart_required=_RESTART_REQUIRED,
                    ),
                    400,
                )
            config.set("camera_configs", normalized["cameras"])
            config.set("plugins", normalized["plugins"])
            config.set("editor_graph", normalized["layout"])
            try:
                config._rebuild_camera_configs()
            except Exception:
                logger.exception("Failed to rebuild camera config wrappers")
            config.save()
            _RESTART_REQUIRED = True
            graph = build_graph(config)
        return jsonify(
            ok=True,
            errors=[],
            warnings=warnings,
            graph=graph,
            restart_required=True,
            note="Restart iSpy to apply camera, pipeline, tracker, utility, or frame-processor changes.",
        )
