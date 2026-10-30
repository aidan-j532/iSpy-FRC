"""Lane-based node editor backend.

The editor is deliberately additive: it reads and writes the existing
``camera_configs`` / ``plugins`` structures, and only keeps lane/viewport
metadata in the namespaced ``editor_graph`` block.  Runtime objects are never
touched, so every save reports that a restart is required.

The graph is not free-form.  A camera owns one horizontal lane and the columns
are always Camera -> Frame Processor -> Pipeline -> Tracker -> Utility.  The
backend is authoritative for that model; the browser only renders the returned
columns, lanes and edges.
"""

from __future__ import annotations

import copy
import logging
import math
import re
import threading
import time
from functools import wraps
from pathlib import Path

from flask import jsonify, render_template, request

from iSpy.config.iSpyConfig import (
    _CAMERA_CORE_KEYS,
    default_vision_model,
    get_pipeline_name,
    get_pipeline_settings,
    is_model_backed_pipeline,
    normalize_camera_entry,
)
from iSpy.config.schema_fields import completion_for_schema
from iSpy.core.request_auth import remote_denied_message, restart_allowed
from iSpy.plugins._loader import load_plugins
from iSpy.plugins.bases import FrameProcessorBase, TrackerBase, UtilityBase
from iSpy.supervisor.config import allow_remote_restart
from iSpy.vision.Cameras import CAMERA_TYPE_LABELS, get_camera_classes
from iSpy.vision.Cameras.OpenCVCamera import OpenCVCamera
from iSpy.vision.pipelines import get_pipeline_classes
from iSpy.web.Backend.PluginStatus import (
    _PLUGIN_ROOT,
    _build_vision_pipeline_payloads,
    _coerce_setting_value,
)
from iSpy.web.Backend.WebModule import WebModule

logger = logging.getLogger(__name__)

COLUMNS = ("camera", "frame_processor", "pipeline", "tracker", "utility")
COLUMN_INDEX = {name: index for index, name in enumerate(COLUMNS)}
COLUMN_LABELS = {
    "camera": "Camera",
    "frame_processor": "Frame Processor",
    "pipeline": "Pipeline",
    "tracker": "Tracker",
    "utility": "Utility",
}
NODE_W = 224
NODE_H = 104
COLUMN_X = {
    "camera": 32,
    "frame_processor": 316,
    "pipeline": 600,
    "tracker": 884,
    "utility": 1168,
}
LANE_TOP = 104
MIN_LANE_HEIGHT = 144
LANE_SLOT_GAP = 8
UTILITY_GAP = 132
UTILITY_TOP = 104

_PLUGIN_TYPES = {
    "frame_processor": ("frame_processors", FrameProcessorBase),
    "tracker": ("trackers", TrackerBase),
    "utility": ("utilities", UtilityBase),
}
_NODE_TYPES = set(COLUMNS)
_CAMERA_NAME_RE = re.compile(r"^[^\x00-\x1f]{1,80}$")
_LOCK = threading.RLock()
_RESTART_REQUIRED = False
_DEVICE_LOCK = threading.Lock()
_DEVICE_CACHE: list[dict] = []
_DEVICE_CACHE_TS = 0.0
_DEVICE_CACHE_SECONDS = 5.0

_CORE_ARTIFACT_TYPES = frozenset({"model", "source", "browse"})
_MODEL_PATH_KEYS = ("file_path", "source_pt", "path")


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


def _label_for(cls, name: str) -> str:
    return str(
        getattr(cls, "display_name", None)
        or getattr(cls, "label", None)
        or name
    )


def _doc_for(cls) -> str:
    return (getattr(cls, "__doc__", None) or "").strip()[:400]


def _defaults_from_schema(schema: dict, *, pipeline_name: str | None = None) -> dict:
    defaults = {}
    for key, defn in (schema or {}).items():
        if not isinstance(defn, dict) or "default" not in defn:
            continue
        value = _deep(defn["default"])
        if (
            pipeline_name
            and key == "vision_model"
            and defn.get("type") == "model"
            and not value
        ):
            value = default_vision_model()
        defaults[key] = value
    return defaults


def _explicit_settings(schema: dict, source: dict | None) -> dict:
    source = source if isinstance(source, dict) else {}
    return {
        key: _deep(source[key])
        for key in (schema or {})
        if key in source
    }


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
            "camera_type": cam_type,
            "label": CAMERA_TYPE_LABELS.get(cam_type, cam_type),
            "schema": schema,
            "defaults": _defaults_from_schema(schema),
        }
    return entries


def _claimed_sources(config) -> set[str]:
    claimed = set()
    raw = config.get("camera_configs", {}) if config is not None else {}
    if not isinstance(raw, dict):
        return claimed
    for entry in raw.values():
        if not isinstance(entry, dict):
            continue
        source = entry.get("source")
        if source is not None:
            claimed.add(str(source))
        if source is None and entry.get("path") is not None:
            claimed.add(str(entry.get("path")))
    return claimed


def _detected_devices(config, *, refresh: bool = False) -> list[dict]:
    """Probe physical cameras through the same discovery used by Cameras."""
    global _DEVICE_CACHE, _DEVICE_CACHE_TS
    now = time.monotonic()
    with _DEVICE_LOCK:
        if refresh or not _DEVICE_CACHE or (now - _DEVICE_CACHE_TS) >= _DEVICE_CACHE_SECONDS:
            try:
                _DEVICE_CACHE = [
                    dict(item)
                    for item in OpenCVCamera.discover(
                        claimed_sources=_claimed_sources(config)
                    )
                    if isinstance(item, dict)
                ]
            except Exception:
                logger.exception("Failed to discover cameras for editor")
                _DEVICE_CACHE = []
            _DEVICE_CACHE_TS = now
        claimed = _claimed_sources(config)
        return [
            dict(item)
            for item in _DEVICE_CACHE
            if str(item.get("path")) not in claimed
        ]


def _source_value(raw):
    if isinstance(raw, int):
        return raw
    text = str(raw if raw is not None else "").strip()
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return text


def _device_name(device: dict) -> str:
    return str(device.get("name") or device.get("path") or "Camera").strip()


def _unique_name(base: str, used: set[str]) -> str:
    name = base or "Camera"
    if name not in used:
        return name
    index = 2
    while f"{name} {index}" in used:
        index += 1
    return f"{name} {index}"


def _replay_sessions(config) -> list[dict]:
    raw = config.get("rollback", {}) if config is not None else {}
    root = Path(str((raw or {}).get("data_dir") or "VideoRecordings"))
    try:
        if not root.is_absolute():
            root = Path.cwd() / root
        if not root.is_dir():
            return []
        sessions = []
        for child in sorted(root.iterdir(), key=lambda item: item.name):
            if not child.is_dir():
                continue
            cams = []
            for sidecar in child.glob("*.json"):
                pass
            for segment in child.iterdir():
                if not segment.is_dir():
                    continue
                for cam in segment.iterdir():
                    if cam.is_dir() and cam.name not in cams:
                        cams.append(cam.name)
            sessions.append({"name": child.name, "path": str(child), "cameras": cams})
        return sessions
    except OSError:
        return []


def _palette(config, *, refresh: bool = False) -> dict:
    pipelines = _pipeline_catalog(config)
    cameras = _camera_catalog()
    catalog = _plugin_catalog()
    raw_plugins = config.get("plugins", {}) if hasattr(config, "get") else {}
    raw_plugins = raw_plugins if isinstance(raw_plugins, dict) else {}
    for ptype, bucket in (
        ("frame_processor", "frame_processors"),
        ("tracker", "trackers"),
        ("utility", "utilities"),
    ):
        current_bucket = raw_plugins.get(bucket)
        current_bucket = current_bucket if isinstance(current_bucket, dict) else {}
        for name, item in catalog.get(ptype, {}).items():
            current = current_bucket.get(name)
            item["settings"] = _deep(current) if isinstance(current, dict) else {}
            merged = _deep(item.get("defaults", {}))
            if isinstance(current, dict):
                merged.update(_deep(current))
            item["defaults"] = merged

    admin = bool(config.get("admin")) if config is not None else False
    cams = config.get("camera_configs", {}) if config is not None else {}
    cams = cams if isinstance(cams, dict) else {}
    used_names = set(cams)
    device_items = []
    for device in _detected_devices(config, refresh=refresh):
        source = _source_value(device.get("path"))
        device_id = device.get("device_id")
        base = _device_name(device)
        name = _unique_name(base, used_names)
        schema = cameras.get("opencv", {}).get("schema", {})
        device_items.append(
            {
                "name": name,
                "type": "camera",
                "kind": "device",
                "camera_type": "opencv",
                "label": base,
                "device_name": base,
                "device_id": device_id,
                "source": source,
                "resolved": device.get("resolution") or device.get("resolution_text"),
                "schema": schema,
                "settings": {
                    "source": source,
                    "device_id": device_id,
                    "name": name,
                },
                "defaults": cameras.get("opencv", {}).get("defaults", {}),
                "detected": True,
            }
        )

    tello_configured = any(
        isinstance(entry, dict) and entry.get("camera_type") == "tello"
        for entry in cams.values()
    )
    if admin and not tello_configured:
        item = _deep(cameras.get("tello", {}))
        item.update(
            {
                "name": "tello",
                "kind": "tello",
                "type": "camera",
                "camera_type": "tello",
                "label": "Tello EDU",
                "settings": {},
                "detected": False,
            }
        )
        device_items.append(item)

    replay_configured = any(
        isinstance(entry, dict) and entry.get("camera_type") == "replay"
        for entry in cams.values()
    )
    if not replay_configured:
        item = _deep(cameras.get("replay", {}))
        item.update(
            {
                "name": "replay",
                "kind": "replay",
                "type": "camera",
                "camera_type": "replay",
                "label": "Replay Session",
                "settings": {},
                "detected": False,
            }
        )
        device_items.append(item)

    return {
        "cameras": device_items,
        "pipelines": list(pipelines.values()),
        "frame_processors": list(catalog["frame_processor"].values()),
        "trackers": list(catalog["tracker"].values()),
        "utilities": list(catalog["utility"].values()),
        "columns": [
            {"type": name, "label": COLUMN_LABELS[name], "x": COLUMN_X[name]}
            for name in COLUMNS
        ],
        "replay_sessions": _replay_sessions(config),
        "admin": admin,
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


def _clean_layout(raw, config=None) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    viewport = raw.get("viewport") if isinstance(raw.get("viewport"), dict) else {}
    layout = {
        "viewport": {
            "x": _finite(viewport.get("x"), 0),
            "y": _finite(viewport.get("y"), 0),
            "zoom": _clamp(_finite(viewport.get("zoom"), 1), 0.25, 2.5),
        },
        "lane_order": [],
        "utility_order": [],
    }
    lane_order = raw.get("lane_order")
    if isinstance(lane_order, list):
        layout["lane_order"] = [
            str(name) for name in lane_order if isinstance(name, str) and name
        ]
    utility_order = raw.get("utility_order")
    if isinstance(utility_order, list):
        layout["utility_order"] = [
            str(name) for name in utility_order if isinstance(name, str) and name
        ]
    # Version-1 layouts stored per-node coordinates.  They are intentionally
    # discarded; only lane/utility order and the viewport survive migration.
    return layout


def _layout_from_config(config) -> dict:
    if config is None:
        return _clean_layout({})
    return _clean_layout(config.get("editor_graph", {}))


def _camera_order(config, layout: dict | None = None) -> list[str]:
    raw = config.get("camera_configs", {}) if config is not None else {}
    names = [name for name in raw if isinstance(name, str)] if isinstance(raw, dict) else []
    saved = (layout or {}).get("lane_order") or []
    ordered = [name for name in saved if name in names]
    ordered.extend(name for name in names if name not in ordered)
    return ordered


def _utility_order(config, layout: dict | None = None) -> list[str]:
    plugins = config.get("plugins", {}) if config is not None else {}
    bucket = plugins.get("utilities", {}) if isinstance(plugins, dict) else {}
    names = [
        name
        for name in bucket
        if isinstance(name, str) and name.lower() not in {"rollback", "roll_back"}
    ] if isinstance(bucket, dict) else []
    saved = (layout or {}).get("utility_order") or []
    ordered = [name for name in saved if name in names]
    ordered.extend(name for name in names if name not in ordered)
    return ordered


def _utility_sources(config, utility_name: str, cameras: list[str]) -> tuple[list[str], bool]:
    raw = config.get("utility_sources") if config is not None else None
    configured = isinstance(raw, dict) and utility_name in raw
    if not configured:
        return list(cameras), False
    values = raw.get(utility_name)
    if not isinstance(values, list):
        return list(cameras), False
    sources = []
    for value in values:
        name = str(value or "").strip()
        if name in cameras and name not in sources:
            sources.append(name)
    return sources, True


def _node_geometry(column: str, index: int, *, utility_index: int | None = None):
    if column == "utility":
        return COLUMN_X[column], UTILITY_TOP + (utility_index or 0) * UTILITY_GAP
    return COLUMN_X[column], index


def _make_node(
    *,
    node_id: str,
    node_type: str,
    label: str,
    column: str,
    lane: int | None,
    x: float,
    y: float,
    schema: dict | None = None,
    settings: dict | None = None,
    defaults: dict | None = None,
    **extra,
) -> dict:
    schema = _deep(schema or {})
    settings = _deep(settings or {})
    completion = completion_for_schema(schema, settings)
    node = {
        "id": node_id,
        "type": node_type,
        "column": column,
        "column_index": COLUMN_INDEX[column],
        "lane": lane,
        "label": label,
        "schema": schema,
        "settings": settings,
        "defaults": _deep(defaults or {}),
        "completion": completion,
        "required": completion["required"],
        "x": x,
        "y": y,
        "w": NODE_W,
        "h": NODE_H,
    }
    node.update(extra)
    return node


def _edge_state(source: dict | None, target: dict | None) -> str:
    if not source or not target:
        return "invalid"
    if source.get("completion", {}).get("complete") and target.get("completion", {}).get(
        "complete"
    ):
        return "complete"
    return "incomplete"


def _make_edge(source: dict, target: dict, kind: str, warning: str | None = None) -> dict:
    edge = {
        "id": f"edge:{source['id']}->{target['id']}",
        "from": source["id"],
        "to": target["id"],
        "kind": kind,
        "lane": source.get("lane"),
        "source_column": source.get("column"),
        "target_column": target.get("column"),
        "state": _edge_state(source, target),
    }
    if warning:
        edge["warning"] = warning
    return edge


def _derive_edges(nodes: list[dict]) -> list[dict]:
    cameras = [node for node in nodes if node["type"] == "camera"]
    pipelines = [node for node in nodes if node["type"] == "pipeline"]
    processors = [node for node in nodes if node["type"] == "frame_processor"]
    trackers = [node for node in nodes if node["type"] == "tracker"]
    utilities = [node for node in nodes if node["type"] == "utility"]
    edges = []
    seen = set()

    def add(source, target, kind):
        key = (source["id"], target["id"])
        if key in seen:
            return
        seen.add(key)
        edges.append(_make_edge(source, target, kind))

    for camera in cameras:
        lane = camera.get("lane")
        lane_fps = [node for node in processors if node.get("lane") == lane]
        lane_pipes = [node for node in pipelines if node.get("lane") == lane]
        lane_trackers = [node for node in trackers if node.get("lane") == lane]
        for processor in lane_fps:
            add(camera, processor, "camera_frame_processor")
        for processor in lane_fps:
            for pipeline in lane_pipes:
                add(processor, pipeline, "frame_processor_pipeline")
        for pipeline in lane_pipes:
            for tracker in lane_trackers:
                add(pipeline, tracker, "pipeline_tracker")

    trackers_by_lane = {}
    for tracker in trackers:
        trackers_by_lane.setdefault(tracker.get("lane"), []).append(tracker)
    for utility in utilities:
        for camera_name in utility.get("sources", []):
            lane = next(
                (node.get("lane") for node in cameras if node.get("camera") == camera_name),
                None,
            )
            if lane is None:
                continue
            for tracker in trackers_by_lane.get(lane, []):
                add(tracker, utility, "tracker_utility")
    return edges


def _column_counts(nodes: list[dict]) -> dict:
    counts = {}
    for column in COLUMNS:
        count = sum(
            1
            for node in nodes
            if node.get("column") == column
            and (column == "utility" or node.get("lane") is not None)
        )
        counts[column] = count
    return counts


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
    lane_order = _camera_order(config, layout)
    utility_order = _utility_order(config, layout)

    nodes = []
    lanes_meta = []
    shared = {}
    lane_heights = []
    for lane_index, camera_name in enumerate(lane_order):
        raw_entry = raw_cams.get(camera_name)
        if not isinstance(raw_entry, dict):
            continue
        entry = _deep(raw_entry)
        normalize_camera_entry(entry)
        cam_type = str(entry.get("camera_type") or "opencv")
        cam_cat = cameras_cat.get(cam_type) or cameras_cat.get("opencv") or {}
        cam_schema = cam_cat.get("schema", {})
        cam_settings = _explicit_settings(cam_schema, entry)
        camera_node = _make_node(
            node_id=f"camera:{camera_name}",
            node_type="camera",
            label=camera_name,
            column="camera",
            lane=lane_index,
            x=COLUMN_X["camera"],
            y=0,
            schema=cam_schema,
            settings=cam_settings,
            defaults=cam_cat.get("defaults", {}),
            camera=camera_name,
            plugin=None,
            camera_type=cam_type,
            calibrated=_camera_calibrated(entry),
            camera_subtitle=(
                str(entry.get("device_id") or entry.get("source") or cam_type)
            ),
        )
        fp_bucket = raw_plugins.get("frame_processors")
        fp_bucket = fp_bucket if isinstance(fp_bucket, dict) else {}
        tracker_bucket = raw_plugins.get("trackers")
        tracker_bucket = tracker_bucket if isinstance(tracker_bucket, dict) else {}
        fp_names = entry.get("frame_processors")
        if not isinstance(fp_names, list):
            fp_names = list(fp_bucket)
        else:
            # The Add-ons page owns enable/disable.  A stale lane assignment
            # must not resurrect a disabled processor.
            fp_names = [name for name in fp_names if name in fp_bucket]
        tracker_names = entry.get("trackers")
        if not isinstance(tracker_names, list):
            tracker_names = list(tracker_bucket)
        else:
            tracker_names = [
                name for name in tracker_names if name in tracker_bucket
            ]

        lane_nodes = [camera_node]
        for fp_index, name in enumerate(dict.fromkeys(str(n) for n in fp_names if str(n).strip())):
            cat = addon_cat.get("frame_processor", {}).get(name)
            if cat is None:
                continue
            settings = _explicit_settings(
                cat.get("schema", {}),
                (raw_plugins.get("frame_processors") or {}).get(name, {}),
            )
            shared.setdefault(("frame_processor", name), []).append(camera_name)
            lane_nodes.append(
                _make_node(
                    node_id=f"frame_processor:{camera_name}:{name}",
                    node_type="frame_processor",
                    label=cat.get("label") or name,
                    column="frame_processor",
                    lane=lane_index,
                    x=COLUMN_X["frame_processor"],
                    y=0,
                    schema=cat.get("schema", {}),
                    settings=settings,
                    defaults=cat.get("defaults", {}),
                    camera=camera_name,
                    plugin=name,
                    slot=fp_index,
                    subtitle=name,
                    supported_pipelines=cat.get("supported_pipelines", []),
                    needs_internet=cat.get("needs_internet", False),
                    shared_with=[camera_name],
                    shared_count=1,
                )
            )

        pipeline_name = get_pipeline_name(entry)
        pipeline_settings = get_pipeline_settings(entry)
        pip_cat = pipelines_cat.get(pipeline_name)
        if pip_cat is None and pipeline_name:
            pip_cat = {
                "name": pipeline_name,
                "label": pipeline_name,
                "schema": {},
                "defaults": {},
                "supported_pipelines": [],
                "needs_internet": False,
            }
        if pip_cat is not None:
            lane_nodes.append(
                _make_node(
                    node_id=f"pipeline:{camera_name}",
                    node_type="pipeline",
                    label=pip_cat.get("label") or pipeline_name,
                    column="pipeline",
                    lane=lane_index,
                    x=COLUMN_X["pipeline"],
                    y=0,
                    schema=pip_cat.get("schema", {}),
                    settings=_explicit_settings(pip_cat.get("schema", {}), pipeline_settings),
                    defaults=pip_cat.get("defaults", {}),
                    camera=camera_name,
                    plugin=pipeline_name,
                    subtitle=pipeline_name,
                    supported_pipelines=pip_cat.get("supported_pipelines", []),
                    needs_internet=pip_cat.get("needs_internet", False),
                )
            )

        for tracker_index, name in enumerate(
            dict.fromkeys(str(n) for n in tracker_names if str(n).strip())
        ):
            cat = addon_cat.get("tracker", {}).get(name)
            if cat is None:
                continue
            settings = _explicit_settings(
                cat.get("schema", {}),
                (raw_plugins.get("trackers") or {}).get(name, {}),
            )
            shared.setdefault(("tracker", name), []).append(camera_name)
            lane_nodes.append(
                _make_node(
                    node_id=f"tracker:{camera_name}:{name}",
                    node_type="tracker",
                    label=cat.get("label") or name,
                    column="tracker",
                    lane=lane_index,
                    x=COLUMN_X["tracker"],
                    y=0,
                    schema=cat.get("schema", {}),
                    settings=settings,
                    defaults=cat.get("defaults", {}),
                    camera=camera_name,
                    plugin=name,
                    slot=tracker_index,
                    subtitle=name,
                    supported_pipelines=cat.get("supported_pipelines", []),
                    needs_internet=cat.get("needs_internet", False),
                    shared_with=[camera_name],
                    shared_count=1,
                )
            )

        slot_count = max(
            1,
            len([node for node in lane_nodes if node["type"] == "frame_processor"]),
            len([node for node in lane_nodes if node["type"] == "tracker"]),
        )
        lane_height = max(MIN_LANE_HEIGHT, slot_count * (NODE_H + LANE_SLOT_GAP) + 32)
        lane_top = LANE_TOP + sum(lane_heights)
        lane_heights.append(lane_height)
        for node in lane_nodes:
            slot = int(node.get("slot", 0) or 0)
            node["y"] = lane_top + 16 + slot * (NODE_H + LANE_SLOT_GAP)
        lanes_meta.append(
            {
                "camera": camera_name,
                "label": camera_name,
                "index": lane_index,
                "y": lane_top,
                "height": lane_height,
            }
        )
        nodes.extend(lane_nodes)

    all_camera_names = [meta["camera"] for meta in lanes_meta]
    utility_lookup = addon_cat.get("utility", {})
    utility_nodes = []
    for utility_index, name in enumerate(utility_order):
        cat = utility_lookup.get(name)
        if cat is None:
            continue
        settings = _explicit_settings(
            cat.get("schema", {}),
            (raw_plugins.get("utilities") or {}).get(name, {}),
        )
        sources, configured = _utility_sources(config, name, all_camera_names)
        shared.setdefault(("utility", name), []).extend(sources)
        node = _make_node(
            node_id=f"utility:{name}",
            node_type="utility",
            label=cat.get("label") or name,
            column="utility",
            lane=None,
            x=COLUMN_X["utility"],
            y=UTILITY_TOP + utility_index * UTILITY_GAP,
            schema=cat.get("schema", {}),
            settings=settings,
            defaults=cat.get("defaults", {}),
            camera=None,
            plugin=name,
            utility_index=utility_index,
            subtitle=name,
            sources=sources,
            sources_configured=configured,
            needs_internet=cat.get("needs_internet", False),
            shared_with=list(sources),
            shared_count=len(sources),
        )
        utility_nodes.append(node)
    nodes.extend(utility_nodes)

    for node in nodes:
        key = (node["type"], node.get("plugin"))
        if node["type"] in {"frame_processor", "tracker"}:
            names = shared.get(key, [])
            node["shared_with"] = list(dict.fromkeys(names))
            node["shared_count"] = len(node["shared_with"])
        elif node["type"] == "utility":
            names = list(dict.fromkeys(shared.get(key, [])))
            node["shared_with"] = names
            node["shared_count"] = len(names)

    edges = _derive_edges(nodes)
    max_lane = LANE_TOP + sum(lane_heights)
    max_utility = UTILITY_TOP + max(1, len(utility_nodes)) * UTILITY_GAP
    graph = {
        "nodes": nodes,
        "edges": edges,
        "lanes": lanes_meta,
        "utilities": [node["id"] for node in utility_nodes],
        "columns": [
            {
                "type": column,
                "label": COLUMN_LABELS[column],
                "index": COLUMN_INDEX[column],
                "x": COLUMN_X[column],
                "count": _column_counts(nodes)[column],
            }
            for column in COLUMNS
        ],
        "utility_sources": {
            node["plugin"]: list(node["sources"]) for node in utility_nodes
        },
        "viewport": _deep(layout["viewport"]),
        "lane_order": list(lane_order),
        "utility_order": [node["plugin"] for node in utility_nodes],
        "layout_saved": bool(
            config is not None
            and isinstance(config.get("editor_graph"), dict)
            and bool(config.get("editor_graph"))
        ),
        "lane_height": max_lane,
        "canvas": {
            "width": COLUMN_X["utility"] + NODE_W + 48,
            "height": max(max_lane, max_utility) + 48,
        },
    }
    return graph


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
    model = settings.get("vision_model") if isinstance(settings, dict) else None
    if isinstance(model, str):
        return [model] if model.strip() else []
    if not isinstance(model, dict):
        return []
    paths = []
    for key in _MODEL_PATH_KEYS:
        value = model.get(key)
        if isinstance(value, str) and value.strip() and value.strip() not in paths:
            paths.append(value.strip())
    return paths


def _node_camera(raw: dict, node_type: str) -> str:
    camera = raw.get("camera")
    if isinstance(camera, str) and camera.strip():
        return camera.strip()
    node_id = str(raw.get("id") or "")
    parts = node_id.split(":")
    if node_type == "pipeline" and len(parts) >= 2:
        return parts[1]
    if node_type in {"frame_processor", "tracker"} and len(parts) >= 3:
        return parts[1]
    return ""


def _node_plugin(raw: dict) -> str:
    plugin = raw.get("plugin")
    if isinstance(plugin, str) and plugin.strip():
        return plugin.strip()
    node_id = str(raw.get("id") or "")
    parts = node_id.split(":")
    return parts[-1].strip() if parts else ""


def _source_is_url_or_file(source) -> bool:
    text = str(source if source is not None else "").strip()
    if not text:
        return False
    lower = text.lower()
    if lower.startswith(("rtsp://", "http://", "https://", "udp://", "tcp://")):
        return True
    suffixes = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
        ".avi",
        ".mp4",
        ".mov",
        ".mkv",
    }
    if Path(text).suffix.lower() in suffixes:
        return True
    try:
        return Path(text).expanduser().is_file()
    except OSError:
        return False


def _camera_is_detected(config, *, source, device_id) -> bool:
    if _source_is_url_or_file(source):
        return True
    if device_id:
        for device in _detected_devices(config):
            if str(device.get("device_id") or "") == str(device_id):
                return True
    source_text = str(source if source is not None else "").strip()
    for device in _detected_devices(config):
        if str(device.get("path") or "").strip() == source_text:
            return True
    return False


def _validate_edges(
    edges,
    nodes_by_id: dict,
    active_pipelines: set | None = None,
) -> list[dict]:
    """Validate edge direction, lane ownership, support, and cycles."""
    errors = []
    if not isinstance(edges, list):
        return [{"code": "edges_invalid", "message": "edges must be a list."}]
    valid_edges = []
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(
                {
                    "code": "edge_invalid",
                    "message": f"Connection {index + 1} is invalid.",
                }
            )
            continue
        source = nodes_by_id.get(edge.get("from"))
        target = nodes_by_id.get(edge.get("to"))
        if source is None or target is None:
            errors.append(
                {
                    "code": "edge_endpoint_missing",
                    "message": "Connection references a node that does not exist.",
                }
            )
            continue
        source_column = source.get("column") or source.get("type")
        target_column = target.get("column") or target.get("type")
        source_index = COLUMN_INDEX.get(source_column)
        target_index = COLUMN_INDEX.get(target_column)
        if source_index is None or target_index is None:
            errors.append(
                {
                    "code": "edge_type",
                    "message": "Connection uses an unsupported node type.",
                    "node": source.get("id"),
                }
            )
            continue
        if source_column == "utility":
            errors.append(
                {
                    "code": "edge_wrong_direction",
                    "message": "Utility nodes have no output. Edges must move left to right.",
                    "node": source.get("id"),
                }
            )
            continue
        if target_index <= source_index:
            errors.append(
                {
                    "code": "edge_wrong_direction",
                    "message": "Edges must move left to right to the next column.",
                    "node": source.get("id"),
                }
            )
            continue
        if target_index > source_index + 1:
            errors.append(
                {
                    "code": "edge_skipped_column",
                    "message": (
                        "Edges cannot skip a column. Connect "
                        f"{COLUMN_LABELS[source_column]} directly to the next column."
                    ),
                    "node": source.get("id"),
                }
            )
            continue
        same_lane = source.get("lane") == target.get("lane")
        allowed_cross_lane = source_column == "tracker" and target_column == "utility"
        if not same_lane and not allowed_cross_lane:
            errors.append(
                {
                    "code": "edge_cross_lane",
                    "message": (
                        "Edges may only connect nodes in the same camera lane, "
                        "except a Tracker -> Utility connection."
                    ),
                    "node": source.get("id"),
                }
            )
            continue
        if source_column == "frame_processor":
            supported = source.get("supported_pipelines") or []
            pipeline = target.get("plugin") or ""
            if supported and pipeline and pipeline not in supported:
                errors.append(
                    {
                        "code": "edge_pipeline_unsupported",
                        "message": (
                            f"Frame processor '{source.get('plugin')}' does not "
                            f"support pipeline '{pipeline}'."
                        ),
                        "node": source.get("id"),
                    }
                )
                continue
        valid_edges.append((source, target))

    # Cycle detection over the structurally valid connections.
    adjacency = {}
    for source, target in valid_edges:
        adjacency.setdefault(source["id"], []).append(target["id"])

    def visit(node_id, visiting, visited):
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        visiting.add(node_id)
        for nxt in adjacency.get(node_id, []):
            if visit(nxt, visiting, visited):
                return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    visiting = set()
    visited = set()
    for node_id in list(adjacency):
        if visit(node_id, visiting, visited):
            errors.append(
                {
                    "code": "edge_cycle",
                    "message": "This connection would create a cycle.",
                    "node": node_id,
                }
            )
            break
    return errors


def _duplicate_errors(nodes: list[dict]) -> list[dict]:
    errors = []
    seen = {}
    for node in nodes:
        key = (node.get("column"), node.get("id"))
        seen.setdefault(key, []).append(node)
    cameras = {}
    for node in nodes:
        if node.get("type") == "camera":
            name = node.get("camera")
            if name in cameras:
                errors.append(
                    {
                        "code": "duplicate_camera",
                        "message": f"Camera '{name}' is declared more than once.",
                        "node": node.get("id"),
                    }
                )
            cameras[name] = node
    lane_counts = {}
    for node in nodes:
        if node.get("column") == "utility":
            key = ("utility", node.get("plugin"))
        else:
            key = (node.get("lane"), node.get("column"))
        lane_counts.setdefault(key, []).append(node)
    for (lane, column), items in lane_counts.items():
        if len(items) <= 1:
            continue
        camera = next((item.get("camera") for item in items if item.get("camera")), "")
        label = COLUMN_LABELS.get(column, column)
        if column == "utility":
            errors.append(
                {
                    "code": "duplicate_utility",
                    "message": (
                        f"Utility '{items[0].get('plugin')}' can only be placed once."
                    ),
                    "node": items[0].get("id"),
                }
            )
        else:
            errors.append(
                {
                    "code": "duplicate_column",
                    "message": (
                        f"Lane '{camera}' has more than one {label} node. "
                        "Keep exactly one node per column."
                    ),
                    "node": items[-1].get("id"),
                }
            )
    return errors


def _settings_completion_errors(node: dict) -> list[dict]:
    progress = node.get("completion") or completion_for_schema(
        node.get("schema", {}), node.get("settings", {})
    )
    if progress.get("complete"):
        return []
    label = {
        "camera": f"Camera '{node.get('camera')}'",
        "pipeline": f"Pipeline '{node.get('plugin')}' on camera '{node.get('camera')}'",
        "frame_processor": f"Frame processor '{node.get('plugin')}'",
        "tracker": f"Tracker '{node.get('plugin')}'",
        "utility": f"Utility '{node.get('plugin')}'",
    }.get(node.get("type"), node.get("label") or node.get("id"))
    return [
        {
            "code": "settings_incomplete",
            "message": (
                f"{label} is missing required field(s): "
                f"{', '.join(progress.get('missing_labels') or progress.get('missing') or [])}."
            ),
            "node": node.get("id"),
            "missing": progress.get("missing", []),
        }
    ]


def _normalize_graph(payload: dict, config) -> tuple[dict, list[dict], list[dict]]:
    """Validate a lane graph and turn it into executable config structures."""
    errors: list[dict] = []
    warnings: list[dict] = []
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
    admin = bool(config.get("admin")) if config is not None else False

    parsed = []
    camera_entries = []
    raw_camera_nodes = []
    for raw_index, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            errors.append(
                {
                    "code": "node_invalid",
                    "message": f"Node {raw_index + 1} is not an object.",
                }
            )
            continue
        node_type = str(raw.get("type") or raw.get("category") or "").strip()
        if node_type not in _NODE_TYPES:
            errors.append(
                {
                    "code": "node_type_invalid",
                    "message": f"Node '{raw.get('id') or raw_index + 1}' has an unsupported type.",
                    "node": raw.get("id"),
                }
            )
            continue
        if node_type == "camera":
            name = str(raw.get("camera") or raw.get("name") or raw.get("label") or "").strip()
            raw["_camera_name"] = name
            raw["_input_index"] = raw_index
            raw_camera_nodes.append(raw)
        parsed.append(raw)

    raw_camera_nodes.sort(
        key=lambda item: (
            _finite(item.get("lane"), item.get("_input_index", 0)),
            item.get("_input_index", 0),
        )
    )
    lane_by_camera = {}
    used_camera_names = set()
    for lane, raw in enumerate(raw_camera_nodes):
        name = raw.get("_camera_name") or ""
        node_id = str(raw.get("id") or f"camera:{name}")
        if not _CAMERA_NAME_RE.match(name):
            errors.append(
                {
                    "code": "camera_name_invalid",
                    "message": "Camera names must be 1-80 visible characters.",
                    "node": node_id,
                }
            )
            continue
        if name in used_camera_names:
            errors.append(
                {
                    "code": "duplicate_camera",
                    "message": f"Camera '{name}' is declared more than once.",
                    "node": node_id,
                }
            )
            continue
        used_camera_names.add(name)
        lane_by_camera[name] = lane
        camera_entries.append((lane, name, raw))

    # Build normalized nodes by column.
    nodes: list[dict] = []
    nodes_by_id: dict[str, dict] = {}
    camera_nodes: dict[str, dict] = {}
    pipeline_nodes: dict[str, dict] = {}
    fp_nodes: list[dict] = []
    tracker_nodes: list[dict] = []
    utility_nodes: list[dict] = []
    settings_by_plugin: dict[tuple[str, str], dict] = {}

    for raw in parsed:
        node_type = str(raw.get("type") or "").strip()
        if node_type == "camera":
            continue
        node_id = str(raw.get("id") or "")
        if not node_id:
            node_id = f"{node_type}:{raw.get('camera') or ''}:{_node_plugin(raw)}"
        if node_id in nodes_by_id:
            errors.append(
                {
                    "code": "duplicate_node",
                    "message": f"Node id '{node_id}' is used more than once.",
                    "node": node_id,
                }
            )
            continue
        camera = _node_camera(raw, node_type)
        if node_type == "utility":
            camera = ""
        if node_type != "utility" and camera not in lane_by_camera:
            errors.append(
                {
                    "code": "camera_missing",
                    "message": (
                        f"{COLUMN_LABELS.get(node_type, node_type)} '{node_id}' "
                        f"references unknown camera '{camera}'."
                    ),
                    "node": node_id,
                }
            )
            continue
        lane = lane_by_camera.get(camera) if node_type != "utility" else None
        node = {
            "id": node_id,
            "type": node_type,
            "column": node_type,
            "column_index": COLUMN_INDEX[node_type],
            "camera": camera,
            "lane": lane,
            "plugin": _node_plugin(raw),
            "settings": _deep(raw.get("settings") if isinstance(raw.get("settings"), dict) else {}),
            "schema": {},
            "defaults": {},
            "sources_provided": (
                node_type == "utility" and isinstance(raw.get("sources"), list)
            ),
            "sources": (
                [
                    str(value).strip()
                    for value in raw.get("sources", [])
                    if str(value).strip()
                ]
                if node_type == "utility"
                else []
            ),
        }
        if node_type == "pipeline":
            if camera in pipeline_nodes:
                errors.append(
                    {
                        "code": "duplicate_column",
                        "message": (
                            f"Lane '{camera}' has more than one Pipeline node. "
                            "Keep exactly one node per column."
                        ),
                        "node": node_id,
                    }
                )
                continue
            cat = pipelines_cat.get(node["plugin"])
            if cat is None:
                errors.append(
                    {
                        "code": "plugin_unknown",
                        "message": f"Unknown pipeline '{node['plugin']}'.",
                        "node": node_id,
                    }
                )
                continue
            node["schema"] = cat.get("schema", {})
            node["defaults"] = cat.get("defaults", {})
            node["settings"] = _coerce_settings(node["schema"], node["settings"])
            _normalize_model(node["settings"])
            node["label"] = cat.get("label") or node["plugin"]
            node["supported_pipelines"] = cat.get("supported_pipelines", [])
            node["needs_internet"] = bool(cat.get("needs_internet"))
            pipeline_nodes[camera] = node
        elif node_type == "frame_processor":
            cat = addon_cat.get("frame_processor", {}).get(node["plugin"])
            if cat is None:
                errors.append(
                    {
                        "code": "plugin_unknown",
                        "message": f"Unknown frame processor '{node['plugin']}'.",
                        "node": node_id,
                    }
                )
                continue
            node["schema"] = cat.get("schema", {})
            node["defaults"] = cat.get("defaults", {})
            node["settings"] = _coerce_settings(node["schema"], node["settings"])
            node["label"] = cat.get("label") or node["plugin"]
            node["supported_pipelines"] = cat.get("supported_pipelines", [])
            node["needs_internet"] = bool(cat.get("needs_internet"))
            fp_nodes.append(node)
        elif node_type == "tracker":
            cat = addon_cat.get("tracker", {}).get(node["plugin"])
            if cat is None:
                errors.append(
                    {
                        "code": "plugin_unknown",
                        "message": f"Unknown tracker '{node['plugin']}'.",
                        "node": node_id,
                    }
                )
                continue
            node["schema"] = cat.get("schema", {})
            node["defaults"] = cat.get("defaults", {})
            node["settings"] = _coerce_settings(node["schema"], node["settings"])
            node["label"] = cat.get("label") or node["plugin"]
            node["supported_pipelines"] = cat.get("supported_pipelines", [])
            node["needs_internet"] = bool(cat.get("needs_internet"))
            tracker_nodes.append(node)
        else:
            if any(
                item.get("plugin") == node["plugin"] for item in utility_nodes
            ):
                errors.append(
                    {
                        "code": "duplicate_utility",
                        "message": (
                            f"Utility '{node['plugin']}' can only be placed once."
                        ),
                        "node": node_id,
                    }
                )
                continue
            cat = addon_cat.get("utility", {}).get(node["plugin"])
            if cat is None:
                errors.append(
                    {
                        "code": "plugin_unknown",
                        "message": f"Unknown utility '{node['plugin']}'.",
                        "node": node_id,
                    }
                )
                continue
            node["schema"] = cat.get("schema", {})
            node["defaults"] = cat.get("defaults", {})
            node["settings"] = _coerce_settings(node["schema"], node["settings"])
            node["label"] = cat.get("label") or node["plugin"]
            node["needs_internet"] = bool(cat.get("needs_internet"))
            utility_nodes.append(node)

        if node["plugin"] and node_type in {"frame_processor", "tracker", "utility"}:
            settings_by_plugin.setdefault((node_type, node["plugin"]), {}).update(
                _deep(node["settings"])
            )
        nodes_by_id[node_id] = node
        nodes.append(node)

    # Shared plugin settings are global by plugin name, exactly like the Add-ons
    # page.  Merge explicit keys from every lane before completion is checked.
    for node in nodes:
        if node["type"] in {"frame_processor", "tracker", "utility"}:
            merged = settings_by_plugin.get((node["type"], node["plugin"]))
            if merged is not None:
                node["settings"] = _deep(merged)

    for lane, name, raw in camera_entries:
        node_id = str(raw.get("id") or f"camera:{name}")
        if node_id in nodes_by_id:
            errors.append(
                {
                    "code": "duplicate_node",
                    "message": f"Node id '{node_id}' is used more than once.",
                    "node": node_id,
                }
            )
            continue
        renamed_from = str(raw.get("renamed_from") or "").strip()
        existing = _deep(existing_cams.get(name) or existing_cams.get(renamed_from) or {})
        cam_type = str(
            raw.get("camera_type")
            or (raw.get("settings") or {}).get("camera_type")
            or existing.get("camera_type")
            or "opencv"
        ).strip()
        if cam_type not in cameras_cat:
            errors.append(
                {
                    "code": "camera_type_unknown",
                    "message": f"Unknown camera type '{cam_type}'.",
                    "node": node_id,
                }
            )
            continue
        is_existing = name in existing_cams or bool(existing)
        if cam_type == "tello" and not admin:
            if is_existing and existing.get("camera_type") == "tello":
                warnings.append(
                    {
                        "code": "camera_tello_legacy",
                        "message": (
                            f"Camera '{name}' is an existing Tello camera. Tello "
                            "cameras are hidden unless admin mode is enabled."
                        ),
                        "node": node_id,
                    }
                )
            else:
                errors.append(
                    {
                        "code": "camera_tello_admin",
                        "message": "Tello EDU cameras require admin mode.",
                        "node": node_id,
                    }
                )
                continue
        schema = cameras_cat[cam_type].get("schema", {})
        settings = _coerce_settings(schema, raw.get("settings"))
        source = settings.get("source")
        device_id = settings.get("device_id")
        if not is_existing and cam_type == "opencv":
            if not _camera_is_detected(config, source=source, device_id=device_id):
                errors.append(
                    {
                        "code": "camera_undetected",
                        "message": (
                            f"Camera '{name}' is not a detected device. Refresh "
                            "devices and pick one from the palette."
                        ),
                        "node": node_id,
                    }
                )
        node = {
            "id": node_id,
            "type": "camera",
            "column": "camera",
            "column_index": 0,
            "camera": name,
            "lane": lane,
            "plugin": None,
            "camera_type": cam_type,
            "renamed_from": renamed_from,
            "schema": schema,
            "defaults": cameras_cat[cam_type].get("defaults", {}),
            "settings": settings,
            "label": name,
            "calibrated": _camera_calibrated(existing),
        }
        camera_nodes[name] = node
        nodes_by_id[node_id] = node
        nodes.append(node)

    replay_nodes = [node for node in camera_nodes.values() if node.get("camera_type") == "replay"]
    for extra in replay_nodes[1:]:
        errors.append(
            {
                "code": "camera_replay_limit",
                "message": "Only one replay camera is allowed.",
                "node": extra["id"],
            }
        )

    errors.extend(_duplicate_errors(list(nodes_by_id.values())))

    # Completion is explicit-value based.  Defaults never count until the
    # browser confirms them or the user presses "Use Defaults".
    for node in nodes_by_id.values():
        node["completion"] = completion_for_schema(
            node.get("schema", {}), node.get("settings", {})
        )
        node["required"] = node["completion"]["required"]
        errors.extend(_settings_completion_errors(node))

    # Lane completeness: one Camera, Frame Processor, Pipeline and Tracker per
    # lane.  Utilities are shared and validated separately below.
    for lane, name, raw in camera_entries:
        camera_node = camera_nodes.get(name)
        if camera_node is None:
            continue
        lane_fps = [node for node in fp_nodes if node.get("camera") == name]
        lane_pipes = [node for node in pipeline_nodes.values() if node.get("camera") == name]
        lane_trackers = [node for node in tracker_nodes if node.get("camera") == name]
        if not lane_fps:
            errors.append(
                {
                    "code": "lane_missing_frame_processor",
                    "message": (
                        f"Lane '{name}' is missing a frame processor. Add one to "
                        "fill the Frame Processor column."
                    ),
                    "node": camera_node["id"],
                }
            )
        if not lane_pipes:
            errors.append(
                {
                    "code": "lane_missing_pipeline",
                    "message": (
                        f"Lane '{name}' is missing a pipeline. Add one to fill "
                        "the Pipeline column."
                    ),
                    "node": camera_node["id"],
                }
            )
        if not lane_trackers:
            errors.append(
                {
                    "code": "lane_missing_tracker",
                    "message": (
                        f"Lane '{name}' is missing a tracker. Add one to fill "
                        "the Tracker column."
                    ),
                    "node": camera_node["id"],
                }
            )
        if lane_pipes and lane_fps:
            pipeline = lane_pipes[0]
            for processor in lane_fps:
                supported = processor.get("supported_pipelines") or []
                if supported and pipeline.get("plugin") not in supported:
                    errors.append(
                        {
                            "code": "edge_pipeline_unsupported",
                            "message": (
                                f"Frame processor '{processor.get('plugin')}' does "
                                f"not support pipeline '{pipeline.get('plugin')}'."
                            ),
                            "node": processor["id"],
                        }
                    )

    # Utility sources come from explicit tracker -> utility edges in the
    # payload, falling back to the utility node's saved source list.
    raw_edges = payload.get("edges")
    supplied_edges = raw_edges if isinstance(raw_edges, list) else []
    if supplied_edges:
        errors.extend(_validate_edges(supplied_edges, nodes_by_id))
    utility_sources = {
        node["plugin"]: list(
            dict.fromkeys(
                [
                    str(value)
                    for value in (node.get("sources") or [])
                    if str(value) in lane_by_camera
                ]
            )
        )
        for node in utility_nodes
        if node.get("sources_provided")
    }
    for edge in supplied_edges:
        if not isinstance(edge, dict):
            continue
        source = nodes_by_id.get(edge.get("from"))
        target = nodes_by_id.get(edge.get("to"))
        if source and target and source.get("type") == "tracker" and target.get("type") == "utility":
            utility_sources.setdefault(target.get("plugin"), [])
            camera = source.get("camera")
            if camera in lane_by_camera and camera not in utility_sources[target.get("plugin")]:
                utility_sources[target.get("plugin")].append(camera)

    for node in utility_nodes:
        plugin = node["plugin"]
        sources = utility_sources.get(plugin)
        if sources is None:
            sources, _configured = _utility_sources(config, plugin, list(lane_by_camera))
            utility_sources[plugin] = sources
        if not sources:
            errors.append(
                {
                    "code": "utility_no_inputs",
                    "message": f"Utility '{plugin}' needs at least one tracker input.",
                    "node": node["id"],
                }
            )
        node["sources"] = list(sources)
        node["shared_with"] = list(sources)
        node["shared_count"] = len(sources)

    # Every tracker must feed at least one utility.
    for tracker in tracker_nodes:
        if not any(
            tracker.get("camera") in utility_sources.get(utility.get("plugin"), [])
            for utility in utility_nodes
        ):
            errors.append(
                {
                    "code": "tracker_without_utility",
                    "message": (
                        f"Tracker '{tracker.get('plugin')}' on lane "
                        f"'{tracker.get('camera')}' has no utility output."
                    ),
                    "node": tracker["id"],
                }
            )

    # A supplied edge list must still contain the five-column structural chain.
    if supplied_edges:
        supplied_keys = {
            (str(edge.get("from")), str(edge.get("to")))
            for edge in supplied_edges
            if isinstance(edge, dict)
        }
        for lane, name, _raw in camera_entries:
            camera_node = camera_nodes.get(name)
            lane_fps = [node for node in fp_nodes if node.get("camera") == name]
            lane_pipes = [node for node in pipeline_nodes.values() if node.get("camera") == name]
            lane_trackers = [node for node in tracker_nodes if node.get("camera") == name]
            expected = []
            if camera_node and lane_fps:
                expected.extend((camera_node["id"], node["id"], "Camera -> Frame Processor") for node in lane_fps)
            if lane_fps and lane_pipes:
                expected.extend((node["id"], lane_pipes[0]["id"], "Frame Processor -> Pipeline") for node in lane_fps)
            if lane_pipes and lane_trackers:
                expected.extend((lane_pipes[0]["id"], node["id"], "Pipeline -> Tracker") for node in lane_trackers)
            for source_id, target_id, label in expected:
                if (source_id, target_id) not in supplied_keys:
                    errors.append(
                        {
                            "code": "lane_missing_connection",
                            "message": f"Lane '{name}' is missing the {label} connection.",
                            "node": camera_node["id"] if camera_node else None,
                        }
                    )

    # Build executable structures.
    new_cams = {}
    for lane, name, raw in camera_entries:
        camera_node = camera_nodes.get(name)
        if camera_node is None:
            continue
        existing = _deep(
            existing_cams.get(name)
            or existing_cams.get(camera_node.get("renamed_from") or "")
            or {}
        )
        if not existing:
            existing = {
                "name": name,
                "camera_type": camera_node.get("camera_type") or "opencv",
                "calibration": {
                    "distance": 0,
                    "game_piece_size": 0,
                    "size": 0,
                    "fov": 0,
                },
            }
        for key, value in camera_node.get("settings", {}).items():
            if key in _CAMERA_CORE_KEYS and key not in {"name", "frame_processors", "trackers"}:
                existing[key] = _deep(value)
        existing["name"] = name
        existing["camera_type"] = camera_node.get("camera_type") or "opencv"
        existing.pop("pipeline", None)
        pipeline = pipeline_nodes.get(name)
        existing["pipeline"] = {
            "name": pipeline.get("plugin") if pipeline else "",
            "settings": _deep(pipeline.get("settings", {})) if pipeline else {},
        }
        existing["frame_processors"] = [
            node.get("plugin")
            for node in fp_nodes
            if node.get("camera") == name
        ]
        existing["trackers"] = [
            node.get("plugin")
            for node in tracker_nodes
            if node.get("camera") == name
        ]
        normalize_camera_entry(existing)
        _normalize_model(get_pipeline_settings(existing))
        new_cams[name] = existing

    new_plugins = _deep(existing_plugins)
    if not isinstance(new_plugins, dict):
        new_plugins = {}
    new_fps = new_plugins.get("frame_processors")
    new_fps = _deep(new_fps) if isinstance(new_fps, dict) else {}
    fp_by_plugin = {}
    for node in fp_nodes:
        fp_by_plugin.setdefault(node["plugin"], {}).update(_deep(node.get("settings", {})))
    for plugin, settings in fp_by_plugin.items():
        new_fps[plugin] = _deep(settings)
    new_plugins["frame_processors"] = new_fps
    tracker_by_plugin = {}
    for node in tracker_nodes:
        tracker_by_plugin.setdefault(node["plugin"], {}).update(_deep(node.get("settings", {})))
    utility_by_plugin = {}
    for node in utility_nodes:
        utility_by_plugin.setdefault(node["plugin"], {}).update(_deep(node.get("settings", {})))
    new_plugins["trackers"] = tracker_by_plugin
    new_plugins["utilities"] = utility_by_plugin
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

    requested_layout = _clean_layout(
        payload.get("layout") if isinstance(payload.get("layout"), dict) else payload,
        config,
    )
    # Utilities do not own a lane, so their vertical order can only come from
    # the saved utility order (the browser keeps node objects in canvas order
    # and also sends this list explicitly).
    requested_utilities = requested_layout.get("utility_order") or []
    utility_rank = {
        name: index for index, name in enumerate(requested_utilities)
    }
    utility_nodes.sort(
        key=lambda node: utility_rank.get(
            node.get("plugin"), len(utility_rank)
        )
    )
    layout = requested_layout
    layout["lane_order"] = [name for _lane, name, _raw in camera_entries]
    layout["utility_order"] = [node["plugin"] for node in utility_nodes]

    for node in nodes:
        if node.get("type") in {"frame_processor", "pipeline", "tracker"}:
            for path in _model_paths(node.get("settings", {})):
                try:
                    if not Path(path).expanduser().exists():
                        warnings.append(
                            {
                                "code": "model_path_missing",
                                "message": (
                                    f"Pipeline '{node.get('plugin')}' references model "
                                    f"'{path}', which does not exist on this machine yet."
                                ),
                                "node": node["id"],
                            }
                        )
                except OSError:
                    pass

    if not camera_entries:
        errors.append(
            {
                "code": "no_cameras",
                "message": "Add at least one camera lane before saving.",
            }
        )

    normalized_nodes = []
    for node in nodes:
        normalized_nodes.append(
            {
                "id": node["id"],
                "type": node["type"],
                "column": node["column"],
                "lane": node.get("lane"),
                "camera": node.get("camera"),
                "plugin": node.get("plugin"),
                "schema": _deep(node.get("schema", {})),
                "settings": _deep(node.get("settings", {})),
                "defaults": _deep(node.get("defaults", {})),
                "completion": node.get("completion", {}),
                "sources": list(node.get("sources", [])),
                "label": node.get("label"),
            }
        )

    return (
        {
            "cameras": new_cams,
            "plugins": new_plugins,
            "utility_sources": utility_sources,
            "layout": layout,
            "nodes": normalized_nodes,
        },
        errors,
        warnings,
    )


def _validation_payload(payload: dict, config) -> tuple[dict, int]:
    normalized, errors, warnings = _normalize_graph(payload, config)
    completion = {
        node["id"]: node.get("completion")
        or completion_for_schema(node.get("schema", {}), node.get("settings", {}))
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


def _require_config_write(view):
    """Match the Cameras/Add-ons config-write policy.

    Config saves are allowed from loopback and, when the documented
    ``allow_remote_restart`` setting permits it, from the robot LAN.  A valid
    ``ISPY_ADMIN_TOKEN`` is always honored.  Upload/create/delete routes keep
    the stricter ``require_local_or_token`` policy.
    """

    @wraps(view)
    def wrapper(self, *args, **kwargs):
        config = self._config()
        allow_remote = allow_remote_restart(config) if config is not None else True
        if restart_allowed(
            request.remote_addr,
            request.headers,
            allow_remote=allow_remote,
        ):
            return view(self, *args, **kwargs)
        message = remote_denied_message(allow_remote)
        return (
            jsonify(
                ok=False,
                error=message,
                errors=[{"code": "forbidden", "message": message}],
            ),
            403,
        )

    return wrapper


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
            self._save_graph_guarded,
            methods=["POST"],
        )
        flask_app.add_url_rule(
            "/api/editor/validate",
            "api_editor_validate",
            self._validate,
            methods=["POST"],
        )
        flask_app.add_url_rule(
            "/api/editor/devices",
            "api_editor_devices",
            self._get_devices,
            methods=["GET"],
        )

    def _config(self):
        return self.context.get("config")

    def _save_graph_guarded(self):
        """Authorize then save.

        The authorization wrapper is kept as a method rather than wrapping the
        already-bound view function: Flask calls the registered function as a
        plain callable, while ``@wraps(bound_method)`` can make its signature
        look bound and lead to the instance being supplied twice.
        """
        config = self._config()
        allow_remote = (
            allow_remote_restart(config) if config is not None else True
        )
        if restart_allowed(
            request.remote_addr,
            request.headers,
            allow_remote=allow_remote,
        ):
            return self._save_graph()
        message = remote_denied_message(allow_remote)
        return (
            jsonify(
                ok=False,
                error=message,
                errors=[{"code": "forbidden", "message": message}],
            ),
            403,
        )

    def _get_graph(self):
        config = self._config()
        if config is None:
            return jsonify(error="No config available"), 500
        refresh = request.args.get("refresh", "").lower() in {"1", "true", "yes"}
        with _LOCK:
            graph = build_graph(config)
            palette = _palette(config, refresh=refresh)
        return jsonify(
            ok=True,
            graph=graph,
            palette=palette,
            restart_required=_RESTART_REQUIRED,
            admin=bool(config.get("admin")),
            config_file=str(getattr(config, "file_path", "") or ""),
        )

    def _get_devices(self):
        config = self._config()
        if config is None:
            return jsonify(error="No config available"), 500
        refresh = request.args.get("refresh", "").lower() in {"1", "true", "yes"}
        return jsonify(devices=_detected_devices(config, refresh=refresh))

    def _validate(self):
        config = self._config()
        if config is None:
            return jsonify(error="No config available"), 500
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return (
                jsonify(
                    ok=False,
                    errors=[{"code": "payload", "message": "JSON object required."}],
                ),
                400,
            )
        result, status = _validation_payload(payload, config)
        return jsonify(result), status

    def _save_graph(self):
        global _RESTART_REQUIRED
        config = self._config()
        if config is None:
            return jsonify(error="No config available"), 500
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return (
                jsonify(
                    ok=False,
                    errors=[{"code": "payload", "message": "JSON object required."}],
                ),
                400,
            )
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
            config.set("utility_sources", normalized["utility_sources"])
            config.set("editor_graph", normalized["layout"])
            try:
                config._rebuild_camera_configs()
            except Exception:
                logger.exception("Failed to rebuild camera config wrappers")
            config.save()
            _RESTART_REQUIRED = True
            graph = build_graph(config)
            palette = _palette(config)
        return jsonify(
            ok=True,
            errors=[],
            warnings=warnings,
            graph=graph,
            palette=palette,
            restart_required=True,
            note=(
                "Restart iSpy to apply camera, pipeline, tracker, utility, or "
                "frame-processor changes."
            ),
        )
