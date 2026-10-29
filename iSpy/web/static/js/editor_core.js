// Pure helpers for the lane-based pipeline editor.  This file is shared by
// the browser and `node --test`, so it has no imports and no DOM access.
//
// A lane owns exactly one node per column:
//   Camera -> Frame Processor -> Pipeline -> Tracker -> Utility
// Cameras are horizontal lanes; utilities are global cards in the last
// column.  Geometry is always derived from lane/column, never from a stored
// x/y on the node.

export const COLUMNS = [
  "camera",
  "frame_processor",
  "pipeline",
  "tracker",
  "utility",
];

export const COLUMN_INDEX = Object.fromEntries(
  COLUMNS.map((name, index) => [name, index]),
);

export const COLUMN_LABELS = {
  camera: "Camera",
  frame_processor: "Frame Processor",
  pipeline: "Pipeline",
  tracker: "Tracker",
  utility: "Utility",
};

export const COLUMN_X = {
  camera: 32,
  frame_processor: 316,
  pipeline: 600,
  tracker: 884,
  utility: 1168,
};

export const NODE_W = 224;
export const NODE_H = 104;
export const NODE_SIZES = Object.fromEntries(
  COLUMNS.map((column) => [column, { w: NODE_W, h: NODE_H }]),
);

export const LANE_TOP = 104;
export const MIN_LANE_HEIGHT = 144;
export const LANE_SLOT_GAP = 8;
export const UTILITY_TOP = 104;
export const UTILITY_GAP = 132;
export const COLUMN_GAP = 284;

const CORE_ARTIFACT_TYPES = new Set(["model", "source", "browse"]);

export function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

export function finite(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

export function deep(value) {
  if (value === undefined) return undefined;
  return JSON.parse(JSON.stringify(value));
}

export function nodeSize(type) {
  return NODE_SIZES[type] || { w: NODE_W, h: NODE_H };
}

export function columnForType(type) {
  return COLUMNS.includes(type) ? type : null;
}

function hasOwn(value, key) {
  return Boolean(
    value && Object.prototype.hasOwnProperty.call(value, key),
  );
}

export function effectiveValue(schema, settings, key) {
  if (hasOwn(settings, key)) return settings[key];
  const defn = schema && schema[key];
  return defn && hasOwn(defn, "default") ? defn.default : undefined;
}

export function isExplicit(settings, key) {
  return hasOwn(settings, key);
}

export function gateActive(schema, settings, defn) {
  if (!defn || !defn.gated_by) return true;
  const gates = Array.isArray(defn.gated_by) ? defn.gated_by : [defn.gated_by];
  return gates.every((name) => Boolean(effectiveValue(schema, settings, name)));
}

export function valueIsBlank(value, defn = {}) {
  if (value === null || value === undefined) return true;
  if (defn.type === "model") {
    if (Array.isArray(value)) {
      return !value.some((part) => String(part ?? "").trim());
    }
    if (typeof value === "object") {
      return !["file_path", "source_pt", "path", "name"].some((key) =>
        String(value[key] ?? "").trim(),
      );
    }
    return !String(value).trim();
  }
  if (["source", "browse", "text", "number", "select"].includes(defn.type)) {
    return !String(value).trim();
  }
  if (defn.type === "list") return Array.isArray(value) && value.length === 0;
  if (defn.type === "range") return !Array.isArray(value) || value.length !== 2;
  return false;
}

export function settingIsRequired(schema, settings = {}, key) {
  const defn = schema && schema[key];
  if (!defn || typeof defn !== "object") return false;
  if (!gateActive(schema, settings, defn)) return false;
  if (defn.required !== null && defn.required !== undefined) {
    return Boolean(defn.required);
  }
  if (defn.nullable) return false;
  if (CORE_ARTIFACT_TYPES.has(defn.type)) return true;
  if (!hasOwn(defn, "default")) return true;
  // A meaningful default is still a value the user must confirm.  A blank
  // default (an empty text field, empty list, ...) stays optional.
  return !valueIsBlank(defn.default, defn);
}

export function requiredKeys(schema, settings = {}) {
  if (!schema || typeof schema !== "object") return [];
  return Object.keys(schema).filter((key) =>
    settingIsRequired(schema, settings, key),
  );
}

export function settingIsComplete(schema, settings, key) {
  if (!hasOwn(settings, key)) return false;
  return !valueIsBlank(settings[key], (schema && schema[key]) || {});
}

export function completion(schema, settings = {}) {
  const required = requiredKeys(schema, settings);
  const missing = required.filter(
    (key) => !settingIsComplete(schema, settings, key),
  );
  const labels = Object.fromEntries(
    required.map((key) => [key, String(schema[key]?.label || key)]),
  );
  return {
    required,
    missing,
    missing_labels: missing.map((key) => labels[key]),
    done: required.length - missing.length,
    total: required.length,
    percent:
      required.length === 0
        ? 100
        : Math.round(
            ((required.length - missing.length) / required.length) * 100,
          ),
    complete: missing.length === 0,
  };
}

export function applyDefaults(node) {
  const settings = { ...(node.settings || {}) };
  for (const [key, value] of Object.entries(node.defaults || {})) {
    if (!hasOwn(settings, key)) settings[key] = deep(value);
  }
  return {
    ...node,
    settings,
    completion: completion(node.schema || {}, settings),
  };
}

export function clearSetting(node, key) {
  const settings = { ...(node.settings || {}) };
  delete settings[key];
  return {
    ...node,
    settings,
    completion: completion(node.schema || {}, settings),
  };
}

export function nodeComplete(node) {
  return completion(node.schema || {}, node.settings || {}).complete;
}

export function nodeState(node) {
  if (node?.completion?.complete) return "complete";
  if ((node?.completion?.required || []).length) return "incomplete";
  return "empty";
}

export function isCameraNode(node) {
  return node?.type === "camera";
}

export function isUtilityNode(node) {
  return node?.type === "utility";
}

export function nodesForLane(nodes, camera) {
  return nodes.filter(
    (node) => node.type !== "utility" && node.camera === camera,
  );
}

export function laneNodes(nodes, column) {
  return nodes.filter((node) => node.type === column);
}

export function sortedCameras(nodes) {
  return nodes
    .filter((node) => node.type === "camera")
    .map((node, index) => ({ node, index }))
    .sort((a, b) => {
      const laneA = finite(a.node.lane, a.index);
      const laneB = finite(b.node.lane, b.index);
      return laneA - laneB || a.index - b.index;
    })
    .map((item) => item.node);
}

export function sortedUtilities(nodes) {
  return nodes
    .filter((node) => node.type === "utility")
    .map((node, index) => ({ node, index }))
    .sort((a, b) => {
      const orderA = finite(a.node.utility_index, a.index);
      const orderB = finite(b.node.utility_index, b.index);
      return orderA - orderB || a.index - b.index;
    })
    .map((item) => item.node);
}

export function assignLaneIndices(nodes) {
  const cameras = sortedCameras(nodes);
  const laneIndex = new Map();
  cameras.forEach((node, index) => laneIndex.set(node.camera, index));
  for (const node of nodes) {
    if (node.type === "camera") node.lane = laneIndex.get(node.camera) ?? 0;
    else if (node.type !== "utility") {
      node.lane = laneIndex.get(node.camera) ?? finite(node.lane, 0);
    }
  }
  return cameras;
}

// Places every node at its column/lane.  Extra frame processors/trackers in
// the same lane stack inside that lane so legacy multi-node configs remain
// editable until the user removes the duplicates.
export function layoutNodes(nodes) {
  const cameras = assignLaneIndices(nodes);
  const utilities = sortedUtilities(nodes);
  utilities.forEach((node, index) => {
    node.utility_index = index;
  });

  const lanes = [];
  let top = LANE_TOP;
  for (const [index, camera] of cameras.entries()) {
    const fps = nodes.filter(
      (node) => node.type === "frame_processor" && node.camera === camera.camera,
    );
    const trackers = nodes.filter(
      (node) => node.type === "tracker" && node.camera === camera.camera,
    );
    const slots = Math.max(1, fps.length, trackers.length);
    const height = Math.max(
      MIN_LANE_HEIGHT,
      slots * (NODE_H + LANE_SLOT_GAP) + 32,
    );
    const lane = {
      camera: camera.camera,
      label: camera.camera,
      index,
      y: top,
      height,
      name: camera.camera,
    };
    lanes.push(lane);
    camera.lane = index;
    camera.x = COLUMN_X.camera;
    camera.y = top + 16;
    for (const [slot, node] of fps.entries()) {
      node.lane = index;
      node.x = COLUMN_X.frame_processor;
      node.y = top + 16 + slot * (NODE_H + LANE_SLOT_GAP);
    }
    const pipeline = nodes.find(
      (node) => node.type === "pipeline" && node.camera === camera.camera,
    );
    if (pipeline) {
      pipeline.lane = index;
      pipeline.x = COLUMN_X.pipeline;
      pipeline.y = top + 16;
    }
    for (const [slot, node] of trackers.entries()) {
      node.lane = index;
      node.x = COLUMN_X.tracker;
      node.y = top + 16 + slot * (NODE_H + LANE_SLOT_GAP);
    }
    top += height;
  }

  for (const [index, node] of utilities.entries()) {
    node.lane = null;
    node.column = "utility";
    node.utility_index = index;
    node.x = COLUMN_X.utility;
    node.y = UTILITY_TOP + index * UTILITY_GAP;
  }
  for (const node of nodes) {
    node.column = node.type;
    node.w = NODE_W;
    node.h = NODE_H;
    node.completion = completion(node.schema || {}, node.settings || {});
  }

  const height = Math.max(
    top,
    UTILITY_TOP + Math.max(1, utilities.length) * UTILITY_GAP,
  );
  return {
    lanes,
    utilities,
    cameraOrder: cameras.map((node) => node.camera),
    utilityOrder: utilities.map((node) => node.plugin),
    canvas: {
      width: COLUMN_X.utility + NODE_W + 48,
      height: height + 48,
    },
    laneHeight: top,
  };
}

export function edgeState(source, target, edge = {}) {
  if (edge.warning) return "invalid";
  const sourceComplete = Boolean(source?.completion?.complete);
  const targetComplete = Boolean(target?.completion?.complete);
  if (sourceComplete && targetComplete) return "complete";
  if (sourceComplete || targetComplete) return "half";
  return "dashed";
}

function makeEdge(source, target, kind) {
  return {
    id: `edge:${source.id}->${target.id}`,
    from: source.id,
    to: target.id,
    kind,
    lane: source.lane,
    source_column: source.type,
    target_column: target.type,
    state: edgeState(source, target),
  };
}

export function deriveEdges(nodes) {
  const cameras = nodes.filter((node) => node.type === "camera");
  const pipelines = nodes.filter((node) => node.type === "pipeline");
  const processors = nodes.filter((node) => node.type === "frame_processor");
  const trackers = nodes.filter((node) => node.type === "tracker");
  const utilities = nodes.filter((node) => node.type === "utility");
  const edges = [];
  const seen = new Set();
  const add = (source, target, kind) => {
    const key = `${source.id}->${target.id}`;
    if (seen.has(key)) return;
    seen.add(key);
    edges.push(makeEdge(source, target, kind));
  };

  for (const camera of cameras) {
    const lane = camera.lane;
    const fps = processors.filter((node) => node.lane === lane);
    const pipes = pipelines.filter((node) => node.lane === lane);
    const laneTrackers = trackers.filter((node) => node.lane === lane);
    for (const processor of fps) {
      add(camera, processor, "camera_frame_processor");
    }
    for (const processor of fps) {
      for (const pipeline of pipes) {
        add(processor, pipeline, "frame_processor_pipeline");
      }
    }
    for (const pipeline of pipes) {
      for (const tracker of laneTrackers) {
        add(pipeline, tracker, "pipeline_tracker");
      }
    }
  }
  for (const utility of utilities) {
    for (const cameraName of utility.sources || []) {
      const camera = cameras.find((node) => node.camera === cameraName);
      if (!camera) continue;
      for (const tracker of trackers.filter(
        (node) => node.lane === camera.lane,
      )) {
        add(tracker, utility, "tracker_utility");
      }
    }
  }
  return edges;
}

// Edge ids that make up one lane's flow: all five-column edges plus every
// branch from that lane into shared utilities.  Other lanes feeding the same
// utility are intentionally excluded.
export function connectedFlow(edges, edgeId) {
  const edge = edges.find((item) => item.id === edgeId);
  if (!edge) return new Set();
  return new Set(
    edges
      .filter((item) => item.lane === edge.lane)
      .map((item) => item.id),
  );
}

export function edgePath(source, target) {
  if (!source || !target) return "";
  const sourceSize = nodeSize(source.type);
  const targetSize = nodeSize(target.type);
  const sx = finite(source.x, 0) + sourceSize.w;
  const sy = finite(source.y, 0) + sourceSize.h / 2;
  const tx = Math.max(sx + 24, finite(target.x, 0));
  const ty = finite(target.y, 0) + targetSize.h / 2;
  const bend = Math.max(36, (tx - sx) * 0.48);
  return `M ${sx} ${sy} C ${sx + bend} ${sy}, ${tx - bend} ${ty}, ${tx} ${ty}`;
}

export function edgePathForNodes(source, target) {
  return edgePath(source, target);
}

export function portDot(nodes, x, y) {
  return { x: finite(x, 0), y: finite(y, 0) };
}

export function inputPorts(node, nodes) {
  if (!node || node.type === "camera") return [];
  node = nodes?.find((item) => item.id === node.id) || node;
  const size = nodeSize(node.type);
  const y = finite(node.y, 0) + size.h / 2;
  if (node.type !== "utility") {
    return [{ x: finite(node.x, 0), y }];
  }
  const sources = node.sources || [];
  return sources.map((camera, index) => ({
    x: finite(node.x, 0),
    y: finite(node.y, 0) + ((index + 1) * size.h) / (sources.length + 1),
    camera,
    label: camera,
    index,
  }));
}

export function connectionError(source, target, nodes = [], edges = []) {
  if (!source || !target) return "Connect a source port to a target node.";
  if (source.id === target.id) return "A node cannot connect to itself.";
  const sourceIndex = COLUMN_INDEX[source.type];
  const targetIndex = COLUMN_INDEX[target.type];
  if (sourceIndex === undefined || targetIndex === undefined) {
    return "Connection uses an unsupported node type.";
  }
  if (source.type === "utility") {
    return "Utility nodes have no output. Edges must move left to right.";
  }
  if (targetIndex <= sourceIndex) {
    return "Edges must move left to right to the next column.";
  }
  if (targetIndex > sourceIndex + 1) {
    return `Edges cannot skip a column. Connect ${COLUMN_LABELS[source.type]} directly to the next column.`;
  }
  const crossLaneAllowed =
    source.type === "tracker" && target.type === "utility";
  if (source.lane !== target.lane && !crossLaneAllowed) {
    return "Edges may only connect nodes in the same camera lane, except a Tracker -> Utility connection.";
  }
  if (edges.some((edge) => edge.from === source.id && edge.to === target.id)) {
    return "Those nodes are already connected.";
  }
  if (source.type === "frame_processor" && target.type === "pipeline") {
    const supported = source.supported_pipelines || [];
    if (supported.length && target.plugin && !supported.includes(target.plugin)) {
      return `Frame processor '${source.plugin}' does not support pipeline '${target.plugin}'.`;
    }
  }
  if (wouldCreateCycle(edges, source.id, target.id)) {
    return "This connection would create a cycle.";
  }
  return "";
}

export function wouldCreateCycle(edges, from, to) {
  const adjacency = new Map();
  for (const edge of edges || []) {
    if (!adjacency.has(edge.from)) adjacency.set(edge.from, []);
    adjacency.get(edge.from).push(edge.to);
  }
  if (!adjacency.has(from)) adjacency.set(from, []);
  adjacency.get(from).push(to);
  const visiting = new Set();
  const done = new Set();
  const visit = (node) => {
    if (visiting.has(node)) return true;
    if (done.has(node)) return false;
    visiting.add(node);
    for (const next of adjacency.get(node) || []) {
      if (visit(next)) return true;
    }
    visiting.delete(node);
    done.add(node);
    return false;
  };
  return visit(from);
}

export function completionOwner(node) {
  switch (node?.type) {
    case "camera":
      return `Camera '${node.camera}'`;
    case "pipeline":
      return `Pipeline '${node.plugin}' on camera '${node.camera}'`;
    case "frame_processor":
      return `Frame processor '${node.plugin}'`;
    case "tracker":
      return `Tracker '${node.plugin}'`;
    case "utility":
      return `Utility '${node.plugin}'`;
    default:
      return node?.label || node?.id || "Node";
  }
}

export function validateGraph(nodes, edges = []) {
  const errors = [];
  const cameras = nodes.filter((node) => node.type === "camera");
  if (!cameras.length) {
    errors.push({
      code: "no_cameras",
      message: "Add at least one camera lane before saving.",
    });
  }
  const seenNames = new Set();
  for (const camera of cameras) {
    if (seenNames.has(camera.camera)) {
      errors.push({
        code: "duplicate_camera",
        message: `Camera '${camera.camera}' is declared more than once.`,
        node: camera.id,
      });
    }
    seenNames.add(camera.camera);
  }
  const utilities = nodes.filter((node) => node.type === "utility");
  const utilityNames = new Set();
  for (const utility of utilities) {
    if (utilityNames.has(utility.plugin)) {
      errors.push({
        code: "duplicate_utility",
        message: `Utility '${utility.plugin}' can only be placed once.`,
        node: utility.id,
      });
    }
    utilityNames.add(utility.plugin);
  }
  for (const camera of cameras) {
    const lane = camera.lane;
    const laneNodes = nodes.filter(
      (node) => node.type !== "utility" && node.lane === lane,
    );
    for (const type of ["frame_processor", "pipeline", "tracker"]) {
      const matches = laneNodes.filter((node) => node.type === type);
      if (!matches.length) {
        const label = COLUMN_LABELS[type].toLowerCase();
        errors.push({
          code: `lane_missing_${type}`,
          message: `Lane '${camera.camera}' is missing a ${label}. Add one to fill the ${COLUMN_LABELS[type]} column.`,
          node: camera.id,
        });
      } else if (matches.length > 1) {
        errors.push({
          code: "duplicate_column",
          message: `Lane '${camera.camera}' has more than one ${COLUMN_LABELS[type]} node. Keep exactly one node per column.`,
          node: matches[matches.length - 1].id,
        });
      }
    }
    const processors = laneNodes.filter(
      (node) => node.type === "frame_processor",
    );
    const pipeline = laneNodes.find((node) => node.type === "pipeline");
    if (pipeline) {
      for (const processor of processors) {
        const supported = processor.supported_pipelines || [];
        if (
          supported.length &&
          pipeline.plugin &&
          !supported.includes(pipeline.plugin)
        ) {
          errors.push({
            code: "edge_pipeline_unsupported",
            message: `Frame processor '${processor.plugin}' does not support pipeline '${pipeline.plugin}'.`,
            node: processor.id,
          });
        }
      }
    }
    for (const tracker of laneNodes.filter((node) => node.type === "tracker")) {
      if (!utilities.some((utility) => (utility.sources || []).includes(camera.camera))) {
        errors.push({
          code: "tracker_without_utility",
          message: `Tracker '${tracker.plugin}' on lane '${camera.camera}' has no utility output.`,
          node: tracker.id,
        });
      }
    }
  }
  for (const utility of utilities) {
    if (!(utility.sources || []).length) {
      errors.push({
        code: "utility_no_inputs",
        message: `Utility '${utility.plugin}' needs at least one tracker input.`,
        node: utility.id,
      });
    }
  }
  const replay = cameras.filter((node) => node.camera_type === "replay");
  if (replay.length > 1) {
    errors.push({
      code: "camera_replay_limit",
      message: "Only one replay camera is allowed.",
      node: replay[1].id,
    });
  }
  if (edges.length) {
    const byId = new Map(nodes.map((node) => [node.id, node]));
    for (const edge of edges) {
      const source = byId.get(edge.from);
      const target = byId.get(edge.to);
      const message = connectionError(source, target, nodes, []);
      if (message) errors.push({ code: "edge_invalid", message, node: source?.id });
    }
  }
  const completionErrors = nodes
    .filter((node) => !completion(node.schema || {}, node.settings || {}).complete)
    .map((node) => {
      const progress = completion(node.schema || {}, node.settings || {});
      return {
        code: "settings_incomplete",
        message: `${completionOwner(node)} is missing required field(s): ${progress.missing_labels.join(", ")}.`,
        node: node.id,
        missing: progress.missing,
      };
    });
  return [...errors, ...completionErrors];
}

export function cameraNameError(name, nodes, currentName = null) {
  const value = String(name ?? "").trim();
  if (!value) return "Camera name is required.";
  if (value.length > 80) return "Camera names must be 80 characters or less.";
  if (/[\u0000-\u001f]/.test(value)) {
    return "Camera names cannot contain control characters.";
  }
  if (
    nodes.some(
      (node) =>
        node.type === "camera" &&
        node.camera === value &&
        node.camera !== currentName,
    )
  ) {
    return `Camera '${value}' already exists.`;
  }
  return "";
}

export function uniqueCameraName(base, nodes) {
  const used = new Set(
    nodes.filter((node) => node.type === "camera").map((node) => node.camera),
  );
  const clean = String(base || "Camera").trim() || "Camera";
  if (!used.has(clean)) return clean;
  let index = 2;
  while (used.has(`${clean} ${index}`)) index += 1;
  return `${clean} ${index}`;
}

export function renameCameraNodes(nodes, from, to) {
  return nodes.map((node) => {
    if (node.type === "camera" && node.camera === from) {
      return {
        ...node,
        camera: to,
        label: to,
        id: `camera:${to}`,
      };
    }
    if (node.type !== "utility" && node.camera === from) {
      const parts = node.id.split(":");
      if (node.type === "pipeline") {
        return { ...node, camera: to, id: `pipeline:${to}` };
      }
      if (parts.length >= 3) {
        return {
          ...node,
          camera: to,
          id: `${node.type}:${to}:${node.plugin}`,
        };
      }
      return { ...node, camera: to };
    }
    if (node.type === "utility") {
      const sources = (node.sources || []).map((name) =>
        name === from ? to : name,
      );
      return { ...node, sources };
    }
    return node;
  });
}

export function targetColumn(type) {
  return columnForType(type);
}

export function laneForCamera(nodes, camera) {
  return nodes.find((node) => node.type === "camera" && node.camera === camera)
    ?.lane;
}

export function firstEmptyCell(nodes, type, preferredLane = null) {
  if (type === "utility") {
    return nodes.some((node) => node.type === "utility")
      ? null
      : { lane: null, camera: null };
  }
  const cameras = sortedCameras(nodes);
  if (!cameras.length) return null;
  const ordered =
    preferredLane === null
      ? cameras
      : [
          cameras.find((camera) => camera.lane === preferredLane),
          ...cameras.filter((camera) => camera.lane !== preferredLane),
        ].filter(Boolean);
  for (const camera of ordered) {
    const occupied = nodes.some(
      (node) => node.type === type && node.camera === camera.camera,
    );
    if (!occupied) return { lane: camera.lane, camera: camera.camera };
  }
  return null;
}

export function occupiedCell(nodes, type, camera) {
  return nodes.find(
    (node) => node.type === type && node.camera === camera,
  );
}

export function removalPlan(nodes, id) {
  const node = nodes.find((item) => item.id === id);
  if (!node) return { remove: [], nodes };
  if (node.type === "camera") {
    const remove = nodes
      .filter((item) => item.type !== "utility" && item.camera === node.camera)
      .map((item) => item.id);
    let next = nodes.filter((item) => !remove.includes(item.id));
    next = next.map((item) => {
      if (item.type !== "utility") return item;
      return {
        ...item,
        sources: (item.sources || []).filter(
          (camera) => camera !== node.camera,
        ),
      };
    });
    return { remove, nodes: next, camera: node.camera };
  }
  if (node.type === "utility") {
    return { remove: [node.id], nodes: nodes.filter((item) => item.id !== id) };
  }
  let next = nodes.filter((item) => item.id !== id);
  return { remove: [node.id], nodes: next };
}

export function utilitySourceChange(nodes, plugin, sources) {
  const allowed = new Set(
    nodes.filter((node) => node.type === "camera").map((node) => node.camera),
  );
  const clean = [];
  for (const name of sources || []) {
    if (allowed.has(name) && !clean.includes(name)) clean.push(name);
  }
  return nodes.map((node) =>
    node.type === "utility" && node.plugin === plugin
      ? { ...node, sources: clean }
      : node,
  );
}

export function nodePayload(nodes, edges, viewport, layout = {}) {
  const cleanNodes = nodes.map((node) => {
    const payload = {
      id: node.id,
      type: node.type,
      camera: node.camera,
      plugin: node.plugin,
      lane: node.lane,
      camera_type: node.camera_type,
      renamed_from: node.renamed_from,
      settings: deep(node.settings || {}),
      sources: node.type === "utility" ? [...(node.sources || [])] : undefined,
    };
    if (payload.renamed_from === undefined) delete payload.renamed_from;
    if (payload.sources === undefined) delete payload.sources;
    if (payload.camera === undefined) delete payload.camera;
    if (payload.plugin === undefined || payload.plugin === null) {
      delete payload.plugin;
    }
    return payload;
  });
  const cleanEdges = (edges || []).map((edge) => ({
    from: edge.from,
    to: edge.to,
  }));
  const cameraOrder =
    layout.cameraOrder ||
    sortedCameras(nodes).map((node) => node.camera);
  const utilityOrder =
    layout.utilityOrder ||
    sortedUtilities(nodes).map((node) => node.plugin);
  return {
    nodes: cleanNodes,
    edges: cleanEdges,
    layout: {
      viewport: {
        x: finite(viewport?.x, 0),
        y: finite(viewport?.y, 0),
        zoom: clamp(finite(viewport?.zoom, 1), 0.25, 2.5),
      },
      lane_order: cameraOrder.slice(),
      utility_order: utilityOrder.slice(),
    },
  };
}

export function fitViewport(canvas, view, padding = 48) {
  const width = Math.max(1, finite(canvas?.width, 1));
  const height = Math.max(1, finite(canvas?.height, 1));
  const viewWidth = Math.max(1, finite(view?.width, 1));
  const viewHeight = Math.max(1, finite(view?.height, 1));
  const zoom = clamp(
    Math.min(
      (viewWidth - padding * 2) / width,
      (viewHeight - padding * 2) / height,
    ),
    0.25,
    1.4,
  );
  return {
    x: (viewWidth - width * zoom) / 2,
    y: (viewHeight - height * zoom) / 2,
    zoom,
  };
}

export function viewportTransform(viewport) {
  return `translate(${finite(viewport?.x, 0)}px, ${finite(viewport?.y, 0)}px) scale(${finite(viewport?.zoom, 1)})`;
}

export function columnCounts(nodes) {
  return Object.fromEntries(
    COLUMNS.map((column) => [
      column,
      nodes.filter((node) => node.type === column).length,
    ]),
  );
}

export function setUtilitySources(node, sources) {
  return {
    ...node,
    sources: [...new Set((sources || []).filter(Boolean))],
  };
}

export function nextUtilityOrder(nodes) {
  const current = sortedUtilities(nodes);
  return current.length ? current[current.length - 1].utility_index + 1 : 0;
}
