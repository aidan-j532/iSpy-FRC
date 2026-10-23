// Pure graph/geometry helpers for the pipeline editor.  Kept dependency-free
// so the same file can be imported by the browser and exercised with node.

export const ROW_Y = {
  camera: 40,
  frame_processor: 220,
  pipeline: 400,
  tracker: 580,
  utility: 760,
};

export const ROW_LABELS = {
  camera: "Camera",
  frame_processor: "Frame Processors",
  pipeline: "Pipeline",
  tracker: "Trackers",
  utility: "Utilities",
};

export const NODE_SIZES = {
  camera: { w: 206, h: 128 },
  frame_processor: { w: 196, h: 92 },
  pipeline: { w: 214, h: 116 },
  tracker: { w: 190, h: 96 },
  utility: { w: 190, h: 96 },
};

export const EDGE_TARGETS = {
  camera: new Set(["frame_processor", "pipeline"]),
  frame_processor: new Set(["frame_processor", "pipeline"]),
  pipeline: new Set(["tracker", "utility"]),
  tracker: new Set(["tracker", "utility"]),
  utility: new Set(),
};

const CORE_ARTIFACT_TYPES = new Set(["model", "source", "browse"]);

export function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

export function finite(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

export function snap(value, grid = 20) {
  const size = Math.max(1, Number(grid) || 20);
  return Math.round(finite(value, 0) / size) * size;
}

export function rowY(type) {
  return ROW_Y[type] ?? 0;
}

export function nodeSize(type) {
  return NODE_SIZES[type] || NODE_SIZES.utility;
}

export function effectiveValue(schema, settings, key) {
  if (settings && Object.prototype.hasOwnProperty.call(settings, key)) {
    return settings[key];
  }
  const defn = schema && schema[key];
  return defn && Object.prototype.hasOwnProperty.call(defn, "default")
    ? defn.default
    : undefined;
}

export function gateActive(schema, settings, defn) {
  if (!defn || !defn.gated_by) return true;
  const gates = Array.isArray(defn.gated_by) ? defn.gated_by : [defn.gated_by];
  return gates.every((name) => Boolean(effectiveValue(schema, settings, name)));
}

export function settingIsRequired(schema, settings, key) {
  const defn = schema && schema[key];
  if (!defn || typeof defn !== "object") return false;
  if (!gateActive(schema, settings, defn)) return false;
  if (Object.prototype.hasOwnProperty.call(defn, "required")) {
    return Boolean(defn.required);
  }
  if (defn.nullable) return false;
  if (CORE_ARTIFACT_TYPES.has(defn.type)) return true;
  return !Object.prototype.hasOwnProperty.call(defn, "default");
}

export function requiredKeys(schema, settings = {}) {
  if (!schema || typeof schema !== "object") return [];
  return Object.keys(schema).filter((key) =>
    settingIsRequired(schema, settings, key),
  );
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
  if (["source", "browse", "text", "number"].includes(defn.type)) {
    return !String(value).trim();
  }
  if (defn.type === "range") {
    return !Array.isArray(value) || value.length !== 2;
  }
  return false;
}

export function settingIsComplete(schema, settings, key) {
  const defn = (schema && schema[key]) || {};
  return !valueIsBlank(effectiveValue(schema, settings, key), defn);
}

export function completion(schema, settings = {}) {
  const required = requiredKeys(schema, settings);
  const missing = required.filter((key) => !settingIsComplete(schema, settings, key));
  const total = required.length;
  const done = total - missing.length;
  const labels = {};
  for (const key of required) {
    labels[key] = String(schema[key]?.label || key);
  }
  return {
    required,
    missing,
    missing_labels: missing.map((key) => labels[key]),
    done,
    total,
    percent: total === 0 ? 100 : Math.round((done / total) * 100),
    complete: missing.length === 0,
  };
}

export function supportedPipelinesOk(pipeline, supported) {
  if (!supported || !supported.length) return true;
  return supported.includes(pipeline);
}

export function activePipelines(nodes) {
  return new Set(
    nodes
      .filter((node) => node.type === "pipeline" && node.plugin)
      .map((node) => node.plugin),
  );
}

export function connectionError(source, target, nodes = [], edges = []) {
  if (!source || !target) return "Connect a source port to a target node.";
  if (source.id === target.id) return "A node cannot connect to itself.";
  if (!EDGE_TARGETS[source.type]?.has(target.type)) {
    return `Can't connect ${source.type.replaceAll("_", " ")} '${source.label}' to ${target.type.replaceAll("_", " ")} '${target.label}'.`;
  }
  if (
    ["camera", "frame_processor"].includes(source.type) &&
    ["frame_processor", "pipeline"].includes(target.type) &&
    source.camera !== target.camera
  ) {
    return "A frame-processor chain cannot cross cameras.";
  }
  if (
    edges.some((edge) => edge.from === source.id && edge.to === target.id)
  ) {
    return "Those nodes are already connected.";
  }
  const pipelines =
    source.type === "pipeline" ? new Set([source.plugin]) : activePipelines(nodes);
  const supported = target.supported_pipelines || [];
  if (supported.length) {
    const bad = [...pipelines].filter((name) => name && !supported.includes(name));
    if (bad.length) {
      return `${target.type === "tracker" ? "Tracker" : "Utility"} '${target.label}' does not support pipeline(s) ${bad.join(", ")}.`;
    }
  }
  if (wouldCreateCycle(edges, source.id, target.id)) {
    return "That connection would create a cycle.";
  }
  return "";
}

export function wouldCreateCycle(edges, from, to) {
  const adjacency = new Map();
  for (const edge of edges) {
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

export function edgeState(source, target, edge = {}) {
  if (edge.warning) return "invalid";
  if (!source?.completion?.complete) return "dashed";
  if (!target?.completion?.complete) return "half";
  return "complete";
}

export function edgePath(source, target) {
  const a = nodeSize(source.type);
  const b = nodeSize(target.type);
  const sx = source.x + a.w;
  const sy = source.y + a.h / 2;
  const tx = target.x;
  const ty = target.y + b.h / 2;
  const bend = Math.max(48, Math.abs(tx - sx) * 0.48);
  return `M ${sx} ${sy} C ${sx + bend} ${sy}, ${tx - bend} ${ty}, ${tx} ${ty}`;
}

export function deriveEdges(nodes) {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const edges = [];
  const seen = new Set();
  const add = (source, target, kind = "flow", warning = null) => {
    if (!source || !target) return;
    const id = `e:${source.id}->${target.id}`;
    if (seen.has(id)) return;
    seen.add(id);
    edges.push({
      id,
      from: source.id,
      to: target.id,
      kind,
      state: edgeState(source, target, { warning }),
      warning,
    });
  };
  const cameras = nodes.filter((node) => node.type === "camera");
  const pipelines = nodes.filter((node) => node.type === "pipeline");
  const trackers = nodes
    .filter((node) => node.type === "tracker")
    .sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
  const utilities = nodes
    .filter((node) => node.type === "utility")
    .sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
  const pipelineNames = activePipelines(nodes);

  for (const camera of cameras) {
    const chain = [camera];
    const processors = nodes
      .filter(
        (node) =>
          node.type === "frame_processor" && node.camera === camera.camera,
      )
      .sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
    chain.push(...processors);
    const pipeline = pipelines.find((node) => node.camera === camera.camera);
    if (pipeline) chain.push(pipeline);
    for (let i = 0; i < chain.length - 1; i += 1) {
      add(chain[i], chain[i + 1]);
    }
  }

  if (trackers.length) {
    for (const pipeline of pipelines) {
      const warning = supportedPipelinesOk(
        pipeline.plugin,
        trackers[0].supported_pipelines,
      )
        ? null
        : `Tracker '${trackers[0].label}' does not list pipeline '${pipeline.plugin}' as supported.`;
      add(pipeline, trackers[0], "flow", warning);
    }
    for (let i = 0; i < trackers.length - 1; i += 1) {
      const target = trackers[i + 1];
      const bad = [...pipelineNames].filter(
        (name) => !supportedPipelinesOk(name, target.supported_pipelines),
      );
      add(
        trackers[i],
        target,
        "order",
        bad.length
          ? `Tracker '${target.label}' does not list pipeline(s) ${bad.join(", ")} as supported.`
          : null,
      );
    }
  }

  if (utilities.length) {
    const sources = trackers.length ? [trackers[trackers.length - 1]] : pipelines;
    for (const source of sources) {
      for (const utility of utilities) {
        const bad = [...pipelineNames].filter(
          (name) => !supportedPipelinesOk(name, utility.supported_pipelines),
        );
        add(
          source,
          utility,
          "flow",
          bad.length
            ? `Utility '${utility.label}' does not list pipeline(s) ${bad.join(", ")} as supported.`
            : null,
        );
      }
    }
  }
  return edges;
}

export function roundRectPath(w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  return `M ${radius} 0 H ${w - radius} A ${radius} ${radius} 0 0 1 ${w} ${radius} V ${h - radius} A ${radius} ${radius} 0 0 1 ${w - radius} ${h} H ${radius} A ${radius} ${radius} 0 0 1 0 ${h - radius} V ${radius} A ${radius} ${radius} 0 0 1 ${radius} 0 Z`;
}

export function ringPath(type, w, h) {
  if (type === "pipeline") {
    return `M 0 0 H ${w - 20} L ${w} 20 V ${h} H 0 Z`;
  }
  if (type === "tracker") {
    return `M 0 0 H ${w - 20} L ${w} ${h / 2} L ${w - 20} ${h} H 0 Z`;
  }
  if (type === "frame_processor") {
    return roundRectPath(w, h, 28);
  }
  if (type === "utility") {
    return roundRectPath(w, h, 20);
  }
  return roundRectPath(w, h, 18);
}

export function bounds(nodes) {
  if (!nodes.length) {
    return { x: 0, y: 0, w: 1200, h: 900 };
  }
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const node of nodes) {
    const size = nodeSize(node.type);
    minX = Math.min(minX, node.x);
    minY = Math.min(minY, node.y);
    maxX = Math.max(maxX, node.x + size.w);
    maxY = Math.max(maxY, node.y + size.h);
  }
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

export function fitViewport(nodes, width, height, padding = 90, maxZoom = 1.15) {
  const box = bounds(nodes);
  const zoom = clamp(
    Math.min(
      (Math.max(1, width) - padding * 2) / Math.max(1, box.w),
      (Math.max(1, height) - padding * 2) / Math.max(1, box.h),
    ),
    0.25,
    maxZoom,
  );
  return {
    zoom,
    x: width / 2 - (box.x + box.w / 2) * zoom,
    y: height / 2 - (box.y + box.h / 2) * zoom,
  };
}

export function uniqueCameraName(nodes, base = "camera") {
  const used = new Set(
    nodes.filter((node) => node.type === "camera").map((node) => node.camera),
  );
  if (!used.has(base)) return base;
  let index = 2;
  while (used.has(`${base}_${index}`)) index += 1;
  return `${base}_${index}`;
}

export function nextOrder(nodes, predicate) {
  const orders = nodes.filter(predicate).map((node) => Number(node.order) || 0);
  return orders.length ? Math.max(...orders) + 1 : 0;
}

export function nodePayload(nodes, edges, viewport, grid) {
  return {
    nodes: nodes.map((node) => ({
      id: node.id,
      type: node.type,
      label: node.label,
      camera: node.camera,
      plugin: node.plugin,
      camera_type: node.camera_type,
      settings: node.settings,
      x: node.x,
      y: rowY(node.type),
      order: node.order,
    })),
    edges: edges.map((edge) => ({ from: edge.from, to: edge.to })),
    layout: {
      version: 1,
      grid,
      viewport: {
        x: finite(viewport.x, 0),
        y: finite(viewport.y, 0),
        zoom: clamp(finite(viewport.zoom, 1), 0.25, 2.5),
      },
      nodes: Object.fromEntries(
        nodes.map((node) => [
          node.id,
          { x: finite(node.x, 0), y: rowY(node.type) },
        ]),
      ),
    },
  };
}
