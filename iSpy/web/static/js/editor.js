import {
  COLUMNS,
  COLUMN_LABELS,
  COLUMN_X,
  NODE_H,
  NODE_W,
  applyDefaults,
  cameraNameError,
  clamp,
  clearSetting,
  completion,
  connectedFlow,
  deep,
  deriveEdges,
  edgePath,
  firstEmptyCell,
  fitViewport,
  inputPorts,
  layoutNodes,
  nodePayload,
  nodeSize,
  removalPlan,
  renameCameraNodes,
  sortedCameras,
  sortedUtilities,
  uniqueCameraName,
  utilitySourceChange,
  validateGraph,
  valueIsBlank,
} from "./editor_core.js";

const state = {
  nodes: [],
  edges: [],
  lanes: [],
  utilities: [],
  canvas: { width: 1440, height: 800 },
  viewport: { x: 24, y: 24, zoom: 1 },
  palette: {
    cameras: [],
    pipelines: [],
    frame_processors: [],
    trackers: [],
    utilities: [],
    replay_sessions: [],
  },
  selected: new Set(),
  selectedEdge: null,
  history: [],
  future: [],
  dirty: false,
  restartRequired: false,
  busy: false,
  admin: false,
  configFile: "",
  paletteFilter: "",
  paletteDrag: null,
  dropLane: null,
  hoverEdge: null,
  drag: null,
  pan: null,
  marquee: null,
  settingEdit: null,
  serverErrors: [],
  serverWarnings: [],
  localErrors: [],
  viewportDirty: false,
};

const el = {};

const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (char) => {
    return {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[char];
  });

const byId = (id) => state.nodes.find((node) => node.id === id);
const getNodes = (type) => state.nodes.filter((node) => node.type === type);
const selectedNode = () =>
  state.selected.size === 1 ? byId([...state.selected][0]) : null;
const selectedNodes = () =>
  [...state.selected].map((id) => byId(id)).filter(Boolean);

function adminHeaders() {
  const headers = { "Content-Type": "application/json" };
  let token = "";
  try {
    token =
      window.localStorage.getItem("ispyAdminToken") ||
      window.localStorage.getItem("ISPY_ADMIN_TOKEN") ||
      window.localStorage.getItem("adminToken") ||
      "";
  } catch (_error) {
    token = "";
  }
  if (token) headers["X-iSpy-Admin-Token"] = token;
  return headers;
}

function toast(message, type = "info") {
  if (typeof window.showToast === "function") {
    window.showToast(message, type);
  } else if (type === "error") {
    console.error(message);
  } else {
    console.log(message);
  }
}

function hasSetting(node, key) {
  return Object.prototype.hasOwnProperty.call(node.settings || {}, key);
}

function definition(node, key) {
  return node.schema?.[key] || {};
}

function defaultFor(node, key) {
  if (Object.prototype.hasOwnProperty.call(node.defaults || {}, key)) {
    return node.defaults[key];
  }
  const defn = definition(node, key);
  return Object.prototype.hasOwnProperty.call(defn, "default")
    ? defn.default
    : undefined;
}

function hasMeaningfulDefault(node, key) {
  const value = defaultFor(node, key);
  return (
    value !== undefined &&
    !valueIsBlank(value, definition(node, key))
  );
}

function displayValue(value) {
  if (value === undefined) return "";
  if (typeof value === "object" && value !== null) {
    return JSON.stringify(value);
  }
  return String(value);
}

function modelPath(value) {
  if (typeof value === "string") return value;
  if (value && typeof value === "object") {
    return value.file_path || value.source_pt || value.path || "";
  }
  return "";
}

function normalizeNode(raw) {
  const settings = deep(raw.settings || {});
  return {
    ...raw,
    settings,
    schema: deep(raw.schema || {}),
    defaults: deep(raw.defaults || {}),
    sources: Array.isArray(raw.sources) ? [...raw.sources] : [],
    completion: completion(raw.schema || {}, settings),
    warnings: Array.isArray(raw.warnings) ? [...raw.warnings] : [],
  };
}

function updateCompletion(node) {
  node.completion = completion(node.schema || {}, node.settings || {});
  return node;
}

function updateAllCompletions() {
  for (const node of state.nodes) updateCompletion(node);
}

function applyLayout() {
  const result = layoutNodes(state.nodes);
  state.lanes = result.lanes;
  state.utilities = result.utilities;
  state.canvas = result.canvas;
  state.cameraOrder = result.cameraOrder;
  state.utilityOrder = result.utilityOrder;
  state.edges = deriveEdges(state.nodes);
  state.localErrors = validateGraph(state.nodes, state.edges);
  return result;
}

function snapshot() {
  return {
    nodes: deep(state.nodes),
    viewport: deep(state.viewport),
    selected: [...state.selected],
    selectedEdge: state.selectedEdge,
  };
}

function restore(snap) {
  if (!snap) return;
  state.nodes = deep(snap.nodes);
  state.viewport = deep(snap.viewport);
  state.selected = new Set(
    (snap.selected || []).filter((id) =>
      state.nodes.some((node) => node.id === id),
    ),
  );
  state.selectedEdge = snap.selectedEdge || null;
  state.settingEdit = null;
  updateAllCompletions();
  applyLayout();
  state.dirty = true;
  renderAll();
}

function pushHistory(before) {
  if (!before) return;
  state.history.push(before);
  if (state.history.length > 80) state.history.shift();
  state.future = [];
}

function mutate(fn, before = null) {
  const prior = before || snapshot();
  const result = fn();
  if (result === false) return false;
  pushHistory(prior);
  state.dirty = true;
  updateAllCompletions();
  applyLayout();
  renderAll();
  return true;
}

function markViewportDirty() {
  state.viewportDirty = true;
  state.dirty = true;
  updateDirtyText();
}

function updateDirtyText() {
  if (!el.dirty) return;
  el.dirty.textContent = state.dirty
    ? "Unsaved lane changes"
    : state.restartRequired
      ? "Saved - restart required"
      : "Saved";
  el.dirty.classList.toggle("dirty", state.dirty);
}

// ---------------------------------------------------------------- API

async function fetchGraph({ refresh = false, paletteOnly = false } = {}) {
  const response = await fetch(
    `/api/editor/graph${refresh ? "?refresh=1" : ""}`,
    { cache: "no-store" },
  );
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  state.palette = {
    ...state.palette,
    ...(data.palette || {}),
    replay_sessions: data.palette?.replay_sessions || [],
  };
  state.admin = Boolean(data.admin);
  state.configFile = data.config_file || "";
  state.restartRequired = Boolean(
    state.restartRequired || data.restart_required,
  );
  if (paletteOnly) {
    renderPalette();
    return data;
  }
  applyServerGraph(data.graph || {}, { resetHistory: true });
  state.restartRequired = Boolean(data.restart_required);
  renderPalette();
  updateRestartBanner();
  if (!data.graph?.layout_saved) {
    requestAnimationFrame(() => fitGraph(false));
  }
  return data;
}

function applyServerGraph(graph, { resetHistory = false } = {}) {
  state.nodes = (graph.nodes || []).map(normalizeNode);
  state.viewport = {
    x: Number(graph.viewport?.x) || 0,
    y: Number(graph.viewport?.y) || 0,
    zoom: Number(graph.viewport?.zoom) || 1,
  };
  state.selected = new Set();
  state.selectedEdge = null;
  state.settingEdit = null;
  state.serverErrors = [];
  state.serverWarnings = graph.warnings || [];
  state.dirty = false;
  state.viewportDirty = false;
  if (resetHistory) {
    state.history = [];
    state.future = [];
  }
  updateAllCompletions();
  applyLayout();
  renderAll();
  updateDirtyText();
}

async function postGraph(url, label) {
  if (state.busy) return;
  state.busy = true;
  updateToolbar();
  const payload = nodePayload(
    state.nodes,
    state.edges,
    state.viewport,
    { cameraOrder: state.cameraOrder, utilityOrder: state.utilityOrder },
  );
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: adminHeaders(),
      body: JSON.stringify(payload),
    });
    const data = await response.json().catch(() => ({}));
    state.serverErrors = Array.isArray(data.errors) ? data.errors : [];
    state.serverWarnings = Array.isArray(data.warnings) ? data.warnings : [];
    if (!response.ok) {
      if (!state.serverErrors.length) {
        state.serverErrors = [
          {
            code: "http",
            message:
              data.error ||
              data.message ||
              `${label} failed with HTTP ${response.status}.`,
          },
        ];
      }
      renderInspector();
      renderStatus();
      toast(state.serverErrors[0]?.message || `${label} failed.`, "error");
      return false;
    }
    if (data.ok === false) {
      renderInspector();
      renderStatus();
      toast(
        state.serverErrors[0]?.message ||
          data.error ||
          `${label} failed validation.`,
        "error",
      );
      return false;
    }
    state.serverErrors = [];
    state.serverWarnings = data.warnings || [];
    if (label === "Save") {
      if (data.graph) {
        applyServerGraph(data.graph, { resetHistory: true });
      } else {
        state.dirty = false;
      }
      if (data.palette) {
        state.palette = { ...state.palette, ...data.palette };
        renderPalette();
      }
      state.restartRequired = Boolean(data.restart_required);
      updateRestartBanner();
      updateDirtyText();
      toast("Pipeline saved. Restart iSpy to apply changes.", "success");
    } else {
      renderInspector();
      toast(
        state.serverErrors.length
          ? `${state.serverErrors.length} validation issue(s).`
          : "Graph is valid.",
        state.serverErrors.length ? "error" : "success",
      );
    }
    renderAll();
    return true;
  } catch (error) {
    state.serverErrors = [
      { code: "network", message: String(error.message || error) },
    ];
    renderInspector();
    renderStatus();
    toast(String(error.message || error), "error");
    return false;
  } finally {
    state.busy = false;
    updateToolbar();
  }
}

// ---------------------------------------------------------------- render

function renderAll() {
  applyLayout();
  renderBands();
  renderLaneLabels();
  renderHeaders();
  renderNodes();
  renderEdges();
  renderInspector();
  renderStatus();
  updateToolbar();
  updateDirtyText();
  applyViewport();
  renderDropState();
}

function applyViewport() {
  if (!el.world) return;
  const { x, y, zoom } = state.viewport;
  el.world.style.transform = `translate(${x}px, ${y}px) scale(${zoom})`;
  el.world.style.width = `${state.canvas.width}px`;
  el.world.style.height = `${state.canvas.height}px`;
  el.edges.setAttribute("width", String(state.canvas.width));
  el.edges.setAttribute("height", String(state.canvas.height));
  el.edges.setAttribute(
    "viewBox",
    `0 0 ${state.canvas.width} ${state.canvas.height}`,
  );
  if (el.zoomLabel) el.zoomLabel.textContent = `${Math.round(zoom * 100)}%`;
  renderHeaders();
  renderDropState();
}

function renderBands() {
  if (!el.bands) return;
  el.bands.innerHTML = COLUMNS.map(
    (column, index) => `<div class="column-band ${index % 2 ? "alt" : ""}" data-column="${column}" style="left:${COLUMN_X[column] - 14}px;width:${NODE_W + 28}px;height:${state.canvas.height}px"></div>`,
  ).join("");
}

function renderLaneLabels() {
  if (!el.lanes) return;
  el.lanes.innerHTML = state.lanes
    .map(
      (lane, index) => `<div class="lane-label" data-lane="${index}" style="left:${COLUMN_X.camera}px;top:${lane.y - 4}px;width:${NODE_W}px">
        <span class="lane-index">${String(index + 1).padStart(2, "0")}</span>
        <span>${esc(lane.camera)}</span>
      </div>`,
    )
    .join("");
}

function renderHeaders() {
  if (!el.headers) return;
  const counts = {};
  for (const column of COLUMNS) {
    counts[column] = state.nodes.filter(
      (node) => node.type === column,
    ).length;
  }
  el.headers.innerHTML = COLUMNS.map((column) => {
    const left = state.viewport.x + COLUMN_X[column] * state.viewport.zoom;
    const width = NODE_W * state.viewport.zoom;
    return `<div class="column-header" data-column="${column}" style="left:${left}px;width:${width}px">
      <span class="column-name">${COLUMN_LABELS[column]}</span>
      <span class="column-count">${counts[column]}</span>
    </div>`;
  }).join("");
}

function nodeSubtitle(node) {
  if (node.type === "camera") {
    const source = node.settings?.source;
    return `${node.camera_type || "opencv"}${source !== undefined ? ` | ${displayValue(source)}` : ""}`;
  }
  if (node.type === "utility") {
    return `${node.sources?.length || 0} input${node.sources?.length === 1 ? "" : "s"}`;
  }
  if (node.type === "pipeline") return node.plugin || "no pipeline";
  return node.plugin || "";
}

function nodeFooter(node) {
  if (["frame_processor", "tracker"].includes(node.type) && node.shared_count > 1) {
    const names = node.shared_with || [];
    return `<span title="${esc(names.join(", "))}">Shared with ${node.shared_count} lanes</span>`;
  }
  if (node.type === "utility" && node.needs_internet) {
    return "<span>Internet required</span>";
  }
  if (node.type === "camera" && node.calibrated) {
    return "<span>Calibrated</span>";
  }
  return `<span>${esc(node.plugin || node.camera_type || "")}</span>`;
}

function nodeRingPath(w, h, radius = 8) {
  const r = Math.min(radius, w / 2, h / 2);
  return `M ${r} 0 H ${w - r} A ${r} ${r} 0 0 1 ${w} ${r} V ${h - r} A ${r} ${r} 0 0 1 ${w - r} ${h} H ${r} A ${r} ${r} 0 0 1 0 ${h - r} V ${r} A ${r} ${r} 0 0 1 ${r} 0 Z`;
}

function nodeHtml(node) {
  const complete = node.completion?.complete;
  const selected = state.selected.has(node.id);
  const missing = node.completion?.missing || [];
  const percent = Math.max(
    0,
    Math.min(100, Number(node.completion?.percent ?? 100)),
  );
  const width = node.w || NODE_W;
  const height = node.h || NODE_H;
  const ring = nodeRingPath(width, height);
  const utilityPorts =
    node.type === "utility"
      ? (node.sources || []).map((camera, index, list) => {
          const top = ((index + 1) * 100) / (list.length + 1);
          return `<div class="utility-port" style="top:${top}%">
            <span class="port-dot input-dot"></span>
            <span class="port-label">${esc(camera)}</span>
          </div>`;
        }).join("")
      : "";
  const input =
    node.type === "camera"
      ? ""
      : node.type === "utility"
        ? utilityPorts
        : '<span class="node-port input-port" title="Input"></span>';
  const output =
    node.type === "utility"
      ? ""
      : '<span class="node-port output-port" title="Output"></span>';
  const handle =
    node.type === "camera"
      ? '<button class="drag-handle lane-handle" type="button" title="Drag to reorder camera lane" aria-label="Reorder camera lane"><span></span><span></span><span></span><span></span><span></span><span></span></button>'
      : node.type === "utility"
        ? '<button class="drag-handle utility-handle" type="button" title="Drag to reorder utility" aria-label="Reorder utility"><span></span><span></span><span></span><span></span><span></span><span></span></button>'
        : "";
  return `<article class="node node-${node.type} ${selected ? "selected" : ""} ${complete ? "complete" : "incomplete"}" data-node-id="${esc(node.id)}" data-column="${node.type}" style="left:${node.x}px;top:${node.y}px;width:${node.w}px;height:${node.h}px">
    <svg class="node-ring" viewBox="0 0 ${width} ${height}" aria-hidden="true">
      <path class="ring-track" d="${ring}" pathLength="100"></path>
      <path class="ring-progress" d="${ring}" pathLength="100" stroke-dasharray="100" stroke-dashoffset="${100 - percent}"></path>
    </svg>
    <div class="node-accent"></div>
    ${input}${output}
    <header class="node-head">
      <span class="node-kind">${COLUMN_LABELS[node.type]}</span>
      ${handle}
      <span class="node-status ${complete ? "ok" : "needs"}" title="${complete ? "Complete" : `Missing: ${esc(missing.join(", "))}`}"></span>
    </header>
    <div class="node-title">${esc(node.label || node.plugin || node.camera)}</div>
    <div class="node-subtitle">${esc(nodeSubtitle(node))}</div>
    <footer class="node-footer">${nodeFooter(node)}</footer>
  </article>`;
}

function renderNodes() {
  if (!el.nodes) return;
  el.nodes.innerHTML = state.nodes.map(nodeHtml).join("");
  applyDropNodeState();
}

function edgePathWithPorts(edge) {
  const source = byId(edge.from);
  const target = byId(edge.to);
  if (!source || !target) return "";
  const sourceSize = nodeSize(source.type);
  const targetSize = nodeSize(target.type);
  const sx = source.x + sourceSize.w;
  const sy = source.y + sourceSize.h / 2;
  const tx = Math.max(sx + 24, target.x);
  let ty = target.y + targetSize.h / 2;
  if (target.type === "utility") {
    const port = inputPorts(target, state.nodes).find(
      (item) => item.camera === source.camera,
    );
    if (port) ty = port.y;
  }
  const bend = Math.max(36, (tx - sx) * 0.48);
  return `M ${sx} ${sy} C ${sx + bend} ${sy}, ${tx - bend} ${ty}, ${tx} ${ty}`;
}

function renderEdges() {
  if (!el.edges) return;
  const paths = state.edges
    .map((edge) => {
      const source = byId(edge.from);
      const target = byId(edge.to);
      if (!source || !target) return "";
      const selected = state.selectedEdge === edge.id;
      const stateClass = edge.state || "dashed";
      const d = edgePathWithPorts(edge);
      const halfFill =
        stateClass === "half"
          ? `<path class="edge-half-fill ${source.completion?.complete ? "from-source" : "from-target"}" d="${d}" pathLength="100"></path>`
          : "";
      return `<path class="edge-hit" data-edge-id="${esc(edge.id)}" d="${d}"></path>
        <path class="edge ${selected ? "selected" : ""} state-${esc(stateClass)}" data-edge-id="${esc(edge.id)}" d="${d}"></path>
        ${halfFill}`;
    })
    .join("");
  el.edges.innerHTML = paths;
}

function renderInspector() {
  if (!el.inspector) return;
  if (state.selectedEdge) {
    const edge = state.edges.find((item) => item.id === state.selectedEdge);
    if (edge) {
      const source = byId(edge.from);
      const target = byId(edge.to);
      el.inspector.innerHTML = `<div class="inspector-head"><div><span class="eyebrow">Connection</span><h2>${esc(source?.label || "?")} <span class="muted">to</span> ${esc(target?.label || "?")}</h2></div></div>
        <div class="inspector-card"><div class="kv"><span>From</span><b>${COLUMN_LABELS[source?.type] || "?"} | ${esc(source?.label || "?")}</b></div>
        <div class="kv"><span>To</span><b>${COLUMN_LABELS[target?.type] || "?"} | ${esc(target?.label || "?")}</b></div>
        <div class="kv"><span>Lane</span><b>${esc(source?.camera || target?.camera || "global")}</b></div></div>
        <p class="inspector-hint">Connections are generated from the fixed column order and utility inputs. Change a utility's inputs or remove a node to rewire.</p>`;
      return;
    }
    state.selectedEdge = null;
  }
  const nodes = selectedNodes();
  if (!nodes.length) {
    el.inspector.innerHTML = overviewHtml();
    bindInspectorActions();
    return;
  }
  if (nodes.length > 1) {
    el.inspector.innerHTML = multiInspectorHtml(nodes);
    bindInspectorActions();
    return;
  }
  el.inspector.innerHTML = singleInspectorHtml(nodes[0]);
  bindInspectorActions();
}

function errorsForNode(node) {
  return [...state.serverErrors, ...state.localErrors].filter(
    (error) =>
      error.node === node.id ||
      (error.node && node.id.endsWith(`:${error.node}`)),
  );
}

function overviewHtml() {
  const counts = Object.fromEntries(
    COLUMNS.map((column) => [
      column,
      state.nodes.filter((node) => node.type === column).length,
    ]),
  );
  const errors = [...state.serverErrors, ...state.localErrors];
  const warnings = state.serverWarnings;
  return `<div class="inspector-head"><div><span class="eyebrow">Pipeline</span><h2>Lane overview</h2><p class="muted">Camera -> Frame Processor -> Pipeline -> Tracker -> Utility</p></div></div>
    <div class="lane-summary">${COLUMNS.map(
      (column) =>
        `<div class="summary-cell"><span>${COLUMN_LABELS[column]}</span><b>${counts[column]}</b></div>`,
    ).join("")}</div>
    <div class="inspector-card">
      <div class="kv"><span>Cameras</span><b>${counts.camera} lane${counts.camera === 1 ? "" : "s"}</b></div>
      <div class="kv"><span>Utilities</span><b>${counts.utility}</b></div>
      <div class="kv"><span>Restart</span><b>${state.restartRequired ? "Required" : "Not required"}</b></div>
    </div>
    ${errorListHtml(errors, warnings)}
    <div class="inspector-hint">
      <p>Click a node to edit it. Drag the camera handle to reorder lanes. Drag utility handles to reorder global utilities.</p>
      <p>Palette items can be clicked or dragged into a lane. A filled cell is replaced; every change can be undone.</p>
    </div>`;
}

function errorListHtml(errors, warnings = []) {
  if (!errors.length && !warnings.length) {
    return '<div class="inspector-ok">No validation problems.</div>';
  }
  return `<div class="issue-list">${errors
    .map(
      (error) =>
        `<button type="button" class="issue issue-error" data-issue-node="${esc(error.node || "")}"><span class="issue-dot"></span><span>${esc(error.message)}</span></button>`,
    )
    .join("")}${warnings
    .map(
      (warning) =>
        `<button type="button" class="issue issue-warning" data-issue-node="${esc(warning.node || "")}"><span class="issue-dot"></span><span>${esc(warning.message)}</span></button>`,
    )
    .join("")}</div>`;
}

function multiInspectorHtml(nodes) {
  return `<div class="inspector-head"><div><span class="eyebrow">Selection</span><h2>${nodes.length} nodes</h2><p class="muted">Shift-click to add or remove nodes from the selection.</p></div></div>
    <div class="inspector-card">${nodes
      .map(
        (node) =>
          `<div class="kv"><span>${COLUMN_LABELS[node.type]}</span><b>${esc(node.label || node.plugin)}</b></div>`,
      )
      .join("")}</div>
    <div class="inspector-actions"><button data-inspector-action="use-defaults-selection">Use Defaults</button><button class="danger" data-inspector-action="remove-selection">Remove</button></div>`;
}

function singleInspectorHtml(node) {
  const errors = errorsForNode(node);
  const missing = node.completion?.missing || [];
  const shared =
    ["frame_processor", "tracker", "utility"].includes(node.type) &&
    node.shared_count > 1;
  const rename =
    node.type === "camera"
      ? `<label class="field"><span class="field-label">Camera name</span><input type="text" data-camera-rename="${esc(node.camera)}" value="${esc(node.camera)}" /></label>`
      : "";
  const sourcePicker =
    node.type === "utility" ? utilitySourcesHtml(node) : "";
  const actions = `<div class="inspector-actions">
    <button data-inspector-action="use-defaults">Use Defaults</button>
    <button class="danger" data-inspector-action="remove">Remove</button>
  </div>`;
  return `<div class="inspector-head"><div><span class="eyebrow">${COLUMN_LABELS[node.type]}</span><h2>${esc(node.label || node.plugin || node.camera)}</h2>
      <p class="muted">${esc(nodeSubtitle(node))}</p></div>
      <span class="completion-label ${node.completion?.complete ? "ok" : "needs"}">${node.completion?.complete ? "Complete" : `${missing.length} missing`}</span>
    </div>
    ${rename}
    ${shared ? `<div class="shared-note">Settings are shared with ${node.shared_count} lanes${node.shared_with?.length ? `: ${esc(node.shared_with.join(", "))}` : ""}.</div>` : ""}
    ${sourcePicker}
    ${missing.length ? `<div class="missing-note">Missing: ${esc(missing.join(", "))}</div>` : ""}
    ${errorListHtml(errors)}
    ${settingsHtml(node)}
    ${actions}`;
}

function utilitySourcesHtml(node) {
  const cameras = sortedCameras(state.nodes);
  return `<div class="inspector-card utility-sources">
    <div class="card-title"><span>Tracker inputs</span><b>${(node.sources || []).length}</b></div>
    ${
      cameras.length
        ? cameras
            .map(
              (camera) => `<label class="source-check"><input type="checkbox" data-utility-source="${esc(camera.camera)}" ${(node.sources || []).includes(camera.camera) ? "checked" : ""}/><span>${esc(camera.camera)}</span><span class="muted">lane ${camera.lane + 1}</span></label>`,
            )
            .join("")
        : '<p class="muted">Add a camera lane first.</p>'
    }
    <p class="inspector-hint">Each checked lane produces one tracker input and one fixed edge into this utility.</p>
  </div>`;
}

function settingsHtml(node) {
  const entries = Object.entries(node.schema || {});
  if (!entries.length) return '<p class="inspector-hint">This node has no configurable settings.</p>';
  const groups = new Map();
  for (const [key, defn] of entries) {
    const section = String(defn?.section || "Settings");
    if (!groups.has(section)) groups.set(section, []);
    groups.get(section).push([key, defn]);
  }
  return [...groups.entries()]
    .map(
      ([section, fields], index) =>
        `<details class="settings-section" ${index < 2 ? "open" : ""}><summary>${esc(section)}</summary>${fields
          .map(([key, defn]) => fieldHtml(node, key, defn))
          .join("")}</details>`,
    )
    .join("");
}

function optionsFor(node, defn) {
  if (node.type === "camera" && node.camera_type === "replay" && defn?.type === "source") {
    const sessions = state.palette.replay_sessions || [];
    return sessions.map((session) => ({
      value: session.path,
      label: `${session.name}${session.cameras?.length ? ` (${session.cameras.join(", ")})` : ""}`,
    }));
  }
  return (Array.isArray(defn?.options) ? defn.options : []).map((option) => {
    if (option && typeof option === "object") {
      return {
        value: option.value ?? option.name ?? option.id ?? "",
        label: option.label ?? option.name ?? option.value ?? String(option),
      };
    }
    return { value: option, label: String(option) };
  });
}

function fieldHtml(node, key, defn) {
  const explicit = hasSetting(node, key);
  const current = explicit ? node.settings[key] : undefined;
  const fallback = defaultFor(node, key);
  const ghost = !explicit && fallback !== undefined;
  const required = (node.completion?.required || []).includes(key);
  const active = !defn?.gated_by || Boolean(
    (Array.isArray(defn.gated_by) ? defn.gated_by : [defn.gated_by]).every(
      (gate) => node.settings?.[gate] ?? node.defaults?.[gate],
    ),
  );
  const label = String(defn?.label || key);
  const hint = defn?.hint || defn?.help || "";
  const common = `data-setting-key="${esc(key)}" ${ghost ? 'data-ghost="1"' : ""} ${active ? "" : "disabled"} class="${ghost ? "ghost-value" : ""}"`;
  let control = "";
  if (defn?.type === "toggle") {
    const checked = explicit ? Boolean(current) : Boolean(fallback);
    control = `<label class="toggle-row"><input type="checkbox" data-setting-key="${esc(key)}" ${checked ? "checked" : ""} ${ghost ? 'data-ghost="1"' : ""} ${active ? "" : "disabled"}/><span class="toggle"></span><span>${checked ? "On" : "Off"}${ghost ? " | default" : ""}</span></label>`;
  } else if (defn?.type === "select" || (node.type === "camera" && node.camera_type === "replay" && key === "source")) {
    const options = optionsFor(node, defn);
    const defaultOption =
      ghost && fallback !== undefined && fallback !== ""
        ? `<option value="__default__" selected>Default: ${esc(displayValue(fallback))}</option>`
        : ghost
          ? `<option value="__default__" selected>Default: blank</option>`
          : "";
    control = `<select ${common}>${defaultOption}${options
      .map(
        (option) =>
          `<option value="${esc(option.value)}" ${!ghost && String(current) === String(option.value) ? "selected" : ""}>${esc(option.label)}</option>`,
      )
      .join("")}${!ghost && current === undefined ? '<option value="" selected>Choose...</option>' : ""}</select>`;
  } else if (defn?.type === "range") {
    const value = explicit ? current : ["", ""];
    const low = Array.isArray(value) ? value[0] : "";
    const high = Array.isArray(value) ? value[1] : "";
    const disabled = active ? "" : "disabled";
    control = `<div class="range-row ${ghost ? "ghost-value" : ""}" data-setting-key="${esc(key)}"><input type="number" data-range-index="0" value="${esc(low)}" placeholder="${esc(Array.isArray(fallback) ? fallback[0] : "")}" ${disabled} ${ghost ? 'data-ghost="1"' : ""}/><span>to</span><input type="number" data-range-index="1" value="${esc(high)}" placeholder="${esc(Array.isArray(fallback) ? fallback[1] : "")}" ${disabled} ${ghost ? 'data-ghost="1"' : ""}/></div>`;
  } else if (defn?.type === "list") {
    const value = explicit ? current : [];
    control = `<textarea rows="2" ${common} placeholder="${ghost ? `Default: ${esc(displayValue(fallback))}` : ""}">${esc(Array.isArray(value) ? value.map((item) => (typeof item === "object" ? JSON.stringify(item) : item)).join("\n") : displayValue(value))}</textarea>`;
  } else if (defn?.type === "model") {
    const path = explicit ? modelPath(current) : modelPath(fallback);
    control = `<div class="input-row"><input type="text" ${common} value="${explicit ? esc(path) : ""}" placeholder="${ghost ? esc(path || "Choose a model") : "Path to model"}" /><button type="button" class="small" data-browse-model="${esc(key)}" ${active ? "" : "disabled"}>Models</button></div>`;
  } else if (defn?.type === "browse") {
    const path = explicit ? displayValue(current) : "";
    control = `<div class="input-row"><input type="text" ${common} value="${path ? esc(path) : ""}" placeholder="${ghost ? `Default: ${esc(displayValue(fallback))}` : "Folder path"}" /><button type="button" class="small" data-browse-folder="${esc(key)}" ${active ? "" : "disabled"}>Browse</button></div>`;
  } else {
    const type =
      defn?.type === "number"
        ? "number"
        : "text";
    const value = explicit ? displayValue(current) : "";
    const step = defn?.step !== undefined ? `step="${esc(defn.step)}"` : type === "number" ? 'step="any"' : "";
    control = `<input type="${type}" ${common} ${step} value="${esc(value)}" placeholder="${ghost ? `Default: ${esc(displayValue(fallback) || "blank")}` : ""}" />`;
  }
  const wrapper = defn?.type === "toggle" ? "div" : "label";
  return `<${wrapper} class="field ${active ? "" : "gated-off"}"><span class="field-label">${esc(label)}${required ? '<span class="required">required</span>' : ""}${ghost ? '<span class="default-badge">default</span>' : ""}</span>${control}${hint ? `<small class="field-help">${esc(hint)}</small>` : ""}</${wrapper}>`;
}

// ---------------------------------------------------------------- palette

function paletteSection(title, section, items, extra = "") {
  const query = state.paletteFilter.toLowerCase();
  const filtered = items.filter((item) => {
    if (!query) return true;
    return `${item.label || ""} ${item.name || ""} ${item.doc || ""}`
      .toLowerCase()
      .includes(query);
  });
  return `<section class="palette-section"><div class="palette-section-head"><h3>${title}</h3><span>${filtered.length}</span>${extra}</div>${filtered
    .map((item) => paletteItemHtml(item, section))
    .join("")}${
      !filtered.length && query
        ? '<p class="palette-empty">No matches.</p>'
        : ""
    }${section === "cameras" && !items.some((item) => item.kind === "device") ? '<p class="palette-empty">No physical cameras detected. Connect a camera and press Refresh.</p>' : ""}</section>`;
}

function paletteItemHtml(item, section) {
  const type = section === "cameras" ? "camera" : item.type;
  const name = item.name || item.label || "";
  const kind = item.kind || type;
  const isUtility = item.type === "utility";
  const placedUtility =
    isUtility &&
    state.nodes.some(
      (node) => node.type === "utility" && node.plugin === item.name,
    );
  const locked = isUtility && placedUtility;
  const lockReason = locked
    ? "This utility is already placed. Utilities can only be used once."
    : "";
  const detail =
    section === "cameras"
      ? item.detected === false
        ? item.kind === "replay"
          ? "Replay camera | one maximum"
          : "Admin only"
        : `${item.source ?? ""}${item.resolved ? ` | ${item.resolved}` : ""}`
      : item.doc || item.plugin || item.name || "";
  return `<div class="palette-item ${locked ? "locked" : ""}" draggable="${locked ? "false" : "true"}" data-palette-section="${section}" data-palette-name="${esc(name)}" data-palette-kind="${esc(kind)}" data-palette-type="${esc(type)}" title="${esc(lockReason || item.label || name)}">
      <div class="palette-item-main"><span class="palette-name">${esc(item.label || name)}</span><small>${esc(detail)}</small></div>
      <span class="palette-add">${locked ? "Placed" : "Add"}</span>
      ${locked ? `<span class="palette-lock">${esc(lockReason)}</span>` : ""}
    </div>`;
}

function renderPalette() {
  if (!el.paletteList) return;
  el.paletteList.innerHTML = [
    paletteSection(
      "Cameras",
      "cameras",
      state.palette.cameras || [],
      '<button type="button" class="mini" data-palette-refresh title="Probe cameras again">Refresh</button>',
    ),
    paletteSection(
      "Frame Processors",
      "frame_processors",
      state.palette.frame_processors || [],
    ),
    paletteSection("Pipelines", "pipelines", state.palette.pipelines || []),
    paletteSection("Trackers", "trackers", state.palette.trackers || []),
    paletteSection("Utilities", "utilities", state.palette.utilities || []),
  ].join("");
}

function paletteItemFromElement(element) {
  if (!element) return null;
  const section = element.dataset.paletteSection;
  const name = element.dataset.paletteName;
  const allow = {
    cameras: state.palette.cameras,
    frame_processors: state.palette.frame_processors,
    pipelines: state.palette.pipelines,
    trackers: state.palette.trackers,
    utilities: state.palette.utilities,
  }[section];
  const item = (allow || []).find((entry) => entry.name === name);
  if (!item) return null;
  if (section === "cameras") {
    return { ...item, type: "camera", paletteSection: section };
  }
  return { ...item, paletteSection: section };
}

function preferredLane() {
  const selected = selectedNode();
  if (selected && selected.type !== "utility") return selected.lane;
  if (state.selected.size) {
    const laneNode = selectedNodes().find((node) => node.type !== "utility");
    if (laneNode) return laneNode.lane;
  }
  return null;
}

function nearestLane(clientY) {
  if (!state.lanes.length) return 0;
  const rect = el.canvas.getBoundingClientRect();
  const worldY = (clientY - rect.top - state.viewport.y) / state.viewport.zoom;
  let best = 0;
  let distance = Infinity;
  state.lanes.forEach((lane, index) => {
    const center = lane.y + lane.height / 2;
    const current = Math.abs(worldY - center);
    if (current < distance) {
      distance = current;
      best = index;
    }
  });
  if (worldY > state.lanes[state.lanes.length - 1].y + state.lanes[state.lanes.length - 1].height) {
    return state.lanes.length;
  }
  return best;
}

function nodeForPlugin(type, camera) {
  return state.nodes.find(
    (node) => node.type === type && node.camera === camera,
  );
}

function addPaletteItem(item, { lane = null, insertIndex = null } = {}) {
  if (!item) return false;
  if (item.type === "camera") return addCameraItem(item, insertIndex);
  if (item.type === "utility") return addUtilityItem(item);
  return addLanePlugin(item, lane);
}

function addCameraItem(item, insertIndex = null) {
  const before = snapshot();
  const existingReplay = state.nodes.some(
    (node) => node.type === "camera" && node.camera_type === "replay",
  );
  if (item.kind === "replay" && existingReplay) {
    toast("Only one replay camera is allowed.", "error");
    return false;
  }
  if (item.kind === "device") {
    const source = String(item.source ?? "");
    const deviceId = String(item.device_id ?? "");
    const claimed = state.nodes.some(
      (node) =>
        node.type === "camera" &&
        (String(node.settings?.source ?? "") === source ||
          (deviceId && String(node.settings?.device_id ?? "") === deviceId)),
    );
    if (claimed) {
      toast("That camera device is already assigned to a lane.", "error");
      return false;
    }
  }
  if (
    item.camera_type === "tello" &&
    !state.admin
  ) {
    toast("Tello EDU cameras require admin mode.", "error");
    return false;
  }
  const base =
    item.kind === "replay"
      ? "replay"
      : item.kind === "tello"
        ? "tello"
        : item.name || item.device_name || "Camera";
  const name = uniqueCameraName(base, state.nodes);
  const settings = deep(item.settings || {});
  if (item.kind === "replay") {
    const sessions = state.palette.replay_sessions || [];
    if (sessions.length && !settings.source) settings.source = sessions[0].path;
  }
  if (item.kind === "device") {
    settings.source = item.source;
    if (item.device_id !== undefined) settings.device_id = item.device_id;
  }
  const camera = normalizeNode({
    id: `camera:${name}`,
    type: "camera",
    column: "camera",
    camera: name,
    label: name,
    lane: 0,
    camera_type: item.camera_type || "opencv",
    schema: deep(item.schema || {}),
    defaults: deep(item.defaults || {}),
    settings,
    calibrated: false,
    warnings: [],
  });
  const cameras = sortedCameras(state.nodes);
  const index =
    insertIndex === null
      ? cameras.length
      : clamp(insertIndex, 0, cameras.length);
  cameras.splice(index, 0, camera);
  state.nodes = state.nodes.filter((node) => node.type !== "camera" || node === camera);
  for (const [laneIndex, cam] of cameras.entries()) {
    cam.lane = laneIndex;
    if (!state.nodes.includes(cam)) state.nodes.push(cam);
  }
  for (const utility of state.nodes.filter((node) => node.type === "utility")) {
    utility.sources = [...new Set([...(utility.sources || []), name])];
  }
  pushHistory(before);
  state.selected = new Set([camera.id]);
  state.selectedEdge = null;
  state.dirty = true;
  updateAllCompletions();
  applyLayout();
  renderAll();
  toast(`Added camera '${name}'.`, "success");
  return true;
}

function addUtilityItem(item) {
  if (state.nodes.some((node) => node.type === "utility")) {
    const placed = state.nodes.find((node) => node.type === "utility");
    toast(
      `Only one utility can be placed at a time. Remove '${placed.label || placed.plugin}' first.`,
      "error",
    );
    return false;
  }
  const name = item.name || item.plugin;
  const sources = sortedCameras(state.nodes).map((node) => node.camera);
  const utility = normalizeNode({
    id: `utility:${name}`,
    type: "utility",
    column: "utility",
    plugin: name,
    label: item.label || name,
    schema: deep(item.schema || {}),
    defaults: deep(item.defaults || {}),
    settings: deep(item.settings || {}),
    sources,
    utility_index: 0,
    needs_internet: Boolean(item.needs_internet),
  });
  return mutate(() => {
    state.nodes.push(utility);
    state.selected = new Set([utility.id]);
    state.selectedEdge = null;
    return true;
  });
}

function addLanePlugin(item, preferred = null) {
  const type = item.type;
  if (!["frame_processor", "pipeline", "tracker"].includes(type)) return false;
  const cameras = sortedCameras(state.nodes);
  if (!cameras.length) {
    toast("Add a camera lane before adding settings to it.", "error");
    return false;
  }
  let lane = preferred;
  if (lane === null || lane === undefined) {
    const selected = selectedNode();
    lane = selected && selected.type !== "utility" ? selected.lane : null;
  }
  if (lane === null || lane === undefined) {
    const empty = firstEmptyCell(state.nodes, type, null);
    if (!empty) {
      toast(
        `Every lane already has a ${COLUMN_LABELS[type]}. Remove one or add a camera lane.`,
        "error",
      );
      return false;
    }
    lane = empty.lane;
  }
  const camera = state.lanes.find((entry) => entry.index === lane)?.camera ||
    cameras.find((cam) => cam.lane === lane)?.camera;
  if (!camera) {
    toast("That lane no longer exists.", "error");
    return false;
  }
  const existing = nodeForPlugin(type, camera);
  if (existing && existing.plugin === item.name) {
    state.selected = new Set([existing.id]);
    state.selectedEdge = null;
    renderAll();
    toast(`${item.label || item.name} is already in lane '${camera}'.`);
    return false;
  }
  const before = snapshot();
  if (existing) {
    state.nodes = state.nodes.filter((node) => node.id !== existing.id);
  }
  const plugin = item.name;
  const id =
    type === "pipeline"
      ? `pipeline:${camera}`
      : `${type}:${camera}:${plugin}`;
  const node = normalizeNode({
    id,
    type,
    column: type,
    camera,
    lane,
    plugin,
    label: item.label || plugin,
    schema: deep(item.schema || {}),
    defaults: deep(item.defaults || {}),
    settings: deep(item.settings || {}),
    supported_pipelines: item.supported_pipelines || [],
    needs_internet: Boolean(item.needs_internet),
    shared_with: [camera],
    shared_count: 1,
  });
  state.nodes.push(node);
  state.selected = new Set([node.id]);
  state.selectedEdge = null;
  state.dirty = true;
  updateAllCompletions();
  applyLayout();
  pushHistory(before);
  renderAll();
  toast(
    existing
      ? `Replaced the ${COLUMN_LABELS[type]} in lane '${camera}' with ${item.label || plugin}.`
      : `Added ${item.label || plugin} to lane '${camera}'.`,
    "success",
  );
  return true;
}

// ---------------------------------------------------------------- inspector editing

function sharedNodesFor(node) {
  if (!["frame_processor", "tracker"].includes(node.type)) return [];
  return state.nodes.filter(
    (item) => item.type === node.type && item.plugin === node.plugin,
  );
}

function syncSharedSettings(node) {
  const peers = sharedNodesFor(node);
  for (const peer of peers) {
    peer.settings = deep(node.settings);
    updateCompletion(peer);
  }
  if (["frame_processor", "tracker", "utility"].includes(node.type)) {
    const peers = state.nodes.filter(
      (item) => item.type === node.type && item.plugin === node.plugin,
    );
    const names = [...new Set(peers.map((item) => item.camera).filter(Boolean))];
    for (const peer of peers) {
      peer.shared_with = names;
      peer.shared_count = names.length;
    }
  }
}

function writeSetting(node, key, value, { push = true, before = null } = {}) {
  const previous = push ? before || snapshot() : null;
  const defn = definition(node, key);
  let next = value;
  if (defn.type === "number") {
    next = value === "" ? "" : Number(value);
    if (next !== "" && !Number.isFinite(next)) next = value;
  } else if (defn.type === "toggle") {
    next = Boolean(value);
  } else if (defn.type === "list") {
    if (Array.isArray(value)) next = value;
    else {
      const text = String(value ?? "").trim();
      if (!text) next = [];
      else {
        try {
          const parsed = JSON.parse(text);
          next = Array.isArray(parsed) ? parsed : text.split(/\n|,/).map((part) => part.trim()).filter(Boolean);
        } catch (_error) {
          next = text.split(/\n|,/).map((part) => part.trim()).filter(Boolean);
        }
      }
    }
  } else if (defn.type === "model") {
    next =
      typeof value === "object" && value !== null
        ? value
        : {
            file_path: String(value ?? "").trim(),
            source_pt: String(value ?? "").trim(),
          };
  }
  node.settings[key] = next;
  node.completion = completion(node.schema || {}, node.settings);
  syncSharedSettings(node);
  if (push) pushHistory(previous);
  state.dirty = true;
  applyLayout();
  renderNodes();
  renderEdges();
  renderHeaders();
  renderStatus();
  updateDirtyText();
  return next;
}

function removeSetting(node, key) {
  const before = snapshot();
  node.settings = clearSetting(node, key).settings;
  updateCompletion(node);
  syncSharedSettings(node);
  pushHistory(before);
  state.dirty = true;
  applyLayout();
  renderAll();
}

function useDefaultsForNode(node) {
  const before = snapshot();
  const updated = applyDefaults(node);
  node.settings = deep(updated.settings);
  updateCompletion(node);
  syncSharedSettings(node);
  pushHistory(before);
  state.dirty = true;
  updateAllCompletions();
  applyLayout();
  renderAll();
}

function confirmGhost(input, node, key) {
  if (!input || !input.dataset?.ghost) return;
  const fallback = defaultFor(node, key);
  if (fallback === undefined) return;
  const defn = definition(node, key);
  if (defn.type === "range") {
    const row = input.closest(".range-row") || input;
    const values = Array.isArray(fallback) ? fallback : ["", ""];
    writeSetting(node, key, deep(values));
    row.classList.remove("ghost-value");
    row.querySelectorAll("[data-range-index]").forEach((control, index) => {
      delete control.dataset.ghost;
      control.value = displayValue(values[index] ?? "");
    });
    return;
  }
  input.dataset.ghost = "";
  input.classList.remove("ghost-value");
  if (defn.type === "toggle") {
    writeSetting(node, key, Boolean(fallback));
    return;
  }
  writeSetting(node, key, deep(fallback));
  if (defn.type === "model") {
    input.value = modelPath(fallback);
  } else if (defn.type === "list" && Array.isArray(fallback)) {
    input.value = fallback
      .map((item) => (typeof item === "object" ? JSON.stringify(item) : item))
      .join("\n");
  } else {
    input.value = displayValue(fallback);
  }
  if (input.tagName === "SELECT") input.value = String(fallback);
}

function bindInspectorActions() {
  if (!el.inspector) return;
  el.inspector.querySelectorAll("[data-issue-node]").forEach((button) => {
    button.addEventListener("click", () => {
      const node = byId(button.dataset.issueNode);
      if (node) selectNode(node.id, false);
    });
  });
}

function handleInspectorInput(event) {
  const target = event.target;
  const node = selectedNode();
  if (!node) return;
  if (target.matches("[data-camera-rename]")) return;
  const key =
    target.dataset.settingKey ||
    target.closest("[data-setting-key]")?.dataset.settingKey;
  if (!key) return;
  if (target.matches("[data-range-index]")) return;
  if (target.tagName === "SELECT" || target.type === "checkbox") return;
  if (target.dataset.ghost === "1") return;
  // Update live text/number fields without re-rendering the inspector.
  const defn = definition(node, key);
  let value = target.value;
  if (defn.type === "number" && value !== "") {
    const number = Number(value);
    value = Number.isFinite(number) ? number : target.value;
  }
  if (defn.type === "model") {
    value = {
      file_path: String(value || "").trim(),
      source_pt: String(value || "").trim(),
    };
  }
  if (
    !state.settingEdit ||
    state.settingEdit.nodeId !== node.id ||
    state.settingEdit.key !== key
  ) {
    // Record the pre-typing graph immediately so Ctrl+Z can revert the edit
    // even before the field's change event fires.
    const before = snapshot();
    state.settingEdit = { nodeId: node.id, key, before };
    pushHistory(before);
  }
  state.dirty = true;
  node.settings[key] = value;
  updateCompletion(node);
  syncSharedSettings(node);
  updateDirtyText();
}

function handleInspectorChange(event) {
  const target = event.target;
  const node = selectedNode();
  if (!node) return;
  if (target.matches("[data-camera-rename]")) {
    renameCamera(node, target.value);
    return;
  }
  const range = target.closest("[data-range-index]");
  const key =
    target.dataset.settingKey ||
    target.closest("[data-setting-key]")?.dataset.settingKey ||
    range?.closest("[data-setting-key]")?.dataset.settingKey;
  if (!key) return;
  const liveEdit =
    state.settingEdit?.nodeId === node.id &&
    state.settingEdit?.key === key;
  state.settingEdit = null;
  const writeOptions = liveEdit ? { push: false } : {};
  const defn = definition(node, key);
  if (range) {
    const current = hasSetting(node, key)
      ? node.settings[key]
      : defaultFor(node, key);
    const values = Array.isArray(current) ? [...current] : ["", ""];
    values[Number(range.dataset.rangeIndex)] = Number(range.value);
    writeSetting(node, key, values, writeOptions);
    return;
  }
  if (target.type === "checkbox" && defn.type === "toggle") {
    writeSetting(node, key, target.checked, writeOptions);
    const label = target.closest(".toggle-row")?.querySelector("span:last-child");
    if (label) label.textContent = target.checked ? "On" : "Off";
    return;
  }
  if (target.tagName === "SELECT") {
    if (target.value === "__default__") {
      const fallback = defaultFor(node, key);
      writeSetting(node, key, deep(fallback), writeOptions);
      target.value = String(fallback ?? "");
      return;
    }
    writeSetting(node, key, target.value, writeOptions);
    return;
  }
  writeSetting(node, key, target.value, writeOptions);
}

function handleInspectorClick(event) {
  const source = event.target.closest("[data-utility-source]");
  if (source) {
    const node = selectedNode();
    if (node?.type === "utility") {
      const before = snapshot();
      const sources = new Set(node.sources || []);
      if (source.checked) sources.add(source.dataset.utilitySource);
      else sources.delete(source.dataset.utilitySource);
      node.sources = [...sources];
      state.nodes = utilitySourceChange(
        state.nodes,
        node.plugin,
        node.sources,
      );
      pushHistory(before);
      state.dirty = true;
      updateAllCompletions();
      applyLayout();
      renderAll();
    }
    return;
  }
  const modelButton = event.target.closest("[data-browse-model]");
  if (modelButton) {
    const node = selectedNode();
    if (node) openModelPicker(node, modelButton.dataset.browseModel);
    return;
  }
  const browseButton = event.target.closest("[data-browse-folder]");
  if (browseButton) {
    const node = selectedNode();
    if (node) browseForSetting(node, browseButton.dataset.browseFolder);
    return;
  }
  const action = event.target.closest("[data-inspector-action]")?.dataset
    .inspectorAction;
  if (!action) return;
  if (action === "use-defaults") {
    const node = selectedNode();
    if (node) useDefaultsForNode(node);
    return;
  }
  if (action === "use-defaults-selection") {
    useDefaultsForSelection();
    return;
  }
  if (action === "remove") {
    const node = selectedNode();
    if (node) removeNodes([node.id]);
    return;
  }
  if (action === "remove-selection") {
    removeNodes([...state.selected]);
  }
}

function useDefaultsForSelection() {
  const nodes = selectedNodes();
  if (!nodes.length) return;
  const before = snapshot();
  for (const node of nodes) {
    const updated = applyDefaults(node);
    node.settings = deep(updated.settings);
    updateCompletion(node);
    syncSharedSettings(node);
  }
  pushHistory(before);
  state.dirty = true;
  updateAllCompletions();
  applyLayout();
  renderAll();
  toast(`Applied defaults to ${nodes.length} node(s).`, "success");
}

// ---------------------------------------------------------------- node actions

function selectNode(id, additive = false) {
  state.settingEdit = null;
  if (additive) {
    if (state.selected.has(id)) state.selected.delete(id);
    else state.selected.add(id);
  } else {
    state.selected = new Set([id]);
  }
  state.selectedEdge = null;
  renderInspector();
  renderNodes();
}

function removeNodes(ids) {
  if (!ids.length) return;
  const before = snapshot();
  let selected = new Set(ids);
  for (const id of [...selected]) {
    if (!state.nodes.some((node) => node.id === id)) continue;
    const plan = removalPlan(state.nodes, id);
    state.nodes = plan.nodes;
    if (plan.camera) {
      for (const other of [...selected]) {
        if (
          state.nodes.some((node) => node.id === other) &&
          !state.nodes.some(
            (node) => node.type !== "utility" && node.camera === plan.camera,
          )
        ) {
          // keep ids from other lanes; nothing to do here
        }
      }
    }
  }
  state.nodes = state.nodes.map((node) => {
    if (node.type !== "utility") return node;
    return {
      ...node,
      sources: (node.sources || []).filter((camera) =>
        state.nodes.some(
          (item) => item.type === "camera" && item.camera === camera,
        ),
      ),
    };
  });
  state.selected = new Set();
  state.selectedEdge = null;
  pushHistory(before);
  state.dirty = true;
  updateAllCompletions();
  applyLayout();
  renderAll();
  toast(`Removed ${ids.length} node(s).`);
}

function renameCamera(node, rawName) {
  if (!node || node.type !== "camera") return;
  const next = String(rawName ?? "").trim();
  if (next === node.camera) return;
  const error = cameraNameError(next, state.nodes, node.camera);
  if (error) {
    toast(error, "error");
    renderInspector();
    return;
  }
  const before = snapshot();
  const old = node.camera;
  const original = node.renamed_from || old;
  state.nodes = renameCameraNodes(state.nodes, old, next);
  const renamed = state.nodes.find(
    (item) => item.type === "camera" && item.camera === next,
  );
  if (renamed) renamed.renamed_from = original;
  state.selected = new Set([`camera:${next}`]);
  pushHistory(before);
  state.dirty = true;
  updateAllCompletions();
  applyLayout();
  renderAll();
  toast(`Renamed camera '${old}' to '${next}'.`, "success");
}

function reorderCamera(cameraName, targetIndex) {
  const cameras = sortedCameras(state.nodes);
  const index = cameras.findIndex((node) => node.camera === cameraName);
  if (index < 0) return false;
  const [camera] = cameras.splice(index, 1);
  const target = clamp(targetIndex, 0, cameras.length);
  cameras.splice(target, 0, camera);
  cameras.forEach((node, laneIndex) => {
    node.lane = laneIndex;
  });
  return true;
}

function reorderUtility(plugin, targetIndex) {
  const utilities = sortedUtilities(state.nodes);
  const index = utilities.findIndex((node) => node.plugin === plugin);
  if (index < 0) return false;
  const [utility] = utilities.splice(index, 1);
  const target = clamp(targetIndex, 0, utilities.length);
  utilities.splice(target, 0, utility);
  utilities.forEach((node, order) => {
    node.utility_index = order;
  });
  return true;
}

// ---------------------------------------------------------------- drop state

function dragTargetColumn(item) {
  return item?.type === "camera" ? "camera" : item?.type || null;
}

function renderDropState() {
  if (!el.canvas) return;
  const active = Boolean(state.paletteDrag);
  el.canvas.classList.toggle("drop-active", active);
  const target = dragTargetColumn(state.paletteDrag);
  for (const band of el.bands?.querySelectorAll(".column-band") || []) {
    band.classList.toggle("drop-target", active && band.dataset.column === target);
    band.classList.toggle("drop-dim", active && band.dataset.column !== target);
  }
  for (const header of el.headers?.querySelectorAll(".column-header") || []) {
    header.classList.toggle("drop-target", active && header.dataset.column === target);
    header.classList.toggle("drop-dim", active && header.dataset.column !== target);
  }
  applyDropNodeState();
  if (el.dropIndicator) {
    if (
      active &&
      state.paletteDrag.type !== "camera" &&
      state.paletteDrag.type !== "utility" &&
      state.dropLane !== null &&
      state.lanes[state.dropLane]
    ) {
      const lane = state.lanes[state.dropLane];
      el.dropIndicator.hidden = false;
      el.dropIndicator.style.left = `${COLUMN_X[target] - 14}px`;
      el.dropIndicator.style.top = `${lane.y + 8}px`;
      el.dropIndicator.style.width = `${NODE_W + 28}px`;
      el.dropIndicator.style.height = `${Math.max(1, Math.min(lane.height - 16, NODE_H + 16))}px`;
    } else if (
      active &&
      state.paletteDrag.type === "camera" &&
      state.dropLane !== null
    ) {
      const lane = state.lanes[Math.min(state.dropLane, state.lanes.length - 1)];
      el.dropIndicator.hidden = false;
      el.dropIndicator.style.left = "18px";
      el.dropIndicator.style.top = `${(lane?.y ?? 104) - 4}px`;
      el.dropIndicator.style.width = `${state.canvas.width - 36}px`;
      el.dropIndicator.style.height = "8px";
    } else {
      el.dropIndicator.hidden = true;
    }
  }
}

function applyDropNodeState() {
  if (!el.nodes) return;
  const active = Boolean(state.paletteDrag);
  const target = dragTargetColumn(state.paletteDrag);
  el.nodes.querySelectorAll(".node").forEach((item) => {
    item.classList.toggle(
      "drop-target",
      active && item.dataset.column === target,
    );
    item.classList.toggle(
      "drop-dim",
      active && item.dataset.column !== target,
    );
  });
}

// ---------------------------------------------------------------- pointer / wheel

function canvasPoint(event) {
  const rect = el.canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
    worldX:
      (event.clientX - rect.left - state.viewport.x) / state.viewport.zoom,
    worldY:
      (event.clientY - rect.top - state.viewport.y) / state.viewport.zoom,
  };
}

function startPan(event) {
  state.pan = {
    pointerId: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    viewport: deep(state.viewport),
    before: snapshot(),
    moved: false,
  };
  el.canvas.classList.add("panning");
  el.canvas.setPointerCapture?.(event.pointerId);
}

function startLaneDrag(event, node) {
  state.drag = {
    kind: "lane",
    camera: node.camera,
    startY: event.clientY,
    before: snapshot(),
    changed: false,
  };
  event.preventDefault();
  el.canvas.setPointerCapture?.(event.pointerId);
}

function startUtilityDrag(event, node) {
  state.drag = {
    kind: "utility",
    plugin: node.plugin,
    startY: event.clientY,
    before: snapshot(),
    changed: false,
  };
  event.preventDefault();
  el.canvas.setPointerCapture?.(event.pointerId);
}

function onPointerDown(event) {
  if (event.button !== 0 && event.button !== 1) return;
  if (state.busy) return;
  const stepping = event.target.closest(".node");
  const handle = event.target.closest(".lane-handle");
  if (handle && stepping) {
    startLaneDrag(event, byId(stepping.dataset.nodeId));
    return;
  }
  const utilityHandle = event.target.closest(".utility-handle");
  if (utilityHandle && stepping) {
    startUtilityDrag(event, byId(stepping.dataset.nodeId));
    return;
  }
  if (stepping) {
    selectNode(stepping.dataset.nodeId, event.shiftKey);
    return;
  }
  if (event.shiftKey) {
    const point = canvasPoint(event);
    state.marquee = {
      startX: point.x,
      startY: point.y,
      currentX: point.x,
      currentY: point.y,
      before: [...state.selected],
    };
    el.marquee.hidden = false;
    el.marquee.style.left = `${point.x}px`;
    el.marquee.style.top = `${point.y}px`;
    el.marquee.style.width = "0px";
    el.marquee.style.height = "0px";
    el.canvas.setPointerCapture?.(event.pointerId);
    return;
  }
  if (event.target === el.canvas || event.target === el.world || event.target === el.bands || event.target === el.lanes || event.target.closest(".column-band")) {
    state.selected = new Set();
    state.selectedEdge = null;
    renderNodes();
    renderInspector();
    startPan(event);
  }
}

function onPointerMove(event) {
  if (state.pan) {
    const dx = event.clientX - state.pan.startX;
    const dy = event.clientY - state.pan.startY;
    if (dx || dy) state.pan.moved = true;
    state.viewport.x = state.pan.viewport.x + dx;
    state.viewport.y = state.pan.viewport.y + dy;
    applyViewport();
    return;
  }
  if (state.marquee) {
    const point = canvasPoint(event);
    state.marquee.currentX = point.x;
    state.marquee.currentY = point.y;
    const left = Math.min(state.marquee.startX, point.x);
    const top = Math.min(state.marquee.startY, point.y);
    el.marquee.style.left = `${left}px`;
    el.marquee.style.top = `${top}px`;
    el.marquee.style.width = `${Math.abs(point.x - state.marquee.startX)}px`;
    el.marquee.style.height = `${Math.abs(point.y - state.marquee.startY)}px`;
    return;
  }
  if (state.drag?.kind === "lane") {
    const index = nearestLane(event.clientY);
    if (reorderCamera(state.drag.camera, index)) {
      state.drag.changed = true;
      updateAllCompletions();
      applyLayout();
      renderBands();
      renderLaneLabels();
      renderHeaders();
      renderNodes();
      renderEdges();
    }
    return;
  }
  if (state.drag?.kind === "utility") {
    const rect = el.canvas.getBoundingClientRect();
    const worldY =
      (event.clientY - rect.top - state.viewport.y) / state.viewport.zoom;
    const utilityNodes = sortedUtilities(state.nodes);
    let best = utilityNodes.length - 1;
    let distance = Infinity;
    utilityNodes.forEach((node, index) => {
      const current = Math.abs(worldY - (node.y + NODE_H / 2));
      if (current < distance) {
        distance = current;
        best = index;
      }
    });
    if (reorderUtility(state.drag.plugin, best)) {
      state.drag.changed = true;
      updateAllCompletions();
      applyLayout();
      renderBands();
      renderLaneLabels();
      renderHeaders();
      renderNodes();
      renderEdges();
    }
  }
}

function onPointerUp(event) {
  if (state.pan) {
    const pan = state.pan;
    state.pan = null;
    el.canvas.classList.remove("panning");
    if (pan.moved) {
      pushHistory(pan.before);
      markViewportDirty();
    }
    return;
  }
  if (state.marquee) {
    const marquee = state.marquee;
    state.marquee = null;
    el.marquee.hidden = true;
    const left = Math.min(marquee.startX, marquee.currentX);
    const right = Math.max(marquee.startX, marquee.currentX);
    const top = Math.min(marquee.startY, marquee.currentY);
    const bottom = Math.max(marquee.startY, marquee.currentY);
    const selected = new Set(marquee.before || []);
    for (const node of state.nodes) {
      const x = state.viewport.x + node.x * state.viewport.zoom;
      const y = state.viewport.y + node.y * state.viewport.zoom;
      const width = node.w * state.viewport.zoom;
      const height = node.h * state.viewport.zoom;
      if (x < right && x + width > left && y < bottom && y + height > top) {
        selected.add(node.id);
      }
    }
    state.selected = selected;
    state.selectedEdge = null;
    renderNodes();
    renderInspector();
    return;
  }
  if (state.drag) {
    const drag = state.drag;
    state.drag = null;
    if (drag.changed) {
      pushHistory(drag.before);
      state.dirty = true;
      updateAllCompletions();
      applyLayout();
      renderAll();
      toast(
        drag.kind === "lane"
          ? "Lane order updated."
          : "Utility order updated.",
        "success",
      );
    }
  }
}

function onWheel(event) {
  if (!el.canvas) return;
  event.preventDefault();
  if (event.ctrlKey || event.metaKey) {
    const point = canvasPoint(event);
    const worldX =
      (point.x - state.viewport.x) / state.viewport.zoom;
    const worldY =
      (point.y - state.viewport.y) / state.viewport.zoom;
    const zoom = clamp(
      state.viewport.zoom * (event.deltaY < 0 ? 1.1 : 0.9),
      0.25,
      2.5,
    );
    state.viewport.zoom = zoom;
    state.viewport.x = point.x - worldX * zoom;
    state.viewport.y = point.y - worldY * zoom;
  } else if (event.shiftKey) {
    state.viewport.x -= event.deltaY || event.deltaX;
  } else {
    state.viewport.y -= event.deltaY;
    state.viewport.x -= event.deltaX;
  }
  markViewportDirty();
  applyViewport();
}

// ---------------------------------------------------------------- palette events

function onPaletteClick(event) {
  if (event.target.closest("[data-palette-refresh]")) {
    refreshDevices();
    return;
  }
  const item = event.target.closest(".palette-item");
  if (!item || item.classList.contains("locked")) return;
  const data = paletteItemFromElement(item);
  if (data) addPaletteItem(data, { lane: null });
}

function onPaletteDragStart(event) {
  const item = event.target.closest(".palette-item");
  if (!item || item.classList.contains("locked")) {
    event.preventDefault();
    return;
  }
  const data = paletteItemFromElement(item);
  if (!data) {
    event.preventDefault();
    return;
  }
  state.paletteDrag = data;
  state.dropLane = null;
  event.dataTransfer.effectAllowed = "copy";
  event.dataTransfer.setData("text/plain", item.dataset.paletteName || "");
  requestAnimationFrame(renderDropState);
}

function onPaletteDragEnd() {
  state.paletteDrag = null;
  state.dropLane = null;
  renderDropState();
}

function onCanvasDragOver(event) {
  if (!state.paletteDrag) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = "copy";
  state.dropLane = nearestLane(event.clientY);
  renderDropState();
}

function onCanvasDrop(event) {
  if (!state.paletteDrag) return;
  event.preventDefault();
  const item = state.paletteDrag;
  const lane = state.dropLane;
  state.paletteDrag = null;
  state.dropLane = null;
  renderDropState();
  addPaletteItem(item, {
    lane: item.type === "camera" ? null : lane,
    insertIndex: item.type === "camera" ? lane : null,
  });
}

async function refreshDevices() {
  try {
    await fetchGraph({ refresh: true, paletteOnly: true });
    toast("Camera devices refreshed.", "success");
  } catch (error) {
    toast(String(error.message || error), "error");
  }
}

// ---------------------------------------------------------------- toolbar

function updateToolbar() {
  if (el.undo) el.undo.disabled = state.busy || !state.history.length;
  if (el.redo) el.redo.disabled = state.busy || !state.future.length;
  if (el.save) {
    el.save.disabled = state.busy;
    el.save.textContent = state.busy ? "Working..." : "Save";
  }
  if (el.validate) el.validate.disabled = state.busy;
  if (el.useDefaultsAll) el.useDefaultsAll.disabled = state.busy;
}

function renderStatus() {
  if (!el.statusLeft) return;
  const errors = [...state.serverErrors, ...state.localErrors];
  el.statusLeft.textContent = state.busy
    ? "Working..."
    : errors.length
      ? `${errors.length} validation issue${errors.length === 1 ? "" : "s"}`
      : state.dirty
        ? "Unsaved changes"
        : "Ready";
  el.statusLeft.classList.toggle("has-errors", errors.length > 0);
  if (el.statusMiddle) {
    el.statusMiddle.textContent = `${getNodes("camera").length} lane${getNodes("camera").length === 1 ? "" : "s"} | ${state.nodes.length} nodes | ${state.edges.length} edges`;
  }
  if (el.statusRight) {
    el.statusRight.textContent = state.configFile || "";
    el.statusRight.title = state.configFile || "";
  }
}

function updateRestartBanner() {
  if (!el.restartBanner) return;
  el.restartBanner.hidden = !state.restartRequired;
}

function fitGraph(push = true) {
  const view = el.canvas.getBoundingClientRect();
  const before = snapshot();
  state.viewport = fitViewport(state.canvas, {
    width: view.width,
    height: view.height,
  });
  if (push) {
    pushHistory(before);
    markViewportDirty();
  } else {
    state.viewportDirty = false;
  }
  applyViewport();
}

function zoomBy(factor) {
  const view = el.canvas.getBoundingClientRect();
  const before = snapshot();
  const oldZoom = state.viewport.zoom;
  const worldX = (view.width / 2 - state.viewport.x) / state.viewport.zoom;
  const worldY = (view.height / 2 - state.viewport.y) / state.viewport.zoom;
  state.viewport.zoom = clamp(state.viewport.zoom * factor, 0.25, 2.5);
  state.viewport.x = view.width / 2 - worldX * state.viewport.zoom;
  state.viewport.y = view.height / 2 - worldY * state.viewport.zoom;
  if (state.viewport.zoom !== oldZoom) {
    pushHistory(before);
    markViewportDirty();
  }
  applyViewport();
}

function undo() {
  if (!state.history.length) return;
  const current = snapshot();
  const snap = state.history.pop();
  state.future.push(current);
  restore(snap);
  updateToolbar();
}

function redo() {
  if (!state.future.length) return;
  const current = snapshot();
  const snap = state.future.pop();
  state.history.push(current);
  restore(snap);
  updateToolbar();
}

async function reloadGraph() {
  if (state.dirty && !(await confirmAction("Discard unsaved changes?", "The lane layout has unsaved edits. Reloading from the saved config will discard them.", "Reload", "danger"))) {
    return;
  }
  try {
    await fetchGraph();
    toast("Reloaded from config.", "success");
  } catch (error) {
    toast(String(error.message || error), "error");
  }
}

async function useDefaultsAll() {
  if (!state.nodes.length) return;
  const before = snapshot();
  for (const node of state.nodes) {
    const updated = applyDefaults(node);
    node.settings = deep(updated.settings);
    updateCompletion(node);
    syncSharedSettings(node);
  }
  pushHistory(before);
  state.dirty = true;
  updateAllCompletions();
  applyLayout();
  renderAll();
  toast(`Applied defaults to all ${state.nodes.length} nodes.`, "success");
}

// ---------------------------------------------------------------- modal

function closeModal() {
  if (el.modalRoot) el.modalRoot.innerHTML = "";
}

function openModal(title, bodyHtml, actions = []) {
  if (!el.modalRoot) return { root: document.createElement("div"), body: document.createElement("div"), close: closeModal };
  el.modalRoot.innerHTML = `<div class="editor-modal-backdrop"><div class="editor-modal" role="dialog" aria-modal="true"><header><h2>${esc(title)}</h2><button type="button" class="modal-close" aria-label="Close">x</button></header><div class="modal-body">${bodyHtml}</div><footer>${actions
    .map(
      (action, index) =>
        `<button type="button" data-modal-action="${index}" class="${action.primary ? "primary" : ""} ${action.danger ? "danger" : ""}">${esc(action.label)}</button>`,
    )
    .join("")}</footer></div></div>`;
  const root = el.modalRoot;
  const body = root.querySelector(".modal-body");
  const close = () => {
    root.innerHTML = "";
  };
  const closeButton = root.querySelector(".modal-close");
  if (closeButton) closeButton.addEventListener("click", close);
  root.querySelectorAll("[data-modal-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const action = actions[Number(button.dataset.modalAction)];
      if (!action?.onClick || action.onClick() !== false) {
        if (action?.keepOpen) return;
        close();
      }
    });
  });
  return { root, body, close, modal: root.querySelector(".editor-modal") };
}

function confirmAction(title, message, confirmLabel = "Continue", tone = "") {
  return new Promise((resolve) => {
    let settled = false;
    const modal = openModal(
      title,
      `<p>${esc(message)}</p>`,
      [
        {
          label: "Cancel",
          onClick: () => {
            settled = true;
            resolve(false);
            return true;
          },
        },
        {
          label: confirmLabel,
          primary: tone !== "danger",
          danger: tone === "danger",
          onClick: () => {
            settled = true;
            resolve(true);
            return true;
          },
        },
      ],
    );
    const closeButton = modal.root.querySelector(".modal-close");
    if (closeButton) {
      closeButton.addEventListener("click", () => {
        if (!settled) {
          settled = true;
          resolve(false);
        }
      });
    }
  });
}

async function browseForSetting(node, key) {
  const modal = openModal(
    `Choose folder for ${definition(node, key).label || key}`,
    "<p>Loading folders...</p>",
    [{ label: "Cancel" }],
  );
  let currentPath = String(node.settings?.[key] || "");
  const load = async (path) => {
    modal.body.innerHTML = "<p>Loading folders...</p>";
    try {
      const response = await fetch(
        `/api/fs/dirs?path=${encodeURIComponent(path || "")}`,
        { cache: "no-store" },
      );
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      currentPath = data.path || path || "";
      const dirs = Array.isArray(data.dirs) ? data.dirs : [];
      modal.body.innerHTML = `<div class="modal-path"><b>${esc(data.name || currentPath || "Folder")}</b><small>${esc(currentPath)}</small></div>
        <div class="picker-list">${data.parent ? `<button type="button" class="picker-item" data-browse-parent><span><b>Parent folder</b><small>${esc(data.parent)}</small></span></button>` : ""}${dirs
          .map(
            (dir) =>
              `<button type="button" class="picker-item" data-browse-path="${esc(dir.path)}"><span><b>${esc(dir.name)}</b><small>${esc(dir.path)}</small></span></button>`,
          )
          .join("")}</div>`;
      modal.body.querySelectorAll("[data-browse-path]").forEach((button) => {
        button.addEventListener("click", () => load(button.dataset.browsePath));
      });
      const parent = modal.body.querySelector("[data-browse-parent]");
      if (parent) parent.addEventListener("click", () => load(data.parent));
    } catch (error) {
      modal.body.innerHTML = `<div class="issue issue-error">${esc(error.message)}</div>`;
    }
  };
  const footer = modal.root.querySelector(".modal-footer, footer");
  if (footer) {
    const use = document.createElement("button");
    use.type = "button";
    use.className = "primary";
    use.textContent = "Use this folder";
    use.addEventListener("click", () => {
      if (!currentPath) return;
      writeSetting(node, key, currentPath);
      modal.close();
    });
    footer.prepend(use);
  }
  load(currentPath);
}

async function openModelPicker(node, key) {
  const modal = openModal("Choose model", "<p>Loading models...</p>", [
    { label: "Cancel" },
  ]);
  try {
    const response = await fetch("/api/models", { cache: "no-store" });
    const data = await response.json();
    const models = data.models || [];
    modal.body.innerHTML = models.length
      ? `<div class="picker-list">${models
          .map(
            (model) =>
              `<button type="button" class="picker-item" data-model-name="${esc(model.name)}"><span><b>${esc(model.name)}</b><small>${esc(model.task || "model")} | ${esc(model.size_mb ?? "?")} MB</small></span></button>`,
          )
          .join("")}</div>`
      : '<p>No models found. Upload one on the Models page.</p><p><a href="/models">Open Models</a></p>';
    modal.body.querySelectorAll("[data-model-name]").forEach((button) => {
      button.addEventListener("click", () => {
        const name = button.dataset.modelName;
        const path = `YoloModels/pytorch/${name}`;
        writeSetting(node, key, { file_path: path, source_pt: path });
        modal.close();
        renderInspector();
        toast(`Selected ${name}.`, "success");
      });
    });
  } catch (_error) {
    modal.body.innerHTML = "<p>Could not load models. Enter the path manually.</p>";
  }
}

async function openShortcuts() {
  openModal(
    "Keyboard shortcuts",
    `<div class="shortcut-grid">
      <span><kbd>Ctrl</kbd> + <kbd>S</kbd></span><b>Save</b>
      <span><kbd>Ctrl</kbd> + <kbd>Z</kbd></span><b>Undo</b>
      <span><kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>Z</kbd></span><b>Redo</b>
      <span><kbd>F</kbd></span><b>Fit graph</b>
      <span><kbd>Shift</kbd> + drag</span><b>Marquee select</b>
      <span><kbd>Delete</kbd></span><b>Remove selected nodes</b>
    </div><p class="inspector-hint">Nodes cannot be moved freely. Drag the camera handle to reorder lanes, or a utility handle to reorder utilities. Scroll vertically to pan, Shift+scroll horizontally, and Ctrl+scroll to zoom.</p>`,
    [{ label: "Close", primary: true }],
  );
}

// ---------------------------------------------------------------- startup / events

function bindEvents() {
  el.canvas.addEventListener("pointerdown", onPointerDown);
  el.canvas.addEventListener("wheel", onWheel, { passive: false });
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);
  el.canvas.addEventListener("dragover", onCanvasDragOver);
  el.canvas.addEventListener("drop", onCanvasDrop);
  el.canvas.addEventListener("dragleave", (event) => {
    if (!el.canvas.contains(event.relatedTarget)) {
      state.dropLane = null;
      renderDropState();
    }
  });
  el.canvas.addEventListener("click", (event) => {
    const edge = event.target.closest("[data-edge-id]");
    if (edge) {
      state.selectedEdge = edge.dataset.edgeId;
      state.selected = new Set();
      renderInspector();
      renderNodes();
      renderEdges();
    }
  });
  el.canvas.addEventListener("mouseover", (event) => {
    const path = event.target.closest("[data-edge-id]");
    if (!path) return;
    state.hoverEdge = path.dataset.edgeId;
    highlightEdge(path.dataset.edgeId);
  });
  el.canvas.addEventListener("mouseout", (event) => {
    const path = event.target.closest("[data-edge-id]");
    if (!path) return;
    state.hoverEdge = null;
    highlightEdge(null);
  });
  el.paletteList.addEventListener("click", onPaletteClick);
  el.paletteList.addEventListener("dragstart", onPaletteDragStart);
  el.paletteList.addEventListener("dragend", onPaletteDragEnd);
  el.paletteSearch?.addEventListener("input", () => {
    state.paletteFilter = el.paletteSearch.value;
    renderPalette();
  });
  el.inspector.addEventListener("input", handleInspectorInput);
  el.inspector.addEventListener("change", handleInspectorChange);
  el.inspector.addEventListener("click", handleInspectorClick);
  el.inspector.addEventListener("focusin", (event) => {
    const node = selectedNode();
    if (!node) return;
    const target = event.target;
    if (target.dataset?.ghost === "1") {
      const key =
        target.dataset.settingKey ||
        target.closest("[data-setting-key]")?.dataset.settingKey;
      if (key) confirmGhost(target, node, key);
    }
  });
  el.inspector.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    const target = event.target;
    if (target.matches("[data-camera-rename]")) {
      event.preventDefault();
      target.blur();
    } else if (target.matches("input[type=text], input[type=number]")) {
      event.preventDefault();
      target.blur();
    }
  });
  el.btnUndo?.addEventListener("click", undo);
  el.btnRedo?.addEventListener("click", redo);
  el.btnFit?.addEventListener("click", () => fitGraph(true));
  el.btnReload?.addEventListener("click", reloadGraph);
  el.btnValidate?.addEventListener("click", () => postGraph("/api/editor/validate", "Validate"));
  el.btnSave?.addEventListener("click", () => postGraph("/api/editor/graph", "Save"));
  el.btnUseDefaultsAll?.addEventListener("click", useDefaultsAll);
  el.btnZoomIn?.addEventListener("click", () => zoomBy(1.15));
  el.btnZoomOut?.addEventListener("click", () => zoomBy(1 / 1.15));
  el.btnShortcuts?.addEventListener("click", openShortcuts);
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("beforeunload", (event) => {
    if (!state.dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });
  window.addEventListener("resize", () => {
    renderHeaders();
    renderDropState();
  });
}

function highlightEdge(edgeId) {
  if (!el.edges) return;
  const flow = edgeId ? connectedFlow(state.edges, edgeId) : new Set();
  el.edges.querySelectorAll("[data-edge-id]").forEach((path) => {
    path.classList.toggle("flow", flow.has(path.dataset.edgeId));
    path.classList.toggle("flow-origin", path.dataset.edgeId === edgeId);
  });
  if (el.nodes) {
    el.nodes.querySelectorAll(".node").forEach((node) => node.classList.remove("port-hover"));
  }
  if (edgeId) {
    const edge = state.edges.find((item) => item.id === edgeId);
    for (const nodeId of [edge?.from, edge?.to]) {
      if (!nodeId) continue;
      const node = el.nodes.querySelector(
        `[data-node-id="${CSS.escape(nodeId)}"]`,
      );
      node?.classList.add("port-hover");
    }
  }
}

function onKeyDown(event) {
  const tag = event.target?.tagName?.toLowerCase();
  const editing =
    ["input", "textarea", "select"].includes(tag) ||
    event.target?.isContentEditable;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    postGraph("/api/editor/graph", "Save");
    return;
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    if (event.shiftKey) redo();
    else undo();
    return;
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
    event.preventDefault();
    redo();
    return;
  }
  if (editing) return;
  if (event.key.toLowerCase() === "f") {
    event.preventDefault();
    fitGraph(true);
  } else if (event.key === "Escape") {
    state.selected = new Set();
    state.selectedEdge = null;
    renderNodes();
    renderInspector();
  } else if (event.key === "Delete" || event.key === "Backspace") {
    if (state.selected.size) {
      event.preventDefault();
      removeNodes([...state.selected]);
    }
  } else if (event.key === "?" || (event.shiftKey && event.key === "/")) {
    openShortcuts();
  }
  // Arrow keys are intentionally not handled: node positions are locked to
  // their column and lane.
}

function cacheElements() {
  el.page = document.getElementById("editor-page");
  el.canvas = document.getElementById("canvas");
  el.world = document.getElementById("world");
  el.bands = document.getElementById("bands");
  el.lanes = document.getElementById("lanes");
  el.headers = document.getElementById("column-headers");
  el.nodes = document.getElementById("nodes");
  el.edges = document.getElementById("edges");
  el.marquee = document.getElementById("marquee");
  el.dropIndicator = document.getElementById("drop-indicator");
  el.inspector = document.getElementById("inspector");
  el.paletteList = document.getElementById("palette-list");
  el.paletteSearch = document.getElementById("palette-search");
  el.dirty = document.getElementById("editor-dirty");
  el.statusLeft = document.getElementById("status-left");
  el.statusMiddle = document.getElementById("status-middle");
  el.statusRight = document.getElementById("status-right");
  el.restartBanner = document.getElementById("restart-banner");
  el.modalRoot = document.getElementById("modal-root");
  el.zoomLabel = document.getElementById("zoom-label");
  el.btnUndo = document.getElementById("btn-undo");
  el.btnRedo = document.getElementById("btn-redo");
  el.btnFit = document.getElementById("btn-fit");
  el.btnReload = document.getElementById("btn-reload");
  el.btnValidate = document.getElementById("btn-validate");
  el.btnSave = document.getElementById("btn-save");
  el.btnUseDefaultsAll = document.getElementById("btn-use-defaults-all");
  el.btnZoomIn = document.getElementById("btn-zoom-in");
  el.btnZoomOut = document.getElementById("btn-zoom-out");
  el.btnShortcuts = document.getElementById("btn-shortcuts");
}

async function boot() {
  cacheElements();
  bindEvents();
  renderPalette();
  updateDirtyText();
  try {
    await fetchGraph();
  } catch (error) {
    state.serverErrors = [
      { code: "load", message: `Could not load editor graph: ${error.message}` },
    ];
    renderInspector();
    renderStatus();
    toast(String(error.message || error), "error");
  }
}

boot();
