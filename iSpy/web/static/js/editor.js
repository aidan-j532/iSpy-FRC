import {
  ROW_Y,
  ROW_LABELS,
  NODE_SIZES,
  clamp,
  finite,
  snap,
  rowY,
  nodeSize,
  effectiveValue,
  settingIsRequired,
  gateActive,
  completion,
  connectionError,
  edgeState,
  deriveEdges,
  edgePath,
  ringPath,
  fitViewport,
  uniqueCameraName,
  nextOrder,
  executionOrder,
  chainReady as coreChainReady,
  addGate,
  removalPlan,
  nodePayload,
} from "./editor_core.js";

const state = {
  nodes: [],
  edges: [],
  palette: {
    cameras: [],
    pipelines: [],
    frame_processors: [],
    trackers: [],
    utilities: [],
  },
  viewport: { x: 80, y: 30, zoom: 1 },
  grid: 20,
  selected: new Set(),
  selectedEdge: null,
  dirty: false,
  restartRequired: false,
  busy: false,
  history: [],
  future: [],
  editBefore: null,
  paletteFilter: "",
  drag: null,
  pan: null,
  connect: null,
  marquee: null,
  serverErrors: [],
  serverWarnings: [],
  spaceDown: false,
  confirmResolve: null,
  publishSources: [],
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

const deep = (value) => {
  if (value === undefined) return undefined;
  return JSON.parse(JSON.stringify(value));
};

const byId = (id) => state.nodes.find((node) => node.id === id);
const getNodes = (type) => state.nodes.filter((node) => node.type === type);
const getCamera = (name) =>
  state.nodes.find((node) => node.type === "camera" && node.camera === name);
const getPipeline = (camera) =>
  state.nodes.find(
    (node) => node.type === "pipeline" && node.camera === camera,
  );
const getProcessors = (camera) =>
  state.nodes.filter(
    (node) => node.type === "frame_processor" && node.camera === camera,
  );

function toast(message, type = "info") {
  if (typeof window.showToast === "function") {
    window.showToast(message, type);
  } else if (type === "error") {
    console.error(message);
  } else {
    console.log(message);
  }
}

function valueFor(node, key) {
  return effectiveValue(node.schema || {}, node.settings || {}, key);
}

function modelPath(value) {
  if (typeof value === "string") return value;
  if (value && typeof value === "object") {
    return value.file_path || value.source_pt || value.path || "";
  }
  return "";
}

function normalizeNode(raw) {
  const node = {
    ...raw,
    settings: deep(raw.settings || {}),
    schema: deep(raw.schema || {}),
    warnings: Array.isArray(raw.warnings) ? [...raw.warnings] : [],
  };
  node.x = finite(raw.x, 0);
  node.y = rowY(node.type);
  node.order = Number.isFinite(Number(raw.order)) ? Number(raw.order) : 0;
  node.completion = completion(node.schema, node.settings);
  node.required = node.completion.required;
  return node;
}

function updateAllCompletions() {
  for (const node of state.nodes) {
    node.y = rowY(node.type);
    node.completion = completion(node.schema || {}, node.settings || {});
    node.required = node.completion.required;
  }
}

function normalizeViewport(raw) {
  const viewport = raw && typeof raw === "object" ? raw : {};
  return {
    x: finite(viewport.x, 80),
    y: finite(viewport.y, 30),
    zoom: clamp(finite(viewport.zoom, 1), 0.25, 2.5),
  };
}

function selectedNodes() {
  return [...state.selected].map(byId).filter(Boolean);
}

function selectedNode() {
  const nodes = selectedNodes();
  return nodes.length === 1 ? nodes[0] : null;
}

function selectedCamera() {
  for (const node of selectedNodes()) {
    if (node.type === "camera") return node;
    if (node.camera) {
      const camera = getCamera(node.camera);
      if (camera) return camera;
    }
  }
  const cameras = getNodes("camera");
  return cameras.length === 1 ? cameras[0] : null;
}

function chainReady() {
  return coreChainReady(state.nodes);
}

function normalizeNodeOrder() {
  const cameras = getNodes("camera");
  const used = new Set();
  const result = [];
  for (const camera of cameras) {
    result.push(camera);
    used.add(camera.id);
    const processors = executionOrder(getProcessors(camera.camera));
    processors.forEach((node, index) => {
      node.order = index;
      result.push(node);
      used.add(node.id);
    });
    const pipelines = state.nodes.filter(
      (node) =>
        node.type === "pipeline" && node.camera === camera.camera,
    );
    pipelines.forEach((node) => {
      result.push(node);
      used.add(node.id);
    });
  }
  for (const type of ["tracker", "utility"]) {
    const group = executionOrder(getNodes(type));
    group.forEach((node, index) => {
      node.order = index;
      result.push(node);
      used.add(node.id);
    });
  }
  for (const node of state.nodes) {
    if (!used.has(node.id)) result.push(node);
  }
  state.nodes = result;
}

function rebuildEdges() {
  updateAllCompletions();
  state.edges = deriveEdges(state.nodes);
}

function snapshot() {
  return JSON.stringify({
    nodes: state.nodes,
    edges: state.edges,
    selected: [...state.selected],
  });
}

function pushHistory(before) {
  if (!before) return;
  if (before !== snapshot()) {
    state.history.push(before);
    if (state.history.length > 100) state.history.shift();
    state.future = [];
  }
}

function commitEdit() {
  if (!state.editBefore) return;
  const before = state.editBefore;
  state.editBefore = null;
  pushHistory(before);
}

function markDirty({ clearErrors = true } = {}) {
  state.dirty = true;
  if (clearErrors) {
    state.serverErrors = [];
    state.serverWarnings = [];
  }
  renderStatus();
}

function updateButtons() {
  if (!el.undo) return;
  el.undo.disabled = state.busy || state.history.length === 0;
  el.redo.disabled = state.busy || state.future.length === 0;
  el.save.disabled = state.busy;
  el.validate.disabled = state.busy;
  el.reload.disabled = state.busy;
  el.save.textContent = state.busy ? "Working…" : "Save";
}

function setBusy(value) {
  state.busy = Boolean(value);
  updateButtons();
  renderStatus();
}

function showRestartBanner() {
  if (!el.restartBanner) return;
  el.restartBanner.classList.toggle("visible", state.restartRequired);
}

function updateViewportLabels() {
  if (el.zoomLabel) {
    el.zoomLabel.textContent = `${Math.round(state.viewport.zoom * 100)}%`;
  }
  if (el.statusRight) {
    el.statusRight.textContent = `Grid ${state.grid} · ${Math.round(
      state.viewport.zoom * 100,
    )}% · ${state.dirty ? "Unsaved changes" : "Saved"}`;
  }
}

function applyViewport() {
  if (!el.world) return;
  el.world.style.transform = `translate(${state.viewport.x}px, ${state.viewport.y}px) scale(${state.viewport.zoom})`;
  const step = Math.max(1, state.grid * state.viewport.zoom);
  el.canvas.style.backgroundSize = `${step}px ${step}px`;
  el.canvas.style.backgroundPosition = `${state.viewport.x}px ${state.viewport.y}px`;
  updateViewportLabels();
}

function renderStatus() {
  if (!el.statusLeft) return;
  const incomplete = state.nodes.filter(
    (node) => !node.completion?.complete,
  ).length;
  const errors = state.serverErrors.length;
  const warnings =
    state.serverWarnings.length +
    state.nodes.reduce(
      (total, node) => total + (node.warnings?.length || 0),
      0,
    );
  const dotClass = errors || incomplete ? "bad" : warnings ? "warn" : "ok";
  el.statusLeft.innerHTML = `<span class="status-dot ${dotClass}"></span>${
    state.nodes.length
  } nodes · ${
    incomplete ? `${incomplete} incomplete` : "all complete"
  }`;
  el.statusMiddle.textContent = errors
    ? `${errors} error${errors === 1 ? "" : "s"}`
    : warnings
      ? `${warnings} warning${warnings === 1 ? "" : "s"}`
      : "";
  el.statusMiddle.style.color = errors
    ? "#fecaca"
    : warnings
      ? "#fde68a"
      : "";
  if (el.dirty) {
    el.dirty.textContent = state.busy
      ? "Saving…"
      : state.dirty
        ? "Unsaved changes"
        : state.restartRequired
          ? "Saved · restart required"
          : "Saved";
    el.dirty.style.color = state.dirty ? "#fde68a" : "";
  }
  updateViewportLabels();
  updateButtons();
}

function renderAll() {
  renderPalette();
  renderRows();
  renderNodes();
  renderEdges();
  renderInspector();
  renderStatus();
  applyViewport();
  showRestartBanner();
}

async function reloadGraph() {
  if (state.busy) return;
  if (
    state.dirty &&
    !(await confirmDialog(
      "Discard unsaved changes?",
      "Reloading reads the saved config and discards unsaved node, setting, and layout changes.",
      "Reload",
    ))
  ) {
    return;
  }
  await loadGraph({ first: false });
}

async function loadGraph({ first = false, silent = false } = {}) {
  try {
    const response = await fetch("/api/editor/graph", {
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    applyGraph(data, { first });
    if (!silent) toast("Editor reloaded from config.", "info");
  } catch (error) {
    el.inspector.innerHTML = `<div class="validation-card"><h3>Editor unavailable</h3><div class="validation-line error"><span>Could not load /api/editor/graph: ${esc(
      error.message,
    )}</span></div></div>`;
    el.statusLeft.textContent = "Graph load failed";
    toast("Could not load the editor graph.", "error");
  }
}

function applyGraph(data, { first = false } = {}) {
  const graph = data.graph || {};
  state.nodes = (graph.nodes || []).map(normalizeNode);
  state.edges = (graph.edges || []).map((edge) => ({ ...edge }));
  state.palette = data.palette || state.palette;
  state.grid = clamp(finite(graph.layout?.grid, 20), 8, 80);
  state.viewport = normalizeViewport(graph.layout?.viewport);
  state.restartRequired = Boolean(data.restart_required);
  state.dirty = false;
  state.history = [];
  state.future = [];
  state.serverErrors = [];
  state.serverWarnings = [];
  state.selectedEdge = null;
  const ids = new Set(state.nodes.map((node) => node.id));
  state.selected = new Set(
    [...state.selected].filter((id) => ids.has(id)),
  );
  updateAllCompletions();
  renderAll();
  if (first && !graph.layout_saved) {
    requestAnimationFrame(() => fitToNodes(false));
  }
}

function renderPalette() {
  const query = state.paletteFilter.trim().toLowerCase();
  const groups = [
    ["Cameras", state.palette.cameras || [], "camera"],
    ["Pipelines", state.palette.pipelines || [], "pipeline"],
    [
      "Frame Processors",
      state.palette.frame_processors || [],
      "frame_processor",
    ],
    ["Trackers", state.palette.trackers || [], "tracker"],
    ["Utilities", state.palette.utilities || [], "utility"],
  ];
  const camera = selectedCamera();
  let html = "";
  for (const [label, items, type] of groups) {
    const filtered = items.filter((item) => {
      if (!query) return true;
      return `${item.name || ""} ${item.label || ""} ${item.doc || ""}`
        .toLowerCase()
        .includes(query);
    });
    if (!filtered.length) continue;
    html += `<h3>${esc(label)} <span>${filtered.length}</span></h3>`;
    for (const item of filtered) {
      const name = item.name || item.label || "";
      const dot = type === "frame_processor" ? "fp" : type;
      const gate = addGate(state.nodes, type, { camera, plugin: name });
      const locked = !gate.allowed;
      const badges = [
        locked ? "locked" : "",
        item.template ? "template" : "",
        item.beta ? "beta" : "",
        item.needs_internet ? "internet" : "",
      ].filter(Boolean);
      const subtitle = (item.doc || item.class_name || "")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 80);
      const detail = locked ? gate.reason : subtitle || name;
      html += `<button class="palette-item ${
        locked ? "locked" : ""
      }" data-add-type="${esc(type)}" data-add-name="${esc(
        name,
      )}" ${locked ? 'disabled aria-disabled="true"' : ""} title="${esc(
        detail,
      )}">
        <span class="palette-dot ${dot}"></span>
        <span class="palette-copy"><b>${esc(
          item.label || name,
        )}</b><small>${esc(detail)}</small></span>
        ${badges
          .map((badge) => `<span class="field-badge">${esc(badge)}</span>`)
          .join("")}
      </button>`;
    }
  }
  el.paletteList.innerHTML =
    html || `<div class="palette-empty">No matching nodes.</div>`;
}

function renderRows() {
  const counts = {};
  for (const node of state.nodes) {
    counts[node.type] = (counts[node.type] || 0) + 1;
  }
  el.rows.innerHTML = Object.keys(ROW_Y)
    .map((type) => {
      const top = ROW_Y[type] - 28;
      return `<div class="editor-row-band" style="top:${top}px;height:152px">
        <span class="editor-row-label">${esc(
          ROW_LABELS[type] || type,
        )} · ${counts[type] || 0}</span>
      </div>`;
    })
    .join("");
}

function iconSvg(type) {
  const paths = {
    camera:
      '<circle cx="12" cy="12" r="3"></circle><path d="M4 7h3l1.5-2h7L17 7h3v12H4z"></path>',
    frame_processor:
      '<path d="M4 8h16v8H4z"></path><path d="M8 8V5M12 8V5M16 8V5M8 16v3M12 16v3M16 16v3"></path>',
    pipeline:
      '<path d="M4 6h8l4 6-4 6H4z"></path><path d="M16 12h4"></path>',
    tracker:
      '<path d="M12 3l7 4v5c0 4-3 7-7 9-4-2-7-5-7-9V7z"></path><circle cx="12" cy="11" r="2.4"></circle>',
    utility:
      '<path d="M12 3l2.2 4.6 5 .7-3.6 3.6.8 5-4.4-2.4-4.4 2.4.8-5L4.8 8.3l5-.7z"></path>',
  };
  return `<svg class="node-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${paths[type] || paths.utility}</svg>`;
}

function nodeChips(node) {
  const chips = [];
  if (node.type === "camera") {
    chips.push(node.camera_type || "opencv");
    if (!node.calibrated) chips.push("calibrate");
  } else if (node.type === "frame_processor") {
    chips.push(`#${(node.order ?? 0) + 1}`);
    chips.push(node.camera || "");
  } else if (node.type === "pipeline") {
    chips.push(node.plugin || "pipeline");
    if (node.beta) chips.push("beta");
    if (node.requires_calibration && !node.calibrated) {
      chips.push("calibration");
    }
  } else {
    chips.push(`#${(node.order ?? 0) + 1}`);
    chips.push("global");
  }
  if (node.needs_internet) chips.push("internet");
  if (node.warnings?.length) chips.push("warning");
  return chips
    .filter(Boolean)
    .map(
      (chip) => {
        const classes = ["node-chip"];
        if (["calibrate", "calibration", "warning"].includes(chip)) {
          classes.push("warn");
        }
        if (/^#\d+$/.test(chip)) classes.push("node-order");
        return `<span class="${classes.join(" ")}" ${
          classes.includes("node-order")
            ? 'title="Execution order - drag horizontally to change it"'
            : ""
        }>${esc(chip)}</span>`;
      },
    )
    .join("");
}

function nodeHtml(node) {
  const size = nodeSize(node.type);
  const path = ringPath(node.type, size.w, size.h);
  const percent = node.completion?.percent ?? 100;
  const selected = state.selected.has(node.id);
  const error = state.serverErrors.some(
    (item) => item.node === node.id,
  );
  const stateIcon = !node.completion?.complete
    ? "×"
    : node.warnings?.length
      ? "!"
      : "✓";
  const progressTitle = !node.completion?.complete
    ? `Missing required: ${(
        node.completion?.missing_labels ||
        node.completion?.missing ||
        []
      ).join(", ")}`
    : node.completion?.total === 0
      ? "No required fields"
      : "All required fields complete";
  const subtitle =
    node.subtitle ||
    (node.type === "camera"
      ? node.camera_type || ""
      : node.plugin || "");
  const ports = `${
    node.type !== "camera"
      ? `<button class="port port-in" data-port-in="${esc(
          node.id,
        )}" title="Input"></button>`
      : ""
  }${
    node.type !== "utility"
      ? `<button class="port port-out" data-port-out="${esc(
          node.id,
        )}" title="Output"></button>`
      : ""
  }`;
  return `<div class="node node-${node.type} ${
    node.completion?.complete ? "complete" : "incomplete"
  } ${selected ? "selected" : ""} ${
    error ? "server-error" : ""
  }" data-node-id="${esc(node.id)}" style="left:${node.x}px;top:${
    node.y
  }px;width:${size.w}px;height:${size.h}px" title="${esc(
    node.id,
  )}">
    <svg class="node-ring" viewBox="0 0 ${size.w} ${size.h}" aria-hidden="true">
      <path class="ring-track" d="${path}" pathLength="100"></path>
      <path class="ring-progress" d="${path}" pathLength="100" stroke-dasharray="100" stroke-dashoffset="${
        100 - percent
      }"></path>
    </svg>
    <div class="node-body" data-drag-handle="${esc(node.id)}">
      <div class="node-topline">${iconSvg(node.type)}<span class="node-label">${esc(
        node.label,
      )}</span></div>
      <div class="node-sub">${esc(subtitle)}</div>
      <div class="node-meta">${nodeChips(node)}</div>
    </div>
    <button class="node-menu" data-node-menu="${esc(
      node.id,
    )}" title="Node actions">⋯</button>
    <div class="node-state ${
      node.completion?.complete && node.warnings?.length ? "warn" : ""
    }" title="${esc(progressTitle)}">${stateIcon}</div>
    ${ports}
  </div>`;
}

function ghostSpecs() {
  const cameras = getNodes("camera");
  const ghosts = [];
  const cameraMax = cameras.length
    ? Math.max(...cameras.map((node) => node.x))
    : 0;
  ghosts.push({
    type: "camera",
    camera: null,
    x: snap(cameras.length ? cameraMax + 460 : 80),
    label: "Add camera",
    hint: "starts a chain",
  });
  for (const camera of cameras) {
    const processors = getProcessors(camera.camera);
    const pipeline = getPipeline(camera.camera);
    const active =
      state.selected.has(camera.id) ||
      processors.some((node) => state.selected.has(node.id)) ||
      (pipeline && state.selected.has(pipeline.id));
    const lastX = processors.length
      ? Math.max(...processors.map((node) => node.x))
      : camera.x + 420;
    if (
      addGate(state.nodes, "pipeline", { camera }).allowed
    ) {
      ghosts.push({
        type: "pipeline",
        camera: camera.camera,
        x: snap(Math.max(camera.x + 680, lastX + 260)),
        label: "Add pipeline",
        hint: "exactly one per camera",
      });
    }
    if (
      active &&
      addGate(state.nodes, "frame_processor", { camera }).allowed
    ) {
      ghosts.push({
        type: "frame_processor",
        camera: camera.camera,
        x: snap(
          pipeline
            ? Math.max(camera.x + 400, pipeline.x - 260)
            : lastX + 240,
        ),
        label: "Add frame processor",
        hint: "optional, ordered",
      });
    }
  }
  if (chainReady()) {
    const trackers = getNodes("tracker");
    const utilities = getNodes("utility");
    const trackerMax = trackers.length
      ? Math.max(...trackers.map((node) => node.x))
      : 0;
    const utilityMax = utilities.length
      ? Math.max(...utilities.map((node) => node.x))
      : 0;
    ghosts.push({
      type: "tracker",
      camera: null,
      x: snap(trackers.length ? trackerMax + 300 : 80),
      label: "Add tracker",
      hint: "global execution order",
    });
    ghosts.push({
      type: "utility",
      camera: null,
      x: snap(utilities.length ? utilityMax + 300 : 80),
      label: "Add utility",
      hint: "runs after trackers",
    });
  }
  return ghosts;
}

function ghostHtml(ghost) {
  const size = nodeSize(ghost.type);
  return `<div class="node-ghost" data-ghost-type="${esc(
    ghost.type,
  )}" data-ghost-camera="${esc(ghost.camera || "")}" style="left:${
    ghost.x
  }px;top:${rowY(ghost.type)}px;width:${size.w}px;height:${
    size.h
  }px" title="${esc(ghost.hint)}"><b>+ ${esc(
    ghost.label,
  )}</b><small>${esc(ghost.hint)}</small></div>`;
}

function renderNodes() {
  const html = state.nodes.map(nodeHtml).join("");
  el.nodes.innerHTML = html + ghostSpecs().map(ghostHtml).join("");
}

function refreshEdgeStates() {
  for (const edge of state.edges) {
    const source = byId(edge.from);
    const target = byId(edge.to);
    edge.state = edgeState(source, target, edge);
  }
}

function renderEdges() {
  refreshEdgeStates();
  const colors = {
    dashed: "#f87171",
    half: "#fbbf24",
    complete: "#34d399",
    invalid: "#f87171",
  };
  const defs = `<defs>${Object.entries(colors)
    .map(
      ([name, color]) =>
        `<marker id="editor-arrow-${name}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="${color}"></path></marker>`,
    )
    .join("")}</defs>`;
  const paths = state.edges
    .map((edge) => {
      const source = byId(edge.from);
      const target = byId(edge.to);
      if (!source || !target) return "";
      const d = edgePath(source, target);
      const selected = state.selectedEdge === edge.id;
      return `<g data-edge-id="${esc(edge.id)}" class="${
        selected ? "edge-selected" : ""
      }">
        <path class="edge-hit" d="${d}"></path>
        <path class="edge-visible edge-${esc(
          edge.state,
        )}" d="${d}" marker-end="url(#editor-arrow-${esc(
          edge.state,
        )})"></path>
      </g>`;
    })
    .join("");
  el.edges.innerHTML = defs + paths;
}

function screenToWorld(clientX, clientY) {
  const rect = el.canvas.getBoundingClientRect();
  return {
    x: (clientX - rect.left - state.viewport.x) / state.viewport.zoom,
    y: (clientY - rect.top - state.viewport.y) / state.viewport.zoom,
  };
}

function worldToScreen(x, y) {
  const rect = el.canvas.getBoundingClientRect();
  return {
    x: rect.left + state.viewport.x + x * state.viewport.zoom,
    y: rect.top + state.viewport.y + y * state.viewport.zoom,
  };
}

function fitToNodes(showMessage = true) {
  const width = el.canvas.clientWidth;
  const height = el.canvas.clientHeight;
  if (!state.nodes.length) {
    state.viewport = { x: 80, y: 30, zoom: 1 };
  } else {
    state.viewport = fitViewport(state.nodes, width, height);
  }
  applyViewport();
  renderStatus();
  if (showMessage) toast("Fitted graph to the canvas.", "info");
}

function zoomAt(factor, clientX, clientY) {
  const rect = el.canvas.getBoundingClientRect();
  const px = clientX - rect.left;
  const py = clientY - rect.top;
  const world = screenToWorld(clientX, clientY);
  const next = clamp(state.viewport.zoom * factor, 0.25, 2.5);
  state.viewport.zoom = next;
  state.viewport.x = px - world.x * next;
  state.viewport.y = py - world.y * next;
  applyViewport();
  renderStatus();
}

function resetViewport() {
  state.viewport = { x: 80, y: 30, zoom: 1 };
  applyViewport();
  renderStatus();
}

function selectOnly(id) {
  state.selected.clear();
  state.selectedEdge = null;
  if (id) state.selected.add(id);
  renderNodes();
  renderInspector();
  renderStatus();
}

function toggleSelection(id) {
  state.selectedEdge = null;
  if (state.selected.has(id)) state.selected.delete(id);
  else state.selected.add(id);
  renderNodes();
  renderInspector();
  renderStatus();
}

function clearSelection() {
  state.selected.clear();
  state.selectedEdge = null;
  renderNodes();
  renderInspector();
  renderStatus();
}

function focusNode(id) {
  const node = byId(id);
  if (!node) return;
  state.selected = new Set([id]);
  state.selectedEdge = null;
  const size = nodeSize(node.type);
  const width = el.canvas.clientWidth;
  const height = el.canvas.clientHeight;
  const zoom = state.viewport.zoom;
  state.viewport.x = width / 2 - (node.x + size.w / 2) * zoom;
  state.viewport.y = height / 2 - (node.y + size.h / 2) * zoom;
  renderNodes();
  renderEdges();
  renderInspector();
  renderStatus();
  applyViewport();
}

function startPan(event) {
  state.pan = {
    startX: event.clientX,
    startY: event.clientY,
    vx: state.viewport.x,
    vy: state.viewport.y,
  };
  el.canvas.classList.add("panning");
  el.canvas.setPointerCapture?.(event.pointerId);
}

function startMarquee(event) {
  const rect = el.canvas.getBoundingClientRect();
  state.marquee = {
    startX: event.clientX,
    startY: event.clientY,
    rect,
    additive: event.shiftKey || event.ctrlKey || event.metaKey,
    moved: false,
  };
  el.marquee.hidden = false;
  el.marquee.style.left = `${event.clientX - rect.left}px`;
  el.marquee.style.top = `${event.clientY - rect.top}px`;
  el.marquee.style.width = "0px";
  el.marquee.style.height = "0px";
  event.preventDefault();
}

function onCanvasPointerDown(event) {
  if (event.button !== 0) return;
  if (
    event.target.closest(
      ".node, .node-ghost, .edge-hit, .node-menu, .port",
    )
  ) {
    return;
  }
  closeContextMenu();
  if (event.shiftKey) {
    startMarquee(event);
  } else {
    startPan(event);
  }
}

function onWheel(event) {
  event.preventDefault();
  const factor = Math.exp(-event.deltaY * 0.0015);
  if (event.ctrlKey || event.metaKey) {
    zoomAt(factor, event.clientX, event.clientY);
    return;
  }
  if (event.shiftKey) {
    state.viewport.x -= event.deltaY || event.deltaX;
  } else {
    state.viewport.x -= event.deltaX;
    state.viewport.y -= event.deltaY;
  }
  applyViewport();
  renderStatus();
}

function startDrag(event, node) {
  if (event.button !== 0) return;
  if (event.target.closest("button, .port")) return;
  event.preventDefault();
  event.stopPropagation();
  const additive = event.shiftKey || event.ctrlKey || event.metaKey;
  if (additive) {
    toggleSelection(node.id);
    if (!state.selected.has(node.id)) return;
  } else if (!state.selected.has(node.id)) {
    selectOnly(node.id);
  }
  if (!state.selected.has(node.id)) return;
  const positions = new Map();
  for (const id of state.selected) {
    const selected = byId(id);
    if (selected) positions.set(id, selected.x);
  }
  state.drag = {
    before: snapshot(),
    startX: event.clientX,
    positions,
    moved: false,
  };
  for (const id of positions.keys()) {
    const nodeEl = el.nodes.querySelector(`[data-node-id="${id}"]`);
    nodeEl?.classList.add("dragging");
  }
}

function nodeAtClientPoint(clientX, clientY) {
  const nodeEl = document
    .elementFromPoint(clientX, clientY)
    ?.closest("[data-node-id]");
  return nodeEl ? byId(nodeEl.dataset.nodeId) : null;
}

function clearConnectHover() {
  el.nodes
    .querySelectorAll(".connect-target, .connect-invalid")
    .forEach((node) => {
      node.classList.remove("connect-target", "connect-invalid");
    });
  const temp = document.getElementById("editor-temp-edge");
  temp?.classList.remove("edge-temp-valid", "edge-temp-invalid");
}

function updateConnectHover(clientX, clientY) {
  if (!state.connect) return;
  clearConnectHover();
  const source = byId(state.connect.sourceId);
  const target = nodeAtClientPoint(clientX, clientY);
  const temp = document.getElementById("editor-temp-edge");
  if (!source || !target || target.id === source.id) {
    state.connect.hoverId = null;
    state.connect.error = "";
    return;
  }
  const error = connectionError(
    source,
    target,
    state.nodes,
    state.edges,
  );
  state.connect.hoverId = target.id;
  state.connect.error = error;
  el.nodes
    .querySelector(`[data-node-id="${target.id}"]`)
    ?.classList.add(error ? "connect-invalid" : "connect-target");
  temp?.classList.add(error ? "edge-temp-invalid" : "edge-temp-valid");
}

function updateTempEdge(clientX, clientY) {
  if (!state.connect) return;
  const source = byId(state.connect.sourceId);
  const temp = document.getElementById("editor-temp-edge");
  if (!source || !temp) return;
  const size = nodeSize(source.type);
  const sx = source.x + size.w;
  const sy = source.y + size.h / 2;
  const point = screenToWorld(clientX, clientY);
  const bend = Math.max(40, Math.abs(point.x - sx) * 0.42);
  temp.setAttribute(
    "d",
    `M ${sx} ${sy} C ${sx + bend} ${sy}, ${
      point.x - bend
    } ${point.y}, ${point.x} ${point.y}`,
  );
}

function startConnect(event, node) {
  if (!node) return;
  event.preventDefault();
  event.stopPropagation();
  if (!node.completion?.complete) {
    toast("Configure this node before connecting it.", "warning");
    return;
  }
  cancelConnect();
  state.connect = { sourceId: node.id, hoverId: null, error: "" };
  el.canvas.classList.add("connecting");
  const sourceEl = el.nodes.querySelector(
    `[data-node-id="${node.id}"] .port-out`,
  );
  sourceEl?.classList.add("active");
  const path = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "path",
  );
  path.setAttribute("id", "editor-temp-edge");
  path.setAttribute("class", "edge-temp");
  el.edges.appendChild(path);
  updateTempEdge(event.clientX, event.clientY);
}

function cancelConnect() {
  state.connect = null;
  document.getElementById("editor-temp-edge")?.remove();
  el.canvas.classList.remove("connecting");
  clearConnectHover();
  el.nodes
    .querySelectorAll(".port-out.active")
    .forEach((port) => port.classList.remove("active"));
}

function finishConnect(event) {
  if (!state.connect) return;
  const source = byId(state.connect.sourceId);
  const target = nodeAtClientPoint(event.clientX, event.clientY);
  const error =
    source && target
      ? connectionError(source, target, state.nodes, state.edges)
      : "";
  cancelConnect();
  renderNodes();
  if (!source) return;
  if (!target) return;
  if (error) {
    toast(`Connection rejected: ${error}`, "error");
    return;
  }
  const before = snapshot();
  state.edges.push({
    id: `e:${source.id}->${target.id}`,
    from: source.id,
    to: target.id,
    kind: "flow",
    state: edgeState(source, target),
  });
  pushHistory(before);
  markDirty();
  renderNodes();
  renderEdges();
  renderInspector();
  toast(`Connected ${source.label} to ${target.label}.`, "success");
}

function onNodesPointerDown(event) {
  const outPort = event.target.closest("[data-port-out]");
  if (outPort) {
    startConnect(event, byId(outPort.dataset.portOut));
    return;
  }
  if (event.target.closest("[data-port-in]")) {
    event.stopPropagation();
    return;
  }
  if (event.target.closest("[data-node-menu]")) {
    event.stopPropagation();
    return;
  }
  const ghost = event.target.closest("[data-ghost-type]");
  if (ghost) {
    event.preventDefault();
    event.stopPropagation();
    openPicker(
      ghost.dataset.ghostType,
      ghost.dataset.ghostCamera || null,
    );
    return;
  }
  const nodeEl = event.target.closest("[data-node-id]");
  if (!nodeEl) return;
  const node = byId(nodeEl.dataset.nodeId);
  if (!node) return;
  startDrag(event, node);
}

function onNodesClick(event) {
  const menu = event.target.closest("[data-node-menu]");
  if (menu) {
    event.preventDefault();
    event.stopPropagation();
    const node = byId(menu.dataset.nodeMenu);
    if (!node) return;
    if (!state.selected.has(node.id)) selectOnly(node.id);
    const rect = menu.getBoundingClientRect();
    openContextMenu(rect.left, rect.bottom + 4, node);
  }
}

function onNodesContext(event) {
  const nodeEl = event.target.closest("[data-node-id]");
  if (!nodeEl) return;
  event.preventDefault();
  const node = byId(nodeEl.dataset.nodeId);
  if (!node) return;
  if (!state.selected.has(node.id)) selectOnly(node.id);
  openContextMenu(event.clientX, event.clientY, node);
}

function onEdgeClick(event) {
  const group = event.target.closest("[data-edge-id]");
  if (!group) return;
  event.stopPropagation();
  state.selected.clear();
  state.selectedEdge = group.dataset.edgeId;
  renderNodes();
  renderEdges();
  renderInspector();
  renderStatus();
}

function onPointerMove(event) {
  if (state.drag) {
    const dx = (event.clientX - state.drag.startX) / state.viewport.zoom;
    if (Math.abs(dx) > 1) state.drag.moved = true;
    for (const [id, startX] of state.drag.positions) {
      const node = byId(id);
      if (!node) continue;
      node.x = snap(startX + dx, state.grid);
      const nodeEl = el.nodes.querySelector(
        `[data-node-id="${id}"]`,
      );
      if (nodeEl) nodeEl.style.left = `${node.x}px`;
    }
    renderEdges();
    return;
  }
  if (state.pan) {
    state.viewport.x =
      state.pan.vx + (event.clientX - state.pan.startX);
    state.viewport.y =
      state.pan.vy + (event.clientY - state.pan.startY);
    applyViewport();
    return;
  }
  if (state.marquee) {
    const left = Math.min(state.marquee.startX, event.clientX);
    const top = Math.min(state.marquee.startY, event.clientY);
    const width = Math.abs(event.clientX - state.marquee.startX);
    const height = Math.abs(event.clientY - state.marquee.startY);
    state.marquee.moved = width > 3 || height > 3;
    const rect = state.marquee.rect;
    el.marquee.style.left = `${left - rect.left}px`;
    el.marquee.style.top = `${top - rect.top}px`;
    el.marquee.style.width = `${width}px`;
    el.marquee.style.height = `${height}px`;
    return;
  }
  if (state.connect) {
    updateConnectHover(event.clientX, event.clientY);
    updateTempEdge(event.clientX, event.clientY);
  }
}

function finishMarquee(event) {
  if (!state.marquee) return;
  const marquee = state.marquee;
  state.marquee = null;
  el.marquee.hidden = true;
  if (!marquee.moved) return;
  const left = Math.min(marquee.startX, event.clientX);
  const top = Math.min(marquee.startY, event.clientY);
  const right = Math.max(marquee.startX, event.clientX);
  const bottom = Math.max(marquee.startY, event.clientY);
  if (!marquee.additive) state.selected.clear();
  for (const node of state.nodes) {
    const screen = worldToScreen(node.x, node.y);
    const size = nodeSize(node.type);
    const nodeRight = screen.x + size.w * state.viewport.zoom;
    const nodeBottom = screen.y + size.h * state.viewport.zoom;
    if (
      nodeRight >= left &&
      screen.x <= right &&
      nodeBottom >= top &&
      screen.y <= bottom
    ) {
      state.selected.add(node.id);
    }
  }
  state.selectedEdge = null;
  renderNodes();
  renderInspector();
  renderStatus();
}

function onPointerUp(event) {
  if (state.drag) {
    const drag = state.drag;
    state.drag = null;
    el.nodes
      .querySelectorAll(".dragging")
      .forEach((node) => node.classList.remove("dragging"));
    if (drag.moved) {
      normalizeNodeOrder();
      rebuildEdges();
      pushHistory(drag.before);
      markDirty();
      renderAll();
    } else {
      renderNodes();
      renderInspector();
    }
    return;
  }
  if (state.pan) {
    state.pan = null;
    el.canvas.classList.remove("panning");
    return;
  }
  if (state.marquee) {
    finishMarquee(event);
    return;
  }
  if (state.connect) {
    finishConnect(event);
  }
}

function renderInspector() {
  if (!el.inspector) return;
  if (state.selectedEdge) {
    const edge = state.edges.find(
      (item) => item.id === state.selectedEdge,
    );
    if (edge) {
      el.inspector.innerHTML = edgeInspectorHtml(edge);
      return;
    }
    state.selectedEdge = null;
  }
  const nodes = selectedNodes();
  if (!nodes.length) {
    el.inspector.innerHTML = overviewHtml();
    return;
  }
  if (nodes.length > 1) {
    el.inspector.innerHTML = multiSelectionHtml(nodes);
    return;
  }
  el.inspector.innerHTML = singleNodeHtml(nodes[0]);
}

function overviewHtml() {
  const incomplete = state.nodes.filter(
    (node) => !node.completion?.complete,
  ).length;
  const cameraCount = getNodes("camera").length;
  const trackerCount = getNodes("tracker").length;
  const utilityCount = getNodes("utility").length;
  const warnings =
    state.serverWarnings.length +
    state.nodes.reduce(
      (total, node) => total + (node.warnings?.length || 0),
      0,
    );
  return `<div class="inspector-head"><div><h2>Editor overview</h2><div class="field-help">Click a node to inspect it. Drag it to move it.</div></div></div>
    <div class="overview-grid">
      <div class="overview-stat"><b>${cameraCount}</b><span>camera${cameraCount === 1 ? "" : "s"}</span></div>
      <div class="overview-stat"><b>${incomplete}</b><span>incomplete node${incomplete === 1 ? "" : "s"}</span></div>
      <div class="overview-stat"><b>${trackerCount}</b><span>tracker${trackerCount === 1 ? "" : "s"}</span></div>
      <div class="overview-stat"><b>${utilityCount}</b><span>utilit${utilityCount === 1 ? "y" : "ies"}</span></div>
    </div>
    ${
      warnings
        ? `<div class="calibration-card">${warnings} warning${
            warnings === 1 ? "" : "s"
          } need attention. Run Validate for details.</div>`
        : ""
    }
    <div class="hint-box">
      <b>Flow:</b> Camera → Frame Processors → Pipeline → Trackers → Utilities.<br />
      Required fields gate the next row. A red dashed outline is incomplete; a
      green check is complete. Use <kbd>Ctrl</kbd>+<kbd>S</kbd> to save and
      <kbd>F</kbd> to fit the canvas.
    </div>
    <div class="inspector-actions">
      <button data-action="validate">Validate</button>
      <button data-action="add-camera">Add camera</button>
      <button data-action="help">Shortcuts</button>
    </div>
    ${validationPanelHtml()}`;
}

function multiSelectionHtml(nodes) {
  return `<div class="inspector-head"><div><h2>${
    nodes.length
  } nodes selected</h2><div class="field-help">Drag any selected node to move the group horizontally. Rows stay locked.</div></div></div>
    <div class="hint-box">Changing a setting updates only the primary node. Use one node at a time for detailed settings.</div>
    ${nodes
      .map(
        (node) => `<button class="picker-item" data-action="focus-node" data-node-id="${esc(
          node.id,
        )}"><span class="palette-dot ${
          node.type === "frame_processor" ? "fp" : node.type
        }"></span><span>${esc(node.label)}</span><small>${esc(
          node.type.replaceAll("_", " "),
        )}</small></button>`,
      )
      .join("")}
    <div class="inspector-actions">
      <button class="danger" data-action="remove-selected">Remove selected</button>
      <button data-action="focus-node" data-node-id="${esc(
        nodes[0].id,
      )}">Inspect first</button>
    </div>
    ${validationPanelHtml()}`;
}

function edgeInspectorHtml(edge) {
  const source = byId(edge.from);
  const target = byId(edge.to);
  return `<div class="inspector-head"><div><h2>Connection</h2><div class="field-help">${esc(
    source?.label || edge.from,
  )} → ${esc(target?.label || edge.to)}</div></div><span class="type-badge">${esc(
    edge.state,
  )}</span></div>
    <div class="hint-box">
      Connections are validated against node type, camera ownership, supported
      pipelines, duplicates, and cycles. Structural flow is rebuilt from the
      camera and global execution-order config.
    </div>
    ${
      edge.warning
        ? `<div class="calibration-card">${esc(edge.warning)}</div>`
        : ""
    }
    <div class="inspector-actions">
      <button data-action="clear-edge">Clear selection</button>
    </div>
    ${validationPanelHtml()}`;
}

function singleNodeHtml(node) {
  const typeLabel = node.type.replaceAll("_", " ");
  const sections = schemaSections(node.schema || {});
  return `<div class="inspector-head">
      <div><h2>${esc(node.label)}</h2><div class="field-help">${esc(
        node.id,
      )}</div></div>
      <span class="type-badge">${esc(typeLabel)}</span>
    </div>
    ${progressCard(node)}
    ${nodeWarningsHtml(node)}
    ${calibrationCard(node)}
    ${sections
      .map(
        ([title, fields]) =>
          `<div class="inspector-section"><h3>${esc(
            title,
          )}</h3>${fields
            .map(([key, defn]) => fieldHtml(key, defn, node))
            .join("")}</div>`,
      )
      .join("")}
    <div class="inspector-actions">
      <button data-action="validate">Validate</button>
      ${
        node.type === "camera"
          ? `<button data-action="duplicate-node">Duplicate</button>`
          : ""
      }
      <button class="danger" data-action="remove-node">Remove</button>
    </div>
    ${validationPanelHtml()}
    <div class="hint-box">${
      node.type === "frame_processor"
        ? "Frame-processor settings are shared by every camera that uses this processor. Assignment order is the execution order."
        : node.type === "tracker"
          ? "Trackers are global. Horizontal order is execution order; all pipelines feed into the first tracker."
          : node.type === "utility"
            ? "Utilities are global and run after trackers. Horizontal order is execution order."
            : node.type === "pipeline"
              ? "Each camera has exactly one pipeline. Calibration is configured on the Cameras page."
              : "The camera starts the chain. Source and camera settings sync with the Cameras page."
    }</div>`;
}

function progressCard(node) {
  const progress = node.completion || completion(node.schema, node.settings);
  const missing = progress.missing || [];
  return `<div class="progress-card ${progress.complete ? "complete" : ""}">
    <div class="progress-top"><b>${
      progress.total === 0
        ? "No required fields"
        : progress.complete
          ? "Complete"
          : `${progress.done}/${progress.total} required`
    }</b><span style="margin-left:auto">${progress.percent}%</span></div>
    <div class="progress-track"><div class="progress-fill" style="width:${progress.percent}%"></div></div>
    ${
      missing.length
        ? `<ul class="missing-list">${missing
            .map(
              (key) =>
                `<li>${esc(
                  node.schema?.[key]?.label || key,
                )}</li>`,
            )
            .join("")}</ul>`
        : `<div class="field-help">${
            progress.total === 0
              ? "This node has no required settings."
              : "All required settings are set."
          }</div>`
    }
  </div>`;
}

function nodeWarningsHtml(node) {
  const warnings = node.warnings || [];
  if (!warnings.length) return "";
  return `<div class="calibration-card">${warnings
    .map((warning) => `<div>${esc(warning)}</div>`)
    .join("")}</div>`;
}

function calibrationCard(node) {
  if (node.type === "camera") {
    const calibrated = Boolean(node.calibrated);
    return `<div class="calibration-card ${
      calibrated ? "ok" : ""
    }">Camera calibration: ${
      calibrated ? "configured" : "not configured"
    }. <a href="/cameras">Open Cameras</a></div>`;
  }
  if (node.type !== "pipeline") return "";
  if (!node.requires_calibration) return "";
  const sections = node.calibration_sections || [];
  return `<div class="calibration-card ${
    node.calibrated ? "ok" : ""
  }">
    <b>Calibration ${node.calibrated ? "configured" : "required"}</b>
    ${
      sections.length
        ? `<div>Uses: ${esc(sections.join(", "))}</div>`
        : ""
    }
    <a href="/cameras">Open Cameras</a>
  </div>`;
}

function schemaSections(schema) {
  const groups = new Map();
  for (const [key, defn] of Object.entries(schema || {})) {
    if (!defn || typeof defn !== "object") continue;
    let section =
      defn.section ||
      defn.group ||
      (key.toLowerCase().includes("quant") ||
      key.toLowerCase().includes("optimi")
        ? "Optimization"
        : "Settings");
    if (key === "vision_model") section = "Model";
    if (!groups.has(section)) groups.set(section, []);
    groups.get(section).push([key, defn]);
  }
  return [...groups.entries()];
}

function gateAttr(defn) {
  if (!defn.gated_by) return "";
  const gates = Array.isArray(defn.gated_by)
    ? defn.gated_by
    : [defn.gated_by];
  return `data-gated-by="${esc(gates.join(","))}"`;
}

function helpText(defn) {
  return defn.hint || defn.description || defn.help || "";
}

function fieldHtml(key, defn, node) {
  const required = settingIsRequired(
    node.schema || {},
    node.settings || {},
    key,
  );
  const active = gateActive(
    node.schema || {},
    node.settings || {},
    defn,
  );
  const badges = [];
  if (defn.nullable) badges.push("nullable");
  if (defn.requires_network) badges.push("network");
  return `<div class="field" data-field-key="${esc(key)}" ${gateAttr(
    defn,
  )} ${active ? "" : "hidden"}>
    <div class="field-label"><span>${esc(
      defn.label || key,
    )}</span>${required ? '<span class="required-mark">*</span>' : ""}${
      !required && (defn.nullable || "default" in defn)
        ? '<span class="optional-mark">Optional</span>'
        : ""
    }${badges
      .map((badge) => `<span class="field-badge">${esc(badge)}</span>`)
      .join("")}</div>
    ${controlHtml(key, defn, node)}
    ${
      helpText(defn)
        ? `<div class="field-help">${esc(helpText(defn))}</div>`
        : ""
    }
  </div>`;
}

function optionsFor(defn, current = undefined) {
  let options = defn.options;
  if (defn.options_source === "publish_sources") {
    options = (state.publishSources || []).map((source) => ({
      value: source.source,
      label: `${source.duplicate ? "⚠ " : ""}${
        source.label || source.source
      }`,
    }));
  }
  if (!Array.isArray(options)) return [];
  const mapped = options.map((option) => {
    if (option && typeof option === "object") {
      return {
        value: option.value ?? option.label ?? "",
        label: option.label ?? option.value ?? "",
      };
    }
    return { value: option, label: option };
  });
  if (
    current !== undefined &&
    current !== null &&
    String(current) !== "" &&
    !mapped.some((option) => String(option.value) === String(current))
  ) {
    mapped.push({
      value: current,
      label: `${current} (saved)`,
    });
  }
  return mapped;
}

function controlHtml(key, defn, node) {
  const value = valueFor(node, key);
  const type = defn.type || "text";
  const attrs = `data-setting-key="${esc(
    key,
  )}" data-value-type="${esc(type)}" ${
    defn.nullable ? 'data-nullable="1"' : ""
  }`;
  if (type === "toggle") {
    return `<label class="toggle-row"><input type="checkbox" ${attrs} ${
      value ? "checked" : ""
    } /><span>${value ? "Enabled" : "Disabled"}</span></label>`;
  }
  if (type === "number") {
    const number = value === null || value === undefined ? "" : value;
    return `<input class="field-input" type="number" ${attrs} value="${esc(
      number,
    )}" step="${esc(defn.step ?? "any")}" ${
      defn.min !== undefined ? `min="${esc(defn.min)}"` : ""
    } ${defn.max !== undefined ? `max="${esc(defn.max)}"` : ""} />`;
  }
  if (type === "range") {
    const pair = Array.isArray(value) ? value : [];
    return `<div class="range-row">
      <input class="field-input" type="number" data-setting-key="${esc(
        key,
      )}" data-range-index="0" data-value-type="number" value="${esc(
        pair[0] ?? 0,
      )}" step="${esc(defn.step ?? "any")}" />
      <span>to</span>
      <input class="field-input" type="number" data-setting-key="${esc(
        key,
      )}" data-range-index="1" data-value-type="number" value="${esc(
        pair[1] ?? 0,
      )}" step="${esc(defn.step ?? "any")}" />
    </div>`;
  }
  if (type === "select") {
    const options = optionsFor(defn, value);
    if (!options.length) {
      return `<input class="field-input" type="text" ${attrs} value="${esc(
        value ?? "",
      )}" />`;
    }
    const current = value === null || value === undefined ? "" : String(value);
    return `<select class="field-select" ${attrs}>
      ${defn.nullable ? '<option value="">—</option>' : ""}
      ${options
        .map(
          (option) =>
            `<option value="${esc(option.value)}" ${
              String(option.value) === current ? "selected" : ""
            }>${esc(option.label)}</option>`,
        )
        .join("")}
    </select>`;
  }
  if (type === "list") {
    return renderListControl(key, defn, value, node);
  }
  if (type === "model") {
    return `<div class="field-row">
      <input class="field-input" type="text" ${attrs} value="${esc(
        modelPath(value),
      )}" placeholder="YoloModels/pytorch/model.pt" />
      <button data-action="pick-model" data-setting-key="${esc(
        key,
      )}">Browse</button>
    </div>`;
  }
  if (type === "browse") {
    return `<div class="field-row">
      <input class="field-input" type="text" ${attrs} value="${esc(
        value ?? "",
      )}" placeholder="Server folder path" />
      <button data-action="browse" data-setting-key="${esc(
        key,
      )}">Browse</button>
    </div>`;
  }
  if (type === "source") {
    return `<input class="field-input" type="text" ${attrs} value="${esc(
      value ?? "",
    )}" placeholder="0, device path, or rtsp://…" />`;
  }
  if (defn.multiline) {
    return `<textarea class="field-textarea" ${attrs} rows="3">${esc(
      value ?? "",
    )}</textarea>`;
  }
  return `<input class="field-input" type="text" ${attrs} value="${esc(
    value ?? "",
  )}" />`;
}

function renderListControl(key, defn, value, node) {
  const rows = Array.isArray(value) ? value : [];
  const fields = defn.fields && typeof defn.fields === "object"
    ? defn.fields
    : {};
  let html = `<div class="list-field">`;
  rows.forEach((row, index) => {
    const item = row && typeof row === "object" ? row : {};
    html += `<div class="list-row"><div class="list-row-head">Item ${
      index + 1
    }<button data-action="list-remove" data-list-key="${esc(
      key,
    )}" data-list-index="${index}">Remove</button></div>`;
    if (!Object.keys(fields).length) {
      html += `<textarea class="field-textarea" data-list-json="${esc(
        key,
      )}" rows="3">${esc(JSON.stringify(item, null, 2))}</textarea>`;
    } else {
      for (const [fieldKey, fieldDefn] of Object.entries(fields)) {
        if (!fieldDefn || typeof fieldDefn !== "object") continue;
        html += `<div class="field" style="margin-bottom:6px"><div class="field-label"><span>${esc(
          fieldDefn.label || fieldKey,
        )}</span>${
          fieldDefn.nullable
            ? '<span class="optional-mark">Optional</span>'
            : ""
        }</div>${listFieldControl(
          key,
          index,
          fieldKey,
          fieldDefn,
          item[fieldKey],
          node,
        )}</div>`;
      }
    }
    html += `</div>`;
  });
  html += `<button class="list-add" data-action="list-add" data-list-key="${esc(
    key,
  )}">+ Add item</button></div>`;
  return html;
}

function listFieldControl(
  listKey,
  index,
  fieldKey,
  defn,
  value,
  node,
) {
  const attrs = `data-list-key="${esc(
    listKey,
  )}" data-list-index="${index}" data-list-field="${esc(
    fieldKey,
  )}" data-value-type="${esc(defn.type || "text")}"`;
  if (defn.type === "toggle") {
    return `<label class="toggle-row"><input type="checkbox" ${attrs} ${
      value ? "checked" : ""
    } /><span>${value ? "Enabled" : "Disabled"}</span></label>`;
  }
  if (defn.type === "select") {
    const options = optionsFor(defn, value);
    if (!options.length) {
      return `<input class="field-input" type="text" ${attrs} value="${esc(
        value ?? "",
      )}" />`;
    }
    const current = value === null || value === undefined ? "" : String(value);
    return `<select class="field-select" ${attrs}>${options
      .map(
        (option) =>
          `<option value="${esc(option.value)}" ${
            String(option.value) === current ? "selected" : ""
          }>${esc(option.label)}</option>`,
      )
      .join("")}</select>`;
  }
  if (defn.type === "number") {
    return `<input class="field-input" type="number" ${attrs} value="${esc(
      value ?? "",
    )}" step="${esc(defn.step ?? "any")}" ${
      defn.min !== undefined ? `min="${esc(defn.min)}"` : ""
    } ${defn.max !== undefined ? `max="${esc(defn.max)}"` : ""} />`;
  }
  return `<input class="field-input" type="text" ${attrs} value="${esc(
    value ?? "",
  )}" />`;
}

function defaultRow(fields) {
  const row = {};
  for (const [key, defn] of Object.entries(fields || {})) {
    if (!defn || typeof defn !== "object") continue;
    if ("default" in defn) row[key] = deep(defn.default);
    else if (defn.type === "toggle") row[key] = false;
    else if (defn.type === "number") row[key] = 0;
    else row[key] = "";
  }
  return row;
}

function validationPanelHtml() {
  const errors = state.serverErrors || [];
  const warnings = state.serverWarnings || [];
  if (!errors.length && !warnings.length) return "";
  return `<div class="validation-card"><h3>Validation</h3>
    ${errors
      .map(
        (error) =>
          `<div class="validation-line error"><span>${esc(
            error.message || error.code,
          )}</span>${
            error.node
              ? `<button data-action="focus-node" data-node-id="${esc(
                  error.node,
                )}">Show</button>`
              : ""
          }</div>`,
      )
      .join("")}
    ${warnings
      .map(
        (warning) =>
          `<div class="validation-line warn"><span>${esc(
            warning.message || warning.code,
          )}</span>${
            warning.node
              ? `<button data-action="focus-node" data-node-id="${esc(
                  warning.node,
                )}">Show</button>`
              : ""
          }</div>`,
      )
      .join("")}
  </div>`;
}

function refreshInspectorProgress() {
  const node = selectedNode();
  if (!node) return;
  const card = el.inspector.querySelector(".progress-card");
  if (card) {
    const holder = document.createElement("div");
    holder.innerHTML = progressCard(node);
    card.replaceWith(holder.firstElementChild);
  }
  applyGates();
  renderPalette();
}

function applyGates() {
  const node = selectedNode();
  if (!node) return;
  el.inspector
    .querySelectorAll(".field[data-field-key]")
    .forEach((field) => {
      const key = field.dataset.fieldKey;
      const defn = node.schema?.[key];
      if (!defn) return;
      field.hidden = !gateActive(
        node.schema || {},
        node.settings || {},
        defn,
      );
    });
}

function valueFromInput(input, defn, node, key, previous) {
  const type = input.dataset.valueType || defn.type || "text";
  if (type === "toggle") return Boolean(input.checked);
  if (type === "number") {
    const raw = input.value.trim();
    if (!raw) return defn.nullable ? null : "";
    const number = Number(raw);
    return Number.isFinite(number) ? number : raw;
  }
  if (type === "model") {
    const raw = input.value.trim();
    if (!raw) return defn.nullable ? null : "";
    const base =
      previous && typeof previous === "object" ? deep(previous) : {};
    return { ...base, file_path: raw, source_pt: raw };
  }
  if (type === "browse" || type === "source") {
    const raw = input.value;
    if (!raw.trim()) return defn.nullable ? null : "";
    return raw;
  }
  if (type === "select" && input.value === "" && defn.nullable) {
    return null;
  }
  if (type === "select" && input.value === "") return "";
  return input.value;
}

function syncSharedSettings(node) {
  if (!node || node.type !== "frame_processor" || !node.plugin) return;
  for (const other of state.nodes) {
    if (
      other !== node &&
      other.type === "frame_processor" &&
      other.plugin === node.plugin
    ) {
      other.settings = deep(node.settings);
      other.completion = completion(other.schema || {}, other.settings);
      other.required = other.completion.required;
    }
  }
}

function applyInspectorValue(input) {
  const node = selectedNode();
  if (!node) return null;
  if (input.dataset.listKey) {
    const listKey = input.dataset.listKey;
    const index = Number(input.dataset.listIndex);
    const field = input.dataset.listField;
    if (!field) return node;
    const listDefn = node.schema?.[listKey] || {};
    const fieldDefn = listDefn.fields?.[field] || {
      type: input.dataset.valueType || "text",
    };
    const rows = Array.isArray(node.settings[listKey])
      ? deep(node.settings[listKey])
      : [];
    if (!rows[index] || typeof rows[index] !== "object") {
      rows[index] = {};
    }
    rows[index][field] = valueFromInput(
      input,
      fieldDefn,
      node,
      field,
      rows[index][field],
    );
    node.settings[listKey] = rows;
    syncSharedSettings(node);
    return node;
  }
  const key = input.dataset.settingKey;
  if (!key) return node;
  const defn = node.schema?.[key] || {
    type: input.dataset.valueType || "text",
  };
  if (input.dataset.rangeIndex !== undefined) {
    const pair = Array.isArray(node.settings[key])
      ? deep(node.settings[key])
      : [0, 0];
    const index = Number(input.dataset.rangeIndex);
    pair[index] = valueFromInput(
      input,
      { ...defn, type: "number", nullable: false },
      node,
      key,
      pair[index],
    );
    node.settings[key] = pair;
  } else {
    node.settings[key] = valueFromInput(
      input,
      defn,
      node,
      key,
      node.settings[key],
    );
  }
  syncSharedSettings(node);
  return node;
}

function handleInspectorInput(event) {
  const input = event.target.closest("input, select, textarea");
  if (!input) return;
  if (
    input.type === "checkbox" ||
    input.tagName === "SELECT" ||
    input.dataset.listJson
  ) {
    return;
  }
  if (!state.editBefore) state.editBefore = snapshot();
  if (!applyInspectorValue(input)) return;
  markDirty();
  updateAllCompletions();
  renderNodes();
  renderEdges();
  refreshInspectorProgress();
  renderStatus();
}

function handleInspectorChange(event) {
  const input = event.target.closest("input, select, textarea");
  if (!input) return;
  if (!state.editBefore) state.editBefore = snapshot();
  if (input.dataset.listJson) {
    const node = selectedNode();
    const key = input.dataset.listJson;
    if (node) {
      try {
        const parsed = JSON.parse(input.value);
        if (!Array.isArray(parsed)) throw new Error("expected an array");
        node.settings[key] = parsed;
      } catch (error) {
        toast("List JSON must be an array.", "error");
      }
    }
  } else if (
    input.type === "checkbox" ||
    input.tagName === "SELECT" ||
    input.dataset.settingKey ||
    input.dataset.listKey
  ) {
    applyInspectorValue(input);
  }
  commitEdit();
  updateAllCompletions();
  renderNodes();
  renderEdges();
  renderStatus();
  if (
    input.type === "checkbox" ||
    input.tagName === "SELECT" ||
    input.dataset.listJson
  ) {
    renderInspector();
    renderPalette();
  } else {
    refreshInspectorProgress();
    applyGates();
  }
}

function browseForSetting(node, key) {
  const defn = node.schema?.[key] || { type: "browse" };
  const current = valueFor(node, key);
  const before = snapshot();
  let currentPath =
    typeof current === "string" ? current.trim() : "";
  const modal = openModal(
    `Choose folder for ${defn.label || key}`,
    "<p>Loading folders…</p>",
    [
      { label: "Cancel" },
      {
        label: "Use this folder",
        primary: true,
        onClick: () => {
          if (!currentPath) return false;
          node.settings[key] = currentPath;
          syncSharedSettings(node);
          pushHistory(before);
          markDirty();
          updateAllCompletions();
          renderAll();
          return true;
        },
      },
    ],
  );
  const load = async (path) => {
    modal.body.innerHTML = "<p>Loading folders…</p>";
    try {
      const response = await fetch(
        `/api/fs/dirs?path=${encodeURIComponent(path || "")}`,
        { cache: "no-store" },
      );
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}`);
      }
      currentPath = data.path || path || "";
      const dirs = Array.isArray(data.dirs) ? data.dirs : [];
      modal.body.innerHTML = `<div class="hint-box" style="margin-bottom:9px"><b>${esc(
        data.name || currentPath || "Project folder",
      )}</b><br /><span style="word-break:break-all">${esc(
        currentPath,
      )}</span></div>
        <div class="picker-list">
          ${
            data.parent
              ? `<button class="picker-item" data-browse-parent><span><b>↑ Parent folder</b><small>${esc(
                  data.parent,
                )}</small></span></button>`
              : ""
          }
          ${dirs
            .map(
              (dir) =>
                `<button class="picker-item" data-browse-path="${esc(
                  dir.path,
                )}"><span><b>${esc(dir.name)}</b><small>${esc(
                  dir.path,
                )}</small></span></button>`,
            )
            .join("")}
        </div>`;
      modal.body
        .querySelectorAll("[data-browse-path]")
        .forEach((button) => {
          button.addEventListener("click", () =>
            load(button.dataset.browsePath),
          );
        });
      const parent = modal.body.querySelector("[data-browse-parent]");
      if (parent) {
        parent.addEventListener("click", () => load(data.parent));
      }
    } catch (error) {
      modal.body.innerHTML = `<div class="validation-card"><h3>Folder unavailable</h3><div class="validation-line error"><span>${esc(
        error.message,
      )}</span></div></div>`;
    }
  };
  load(currentPath);
}

function handleInspectorClick(event) {
  const target = event.target.closest("[data-action]");
  if (!target) return;
  const action = target.dataset.action;
  const node = selectedNode();
  if (action === "validate") {
    validateGraph();
    return;
  }
  if (action === "add-camera") {
    openPicker("camera");
    return;
  }
  if (action === "help") {
    openShortcuts();
    return;
  }
  if (action === "clear-edge") {
    state.selectedEdge = null;
    renderNodes();
    renderEdges();
    renderInspector();
    return;
  }
  if (action === "focus-node") {
    focusNode(target.dataset.nodeId);
    return;
  }
  if (!node) return;
  if (action === "remove-node") {
    removeNodes([node.id]);
    return;
  }
  if (action === "remove-selected") {
    removeNodes([...state.selected]);
    return;
  }
  if (action === "duplicate-node") {
    duplicateNode(node);
    return;
  }
  if (action === "browse") {
    browseForSetting(node, target.dataset.settingKey);
    return;
  }
  if (action === "pick-model") {
    openModelPicker(node, target.dataset.settingKey);
    return;
  }
  if (action === "list-add") {
    const key = target.dataset.listKey;
    const defn = node.schema?.[key] || {};
    const rows = Array.isArray(node.settings[key])
      ? deep(node.settings[key])
      : [];
    commitEdit();
    const before = snapshot();
    rows.push(defaultRow(defn.fields || {}));
    node.settings[key] = rows;
    syncSharedSettings(node);
    pushHistory(before);
    markDirty();
    updateAllCompletions();
    renderAll();
    return;
  }
  if (action === "list-remove") {
    const key = target.dataset.listKey;
    const index = Number(target.dataset.listIndex);
    const rows = Array.isArray(node.settings[key])
      ? deep(node.settings[key])
      : [];
    commitEdit();
    const before = snapshot();
    rows.splice(index, 1);
    node.settings[key] = rows;
    syncSharedSettings(node);
    pushHistory(before);
    markDirty();
    updateAllCompletions();
    renderAll();
  }
}

function openModal(title, bodyHtml, actions = []) {
  closeModal();
  const root = el.modalRoot;
  root.innerHTML = `<div class="modal-backdrop"><div class="modal-card" role="dialog" aria-modal="true"><h3>${esc(
    title,
  )}</h3><div class="modal-body">${bodyHtml}</div><div class="modal-actions"></div></div></div>`;
  const backdrop = root.querySelector(".modal-backdrop");
  backdrop.addEventListener("pointerdown", (event) => {
    if (event.target === backdrop) closeModal();
  });
  const actionsEl = root.querySelector(".modal-actions");
  for (const action of actions) {
    const button = document.createElement("button");
    button.textContent = action.label;
    if (action.primary) button.className = "primary";
    if (action.danger) button.className = "danger";
    button.addEventListener("click", () => {
      const close = action.onClick ? action.onClick() : true;
      if (close !== false) closeModal();
    });
    actionsEl.appendChild(button);
  }
  return {
    root,
    body: root.querySelector(".modal-body"),
    close: closeModal,
  };
}

function closeModal() {
  if (state.confirmResolve) {
    const resolve = state.confirmResolve;
    state.confirmResolve = null;
    resolve(false);
  }
  if (el.modalRoot) el.modalRoot.innerHTML = "";
}

function openPicker(type, camera = null) {
  const groups = {
    camera: ["cameras", "camera", "Add camera"],
    pipeline: ["pipelines", "pipeline", "Add pipeline"],
    frame_processor: [
      "frame_processors",
      "frame_processor",
      "Add frame processor",
    ],
    tracker: ["trackers", "tracker", "Add tracker"],
    utility: ["utilities", "utility", "Add utility"],
  };
  const [group, dotType, title] = groups[type] || [];
  if (!group) return;
  let cameraNode = null;
  if (type !== "camera") {
    cameraNode =
      typeof camera === "string" ? getCamera(camera) : camera;
    if (!cameraNode) cameraNode = selectedCamera();
    if (!cameraNode) {
      toast("Select a camera node first.", "warning");
      return;
    }
  }
  const items = state.palette[group] || [];
  const body = items.length
    ? `<div class="picker-list">${items
        .map((item, index) => {
          const subtitle = (item.doc || item.class_name || "")
            .replace(/\s+/g, " ")
            .trim()
            .slice(0, 90);
          return `<button class="picker-item" data-picker-index="${index}">
            <span class="palette-dot ${dotType}"></span>
            <span><b>${esc(item.label || item.name)}</b><small>${esc(
              subtitle || item.name,
            )}</small></span>
            ${
              item.needs_internet || item.beta || item.template
                ? `<small>${esc(
                    [
                      item.template ? "template" : "",
                      item.beta ? "beta" : "",
                      item.needs_internet ? "internet" : "",
                    ]
                      .filter(Boolean)
                      .join(" · "),
                  )}</small>`
                : ""
            }
          </button>`;
        })
        .join("")}</div>`
    : `<p>No ${esc(type.replaceAll("_", " "))} plugins were discovered.</p>`;
  const modal = openModal(title, body, [{ label: "Cancel" }]);
  modal.root.querySelectorAll("[data-picker-index]").forEach((button) => {
    button.addEventListener("click", () => {
      const item = items[Number(button.dataset.pickerIndex)];
      modal.close();
      if (!item) return;
      if (type === "camera") addCamera(item.name);
      else if (type === "pipeline") addPipeline(item, cameraNode);
      else if (type === "frame_processor") {
        addFrameProcessor(item, cameraNode);
      } else {
        addGlobal(item, type);
      }
    });
  });
}

async function openModelPicker(node, key) {
  const modal = openModal(
    "Choose model",
    "<p>Loading available models…</p>",
    [{ label: "Cancel" }],
  );
  try {
    const response = await fetch("/api/models", { cache: "no-store" });
    const data = await response.json();
    const models = data.models || [];
    modal.body.innerHTML = models.length
      ? `<div class="picker-list">${models
          .map(
            (model) =>
              `<button class="picker-item" data-model-name="${esc(
                model.name,
              )}"><span><b>${esc(model.name)}</b><small>${esc(
                model.task || "model",
              )} · ${esc(model.size_mb ?? "?")} MB</small></span></button>`,
          )
          .join("")}</div>`
      : `<p>No models found. Upload one on the Models page.</p><p><a href="/models">Open Models</a></p>`;
    modal.root.querySelectorAll("[data-model-name]").forEach((button) => {
      button.addEventListener("click", () => {
        const name = button.dataset.modelName;
        const path = `YoloModels/pytorch/${name}`;
        const before = snapshot();
        node.settings[key] = { file_path: path, source_pt: path };
        syncSharedSettings(node);
        pushHistory(before);
        markDirty();
        updateAllCompletions();
        modal.close();
        renderAll();
        toast(`Selected ${name}.`, "success");
      });
    });
  } catch (error) {
    modal.body.innerHTML = `<p>Could not load models. Enter the path manually.</p>`;
  }
}

function openShortcuts() {
  openModal(
    "Keyboard shortcuts",
    `<div class="hint-box" style="line-height:1.9">
      <kbd>Ctrl</kbd>+<kbd>S</kbd> save &nbsp; · &nbsp;
      <kbd>Ctrl</kbd>+<kbd>Z</kbd> undo &nbsp; · &nbsp;
      <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd> redo<br />
      <kbd>F</kbd> fit &nbsp; · &nbsp; <kbd>+</kbd>/<kbd>−</kbd> zoom &nbsp; · &nbsp;
      <kbd>0</kbd> reset view<br />
      <kbd>Shift</kbd>+drag background select &nbsp; · &nbsp;
      <kbd>Ctrl</kbd>+click multi-select<br />
      <kbd>←</kbd>/<kbd>→</kbd> nudge selected &nbsp; · &nbsp;
      <kbd>Delete</kbd> remove &nbsp; · &nbsp;
      <kbd>Ctrl</kbd>+<kbd>D</kbd> duplicate camera<br />
      Drag an output port to another node to connect. Invalid pairs are rejected.
    </div>`,
    [{ label: "Close", primary: true }],
  );
}

function openContextMenu(x, y, node) {
  closeContextMenu();
  if (!node) return;
  const actions = [
    { id: "edit", label: "Edit settings" },
    { id: "focus", label: "Center node" },
  ];
  if (node.type === "camera") {
    actions.push({ id: "duplicate", label: "Duplicate camera" });
  }
  if (
    ["camera", "pipeline", "frame_processor"].includes(node.type) &&
    node.camera
  ) {
    const gate = addGate(state.nodes, "frame_processor", {
      camera: node.camera,
    });
    actions.push({
      id: "add-frame-processor",
      label: gate.allowed ? "Add frame processor" : "Add frame processor (locked)",
      disabled: !gate.allowed,
      title: gate.reason,
    });
  }
  actions.push({ id: "remove", label: "Remove", danger: true });
  el.contextRoot.innerHTML = `<div class="context-menu" role="menu">${actions
    .map(
      (action) =>
        `<button role="menuitem" data-context-action="${action.id}" class="${
          action.danger ? "danger" : ""
        }" ${action.disabled ? 'disabled aria-disabled="true"' : ""} title="${esc(
          action.title || "",
        )}">${esc(action.label)}</button>`,
    )
    .join("")}</div>`;
  const menu = el.contextRoot.querySelector(".context-menu");
  menu.style.left = `${Math.round(x)}px`;
  menu.style.top = `${Math.round(y)}px`;
  const rect = menu.getBoundingClientRect();
  if (x + rect.width > window.innerWidth - 8) {
    menu.style.left = `${Math.max(8, x - rect.width)}px`;
  }
  if (y + rect.height > window.innerHeight - 8) {
    menu.style.top = `${Math.max(8, y - rect.height)}px`;
  }
  menu.querySelectorAll("[data-context-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.dataset.contextAction;
      closeContextMenu();
      if (action === "edit") {
        state.selected = new Set([node.id]);
        state.selectedEdge = null;
        renderNodes();
        renderInspector();
        renderStatus();
      } else if (action === "focus") {
        focusNode(node.id);
      } else if (action === "duplicate") {
        duplicateNode(node);
      } else if (action === "add-frame-processor") {
        openPicker("frame_processor", node.camera);
      } else if (action === "remove") {
        removeNodes([node.id]);
      }
    });
  });
}

function closeContextMenu() {
  if (el.contextRoot) el.contextRoot.innerHTML = "";
}

function makeNode({
  id,
  type,
  label,
  camera = null,
  plugin = null,
  schema = {},
  settings = {},
  x = 0,
  order = 0,
  subtitle = "",
  cameraType = null,
  warnings = [],
  needsInternet = false,
  extra = {},
}) {
  const node = {
    id,
    type,
    category: type,
    label,
    camera,
    plugin,
    camera_type: cameraType,
    schema: deep(schema || {}),
    settings: deep(settings || {}),
    x: snap(x),
    y: rowY(type),
    order,
    subtitle,
    warnings: [...warnings],
    needs_internet: Boolean(needsInternet),
    ...extra,
  };
  node.completion = completion(node.schema, node.settings);
  node.required = node.completion.required;
  return node;
}

function addCamera(typeName) {
  const data = (state.palette.cameras || []).find(
    (item) => item.name === typeName,
  );
  if (!data) {
    toast("That camera type is not available.", "error");
    return;
  }
  const cameras = getNodes("camera");
  const x = cameras.length
    ? Math.max(...cameras.map((node) => node.x)) + 460
    : 80;
  const name = uniqueCameraName(state.nodes, "camera");
  const settings = {
    ...deep(data.defaults || {}),
    camera_type: typeName,
  };
  const node = makeNode({
    id: `camera:${name}`,
    type: "camera",
    label: name,
    camera: name,
    schema: data.schema || {},
    settings,
    x,
    subtitle: `${data.label || typeName} · ${settings.source ?? ""}`,
    cameraType: typeName,
  });
  const before = snapshot();
  state.nodes.push(node);
  normalizeNodeOrder();
  rebuildEdges();
  state.selected = new Set([node.id]);
  state.selectedEdge = null;
  pushHistory(before);
  markDirty();
  renderAll();
  toast(`Added camera '${name}'. Configure its source to continue.`, "success");
}

function addPipeline(item, camera) {
  if (!item || !camera) return;
  const gate = addGate(state.nodes, "pipeline", { camera });
  if (!gate.allowed) {
    toast(gate.reason, "warning");
    return;
  }
  const processors = getProcessors(camera.camera);
  const lastX = processors.length
    ? Math.max(...processors.map((node) => node.x))
    : camera.x;
  const node = makeNode({
    id: `pipeline:${camera.camera}`,
    type: "pipeline",
    label: item.label || item.name,
    camera: camera.camera,
    plugin: item.name,
    schema: item.schema || {},
    settings: item.defaults || {},
    x: Math.max(camera.x + 680, lastX + 280),
    subtitle: item.doc || item.class_name || "",
    needsInternet: item.needs_internet,
    extra: {
      beta: Boolean(item.beta),
      builtin: Boolean(item.template),
      requires_calibration: Boolean(item.requires_calibration),
      calibration_sections: deep(item.calibration_sections || []),
      calibrated: Boolean(camera.calibrated),
      show_common_fields: item.show_common_fields !== false,
      supported_pipelines: deep(item.supported_pipelines || []),
    },
  });
  const before = snapshot();
  state.nodes.push(node);
  normalizeNodeOrder();
  rebuildEdges();
  state.selected = new Set([node.id]);
  state.selectedEdge = null;
  pushHistory(before);
  markDirty();
  renderAll();
  toast(`Added pipeline '${node.label}' for ${camera.camera}.`, "success");
}

function addFrameProcessor(item, camera) {
  if (!item || !camera) return;
  const gate = addGate(state.nodes, "frame_processor", {
    camera,
    plugin: item.name,
  });
  if (!gate.allowed) {
    toast(gate.reason, "warning");
    return;
  }
  const processors = getProcessors(camera.camera);
  const shared = state.nodes.find(
    (node) => node.type === "frame_processor" && node.plugin === item.name,
  );
  const pipeline = getPipeline(camera.camera);
  const order = nextOrder(
    state.nodes,
    (node) =>
      node.type === "frame_processor" && node.camera === camera.camera,
  );
  let x = processors.length
    ? Math.max(...processors.map((node) => node.x)) + 260
    : camera.x + 420;
  if (pipeline && x + 216 > pipeline.x) {
    pipeline.x = snap(x + 300);
  }
  const node = makeNode({
    id: `frame_processor:${camera.camera}:${item.name}`,
    type: "frame_processor",
    label: item.label || item.name,
    camera: camera.camera,
    plugin: item.name,
    schema: shared?.schema || item.schema || {},
    settings: shared?.settings || item.defaults || {},
    x,
    order,
    subtitle: item.doc || item.class_name || "",
    needsInternet: item.needs_internet,
    extra: {
      supported_pipelines: deep(item.supported_pipelines || []),
      builtin: Boolean(item.template),
    },
  });
  const before = snapshot();
  state.nodes.push(node);
  normalizeNodeOrder();
  rebuildEdges();
  state.selected = new Set([node.id]);
  state.selectedEdge = null;
  pushHistory(before);
  markDirty();
  renderAll();
  toast(`Assigned '${node.label}' to ${camera.camera}.`, "success");
}

function addGlobal(item, type) {
  if (!item) return;
  if (!["tracker", "utility"].includes(type)) return;
  const gate = addGate(state.nodes, type, { plugin: item.name });
  if (!gate.allowed) {
    toast(gate.reason, "warning");
    return;
  }
  const group = getNodes(type);
  const order = nextOrder(state.nodes, (node) => node.type === type);
  const x = group.length
    ? Math.max(...group.map((node) => node.x)) + 300
    : 80;
  const node = makeNode({
    id: `${type}:${item.name}`,
    type,
    label: item.label || item.name,
    plugin: item.name,
    schema: item.schema || {},
    settings: item.defaults || {},
    x,
    order,
    subtitle: item.doc || item.class_name || "",
    needsInternet: item.needs_internet,
    extra: {
      supported_pipelines: deep(item.supported_pipelines || []),
      builtin: Boolean(item.template),
    },
  });
  const before = snapshot();
  state.nodes.push(node);
  normalizeNodeOrder();
  rebuildEdges();
  state.selected = new Set([node.id]);
  state.selectedEdge = null;
  pushHistory(before);
  markDirty();
  renderAll();
  toast(`Added ${type} '${node.label}'.`, "success");
}

async function removeNodes(ids) {
  if (state.confirmResolve) return;
  const plan = removalPlan(state.nodes, ids);
  if (!plan.selectedIds.length) return;
  if (plan.blocked) {
    toast(plan.reason, "warning");
    return;
  }
  let message = plan.cameraNames.length
    ? `Remove ${plan.cameraNames.length} camera${
        plan.cameraNames.length === 1 ? "" : "s"
      } and ${plan.chainCount} node${
        plan.chainCount === 1 ? "" : "s"
      }? The camera config will be deleted when you save.`
    : `Remove ${plan.chainCount} selected node${
        plan.chainCount === 1 ? "" : "s"
      }?`;
  if (plan.orphanedCameras.length) {
    message += ` Camera${
      plan.orphanedCameras.length === 1 ? "" : "s"
    } ${plan.orphanedCameras
      .map((camera) => `'${camera}'`)
      .join(", ")} will have no pipeline. The graph cannot be saved until you add one.`;
  }
  if (!(await confirmDialog("Remove nodes", message))) return;
  const removeSet = new Set(plan.removeIds);
  const before = snapshot();
  state.nodes = state.nodes.filter((node) => !removeSet.has(node.id));
  state.selected = new Set(
    [...state.selected].filter((id) => !removeSet.has(id)),
  );
  state.selectedEdge = null;
  normalizeNodeOrder();
  rebuildEdges();
  pushHistory(before);
  markDirty();
  renderAll();
  toast("Removed from the editor. Save to write the change.", "info");
}

function duplicateNode(node) {
  if (!node || node.type !== "camera") {
    toast("Only camera nodes can be duplicated; plugin nodes are unique.", "info");
    return;
  }
  const name = uniqueCameraName(state.nodes, `${node.camera}_copy`);
  const x = node.x + 260;
  const clone = makeNode({
    id: `camera:${name}`,
    type: "camera",
    label: name,
    camera: name,
    schema: node.schema,
    settings: node.settings,
    x,
    subtitle: node.subtitle,
    cameraType: node.camera_type,
    warnings: node.warnings,
  });
  const before = snapshot();
  state.nodes.push(clone);
  normalizeNodeOrder();
  rebuildEdges();
  state.selected = new Set([clone.id]);
  state.selectedEdge = null;
  pushHistory(before);
  markDirty();
  renderAll();
  toast(`Duplicated camera as '${name}'.`, "success");
}

function buildPayload() {
  return nodePayload(state.nodes, state.edges, state.viewport, state.grid);
}

async function validateGraph({ silent = false } = {}) {
  if (state.busy) return false;
  setBusy(true);
  try {
    const response = await fetch("/api/editor/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    let data = {};
    try {
      data = await response.json();
    } catch (error) {
      data = { ok: false, errors: [{ code: "response", message: "Validation returned an invalid response." }] };
    }
    state.serverErrors = Array.isArray(data.errors) ? data.errors : [];
    state.serverWarnings = Array.isArray(data.warnings) ? data.warnings : [];
    if (data.completion && typeof data.completion === "object") {
      for (const node of state.nodes) {
        if (data.completion[node.id]) {
          node.completion = data.completion[node.id];
          node.required = node.completion.required || [];
        }
      }
    }
    if (!state.serverErrors.length && !response.ok) {
      state.serverErrors = [
        {
          code: "http",
          message: `Validation failed with HTTP ${response.status}.`,
        },
      ];
    }
    const ok = Boolean(data.ok) && response.ok;
    renderAll();
    if (!silent) {
      if (ok) toast("Validation passed.", "success");
      else toast("Validation found errors. Review the inspector.", "error");
    }
    return ok;
  } catch (error) {
    if (!silent) toast("Could not validate the graph.", "error");
    return false;
  } finally {
    setBusy(false);
  }
}

async function saveGraph() {
  if (state.busy) return;
  const valid = await validateGraph({ silent: true });
  if (!valid) {
    toast("Fix validation errors before saving.", "error");
    return;
  }
  setBusy(true);
  try {
    const response = await fetch("/api/editor/graph", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    let data = {};
    try {
      data = await response.json();
    } catch (error) {
      data = {};
    }
    if (!response.ok || !data.ok) {
      state.serverErrors = Array.isArray(data.errors)
        ? data.errors
        : [
            {
              code: "save",
              message: data.error || `Save failed with HTTP ${response.status}.`,
            },
          ];
      state.serverWarnings = Array.isArray(data.warnings)
        ? data.warnings
        : [];
      renderAll();
      toast("Could not save the graph.", "error");
      return;
    }
    state.restartRequired = true;
    state.dirty = false;
    state.serverErrors = [];
    const savedWarnings = Array.isArray(data.warnings)
      ? data.warnings
      : [];
    state.serverWarnings = savedWarnings;
    if (data.graph) {
      applyGraph(
        {
          graph: data.graph,
          palette: data.palette || state.palette,
          restart_required: true,
        },
        { first: false },
      );
    } else {
      await loadGraph({ silent: true });
    }
    state.serverWarnings = savedWarnings;
    renderAll();
    showRestartBanner();
    renderStatus();
    toast(
      "Saved. Restart iSpy to apply camera and pipeline changes.",
      "success",
    );
  } catch (error) {
    toast("Could not save the graph.", "error");
  } finally {
    setBusy(false);
  }
}

function restoreSnapshot(snapshotText) {
  let data;
  try {
    data = JSON.parse(snapshotText);
  } catch (error) {
    return;
  }
  state.nodes = (data.nodes || []).map(normalizeNode);
  state.edges = (data.edges || []).map((edge) => ({ ...edge }));
  state.selected = new Set(data.selected || []);
  state.selectedEdge = null;
  state.editBefore = null;
  updateAllCompletions();
  renderAll();
}

function undo() {
  if (state.busy) return;
  if (state.editBefore) commitEdit();
  if (!state.history.length) return;
  const current = snapshot();
  const previous = state.history.pop();
  state.future.push(current);
  restoreSnapshot(previous);
  state.dirty = true;
  state.serverErrors = [];
  state.serverWarnings = [];
  renderStatus();
  toast("Undo.", "info");
}

function redo() {
  if (state.busy) return;
  if (state.editBefore) commitEdit();
  if (!state.future.length) return;
  const current = snapshot();
  const next = state.future.pop();
  state.history.push(current);
  restoreSnapshot(next);
  state.dirty = true;
  state.serverErrors = [];
  state.serverWarnings = [];
  renderStatus();
  toast("Redo.", "info");
}

function confirmDialog(title, message, confirmLabel = "Remove") {
  return new Promise((resolve) => {
    const root = el.modalRoot;
    root.innerHTML = `<div class="modal-backdrop"><div class="modal-card" role="alertdialog" aria-modal="true"><h3>${esc(
      title,
    )}</h3><div class="modal-body"><p>${esc(
      message,
    )}</p></div><div class="modal-actions"><button data-confirm-cancel>Cancel</button><button class="danger" data-confirm-ok>${esc(
      confirmLabel,
    )}</button></div></div></div>`;
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      state.confirmResolve = null;
      root.innerHTML = "";
      resolve(value);
    };
    state.confirmResolve = finish;
    const backdrop = root.querySelector(".modal-backdrop");
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) finish(false);
    });
    root
      .querySelector("[data-confirm-cancel]")
      .addEventListener("click", () => finish(false));
    root
      .querySelector("[data-confirm-ok]")
      .addEventListener("click", () => finish(true));
  });
}

function moveSelected(delta) {
  const nodes = selectedNodes();
  if (!nodes.length) return;
  const before = snapshot();
  for (const node of nodes) {
    node.x = snap(node.x + delta, state.grid);
  }
  pushHistory(before);
  markDirty();
  renderNodes();
  renderEdges();
  renderStatus();
}

function addPaletteItem(type, name) {
  if (type === "camera") {
    addCamera(name);
    return;
  }
  const groups = {
    pipeline: "pipelines",
    frame_processor: "frame_processors",
    tracker: "trackers",
    utility: "utilities",
  };
  const group = groups[type];
  if (!group) return;
  const item = (state.palette[group] || []).find(
    (entry) => entry.name === name,
  );
  if (!item) {
    toast("That plugin is not available.", "error");
    return;
  }
  if (type === "tracker" || type === "utility") {
    addGlobal(item, type);
    return;
  }
  const camera = selectedCamera();
  if (!camera) {
    toast("Select a camera node first.", "warning");
    return;
  }
  if (type === "pipeline") addPipeline(item, camera);
  else addFrameProcessor(item, camera);
}

function isEditingTarget(target) {
  return (
    target instanceof HTMLElement &&
    (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) ||
      target.isContentEditable)
  );
}

function onKeyDown(event) {
  if (event.key === "Escape") {
    if (el.modalRoot?.innerHTML) {
      closeModal();
      event.preventDefault();
      return;
    }
    if (el.contextRoot?.innerHTML) {
      closeContextMenu();
      event.preventDefault();
      return;
    }
    if (state.connect) {
      cancelConnect();
      renderNodes();
      event.preventDefault();
      return;
    }
    if (state.marquee) {
      state.marquee = null;
      el.marquee.hidden = true;
      event.preventDefault();
      return;
    }
    clearSelection();
    return;
  }
  const modifier = event.ctrlKey || event.metaKey;
  const key = event.key.toLowerCase();
  if (modifier && key === "s") {
    event.preventDefault();
    saveGraph();
    return;
  }
  if (el.modalRoot?.innerHTML) return;
  if (isEditingTarget(event.target)) return;
  if (modifier && key === "z" && !event.shiftKey) {
    event.preventDefault();
    undo();
    return;
  }
  if (
    modifier &&
    ((key === "z" && event.shiftKey) || key === "y")
  ) {
    event.preventDefault();
    redo();
    return;
  }
  if (modifier && key === "a") {
    event.preventDefault();
    state.selected = new Set(state.nodes.map((node) => node.id));
    state.selectedEdge = null;
    renderNodes();
    renderInspector();
    renderStatus();
    return;
  }
  if (modifier && key === "d") {
    event.preventDefault();
    const camera = selectedNodes().find((node) => node.type === "camera");
    if (camera) duplicateNode(camera);
    else toast("Select a camera to duplicate.", "info");
    return;
  }
  if (event.key === "Delete" || event.key === "Backspace") {
    if (!state.selected.size) return;
    event.preventDefault();
    removeNodes([...state.selected]);
    return;
  }
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    if (!state.selected.size) return;
    event.preventDefault();
    const step = state.grid * (event.shiftKey ? 5 : 1);
    moveSelected(event.key === "ArrowLeft" ? -step : step);
    return;
  }
  if (key === "f") {
    event.preventDefault();
    fitToNodes();
    return;
  }
  if (event.key === "+" || event.key === "=") {
    event.preventDefault();
    const rect = el.canvas.getBoundingClientRect();
    zoomAt(
      1.2,
      rect.left + rect.width / 2,
      rect.top + rect.height / 2,
    );
    return;
  }
  if (event.key === "-" || event.key === "_") {
    event.preventDefault();
    const rect = el.canvas.getBoundingClientRect();
    zoomAt(
      1 / 1.2,
      rect.left + rect.width / 2,
      rect.top + rect.height / 2,
    );
    return;
  }
  if (event.key === "0") {
    event.preventDefault();
    resetViewport();
    return;
  }
  if (event.key === "?") {
    event.preventDefault();
    openShortcuts();
    return;
  }
  if (event.key === " ") {
    state.spaceDown = true;
    el.canvas.classList.add("space-pan");
  }
}

function onKeyUp(event) {
  if (event.key === " ") {
    state.spaceDown = false;
    el.canvas.classList.remove("space-pan");
  }
}

function cacheElements() {
  Object.assign(el, {
    page: document.getElementById("editor-page"),
    restartBanner: document.getElementById("restart-banner"),
    dirty: document.getElementById("editor-dirty"),
    undo: document.getElementById("btn-undo"),
    redo: document.getElementById("btn-redo"),
    fit: document.getElementById("btn-fit"),
    reload: document.getElementById("btn-reload"),
    zoomIn: document.getElementById("btn-zoom-in"),
    zoomOut: document.getElementById("btn-zoom-out"),
    zoomLabel: document.getElementById("zoom-label"),
    validate: document.getElementById("btn-validate"),
    save: document.getElementById("btn-save"),
    paletteSearch: document.getElementById("palette-search"),
    paletteList: document.getElementById("palette-list"),
    canvas: document.getElementById("canvas"),
    world: document.getElementById("world"),
    rows: document.getElementById("rows"),
    edges: document.getElementById("edges"),
    nodes: document.getElementById("nodes"),
    marquee: document.getElementById("marquee"),
    inspector: document.getElementById("inspector"),
    statusLeft: document.getElementById("status-left"),
    statusMiddle: document.getElementById("status-middle"),
    statusRight: document.getElementById("status-right"),
    modalRoot: document.getElementById("modal-root"),
    contextRoot: document.getElementById("context-root"),
  });
}

function bindEvents() {
  el.save.addEventListener("click", saveGraph);
  el.validate.addEventListener("click", () => validateGraph());
  el.undo.addEventListener("click", undo);
  el.redo.addEventListener("click", redo);
  el.fit.addEventListener("click", () => fitToNodes());
  el.reload.addEventListener("click", reloadGraph);
  el.zoomIn.addEventListener("click", () => {
    const rect = el.canvas.getBoundingClientRect();
    zoomAt(
      1.2,
      rect.left + rect.width / 2,
      rect.top + rect.height / 2,
    );
  });
  el.zoomOut.addEventListener("click", () => {
    const rect = el.canvas.getBoundingClientRect();
    zoomAt(
      1 / 1.2,
      rect.left + rect.width / 2,
      rect.top + rect.height / 2,
    );
  });
  el.paletteSearch.addEventListener("input", () => {
    state.paletteFilter = el.paletteSearch.value;
    renderPalette();
  });
  el.paletteList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-add-type]");
    if (!button) return;
    addPaletteItem(button.dataset.addType, button.dataset.addName);
  });
  el.canvas.addEventListener("pointerdown", onCanvasPointerDown);
  el.canvas.addEventListener("wheel", onWheel, { passive: false });
  el.canvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    if (!event.target.closest("[data-node-id]")) closeContextMenu();
  });
  el.nodes.addEventListener("pointerdown", onNodesPointerDown);
  el.nodes.addEventListener("click", onNodesClick);
  el.nodes.addEventListener("contextmenu", onNodesContext);
  el.edges.addEventListener("click", onEdgeClick);
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);
  window.addEventListener("pointercancel", onPointerUp);
  document.addEventListener("keydown", onKeyDown);
  document.addEventListener("keyup", onKeyUp);
  document.addEventListener(
    "pointerdown",
    (event) => {
      if (!event.target.closest("#context-root")) closeContextMenu();
    },
    true,
  );
  el.inspector.addEventListener("input", handleInspectorInput);
  el.inspector.addEventListener("change", handleInspectorChange);
  el.inspector.addEventListener("click", handleInspectorClick);
  el.inspector.addEventListener("focusin", (event) => {
    if (
      isEditingTarget(event.target) &&
      !state.editBefore
    ) {
      state.editBefore = snapshot();
    }
  });
  el.inspector.addEventListener("focusout", (event) => {
    if (isEditingTarget(event.target)) commitEdit();
  });
  window.addEventListener("beforeunload", (event) => {
    if (!state.dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });
  let lastReload = 0;
  document.addEventListener("visibilitychange", () => {
    if (
      document.hidden ||
      state.dirty ||
      state.busy ||
      Date.now() - lastReload < 1500
    ) {
      return;
    }
    lastReload = Date.now();
    loadGraph({ silent: true });
  });
}

async function loadPublishSources() {
  try {
    const response = await fetch("/api/plugins/publish-sources", {
      cache: "no-store",
    });
    if (!response.ok) return;
    const data = await response.json();
    state.publishSources = Array.isArray(data.sources) ? data.sources : [];
    if (
      selectedNode() &&
      el.inspector &&
      !el.inspector.contains(document.activeElement)
    ) {
      renderInspector();
    }
  } catch (error) {
    state.publishSources = [];
  }
}

function init() {
  cacheElements();
  bindEvents();
  updateButtons();
  loadPublishSources();
  loadGraph({ first: true });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init, { once: true });
} else {
  init();
}





