// -------- editor entry: builds the shared context and boots everything.
// one place to see cross-module dependencies, everyone else gets ctx. --------
import { api } from './api.js';
import * as canvas from './canvas.js';
import * as contextMenu from './contextMenu.js';
import * as connections from './connections.js';
import { $, nodeEl } from './dom.js';
import * as graph from './graph.js';
import { createHistory } from './history.js';
import * as inspector from './inspector.js';
import * as palette from './palette.js';
import * as renderer from './renderer.js';
import * as selection from './selection.js';
import { loadLayout, loadView, saveLayout } from './storage.js';

function buildCtx() {
  const ctx = {
    graph: graph.createGraph(),
    view: loadView(),
    selected: new Set(),
    history: createHistory(),
    wireTmp: null,
    runtime: null,
    currentEdit: null,
  };
  ctx.graph.layout = loadLayout();
  // cross-module moves got injected here so the modules never import each other in a cycle
  ctx.screenToWorld = (cx, cy) => canvas.screenToWorld(ctx, cx, cy);
  ctx.applyView = () => canvas.applyView(ctx);
  ctx.renderWires = () => connections.renderWires(ctx);
  ctx.paintSelection = () => selection.paintSelection(ctx);
  ctx.pushUndo = () => ctx.history.push(ctx.graph);
  ctx.restore = snap => restoreSnap(ctx, snap);
  ctx.zoomAroundCenter = f => canvas.zoomAroundCenter(ctx, f);
  ctx.addPipelineNode = name => renderer.addPipelineNode(ctx, name);
  ctx.wireCamera = (key, force) => connections.wireCamera(ctx, key, force);
  ctx.addCameraNode = (key, opts) => renderer.addCameraNode(ctx, key, opts);
  ctx.deleteCameraNode = key => renderer.deleteCameraNode(ctx, key);
  ctx.removePipelineNode = key => renderer.removePipelineNode(ctx, key);
  ctx.renderPalette = () => palette.renderPalette(ctx);
  ctx.renderAddons = () => palette.renderAddons(ctx);
  ctx.loadState = force => loadState(ctx, force);
  ctx.editNode = key => inspector.editNode(ctx, key);
  return ctx;
}

async function loadState(ctx, force) {
  try {
    const g = await api.graph();
    graph.loadServerGraph(ctx.graph, g);
    const a = await api.plugins();
    ctx.graph.addonList = (a.available || []).filter(p => p.type !== 'vision_pipeline');
    const m = await api.models();
    ctx.graph.models = m.models || [];
    // pipeline nodes rebuilt from scratch each load (wires are in config)
    document.querySelectorAll('#graph-layer .node.pipeline').forEach(el => el.remove());
    ctx.graph.pipes = {};
    palette.renderPalette(ctx);
    palette.renderAddons(ctx);
    renderer.syncCamerasToCanvas(ctx);
    connections.renderWires(ctx);
  } catch (e) {
    console.error('load failed', e);
    showToast('Failed to load node graph', 'error');
  }
}

function restoreSnap(ctx, snap) {
  document.querySelectorAll('#graph-layer .node').forEach(el => el.remove());
  ctx.graph.layout = JSON.parse(JSON.stringify(snap.layout));
  ctx.graph.pipes = JSON.parse(JSON.stringify(snap.pipes));
  ctx.selected.clear();
  Object.keys(ctx.graph.cameras).forEach(key => {
    if (!ctx.graph.layout[key]) ctx.graph.layout[key] = graph.nextPos(ctx.graph);
    renderer.addCameraNode(ctx, key, { autowire: false });
  });
  Object.keys(ctx.graph.pipes).forEach(key => {
    const pos = ctx.graph.layout[key] || graph.nextPos(ctx.graph);
    ctx.graph.layout[key] = pos;
    const div = document.createElement('div');
    div.id = 'node-' + key;
    div.innerHTML = renderer.pipelineNodeHtml(key, ctx.graph.pipes[key].name);
    div.style.left = pos.x + 'px'; div.style.top = pos.y + 'px';
    $('#graph-layer').appendChild(div);
    selection.makeDraggable(ctx, div, key);
  });
  // re-derive wires from config, same as a fresh load (snapshot only
  // remembers positions + which pipeline nodes existed, not the wires)
  Object.keys(ctx.graph.cameras).forEach(key => connections.wireCamera(ctx, key));
  saveLayout(ctx.graph.layout);
  connections.renderWires(ctx);
}

function resetLayout(ctx) {
  ctx.graph.layout = {};
  saveLayout(ctx.graph.layout);
  Object.keys(ctx.graph.pipes).forEach(k => { const el = nodeEl(k); if (el) el.remove(); });
  ctx.graph.pipes = {};
  document.querySelectorAll('#graph-layer .node.camera').forEach(el => el.remove());
  ctx.renderWires();
  loadState(ctx, true);
}

// -------- runtime status --------
async function refreshRuntime(ctx) {
  try {
    const j = await api.status();
    ctx.runtime = {
      running: !!j.vision_running,
      fps: j.fps,
      vision_ms: j.vision_ms,
      detections: j.detections,
      cams: {},
    };
    (j.cameras || []).forEach(c => {
      // cams carry display name/source, main config name is the key here
      const key = Object.keys(ctx.graph.cameras).find(k => {
        const cam = ctx.graph.cameras[k];
        return c.name === (cam.name || k) || c.name === cam.source || c.name === k;
      });
      if (key) ctx.runtime.cams[key] = c;
    });
  } catch { ctx.runtime = null; }
  paintRuntime(ctx);
}

function paintRuntime(ctx) {
  const badge = $('#runtime-badge');
  if (!ctx.runtime) { badge.classList.remove('running'); $('#runtime-text').textContent = 'vision: no connection'; return; }
  badge.classList.toggle('running', ctx.runtime.running);
  const parts = [ctx.runtime.running ? 'running' : 'stopped'];
  if (ctx.runtime.fps != null && ctx.runtime.fps > 0) parts.push(ctx.runtime.fps + ' fps');
  if (ctx.runtime.vision_ms != null) parts.push(ctx.runtime.vision_ms + ' ms');
  if (ctx.runtime.detections != null) parts.push(ctx.runtime.detections + ' detections');
  $('#runtime-text').textContent = 'vision: ' + parts.join(' \u00b7 ');
  Object.keys(ctx.graph.cameras).forEach(key => {
    const el = nodeEl(key);
    if (!el) return;
    el.innerHTML = renderer.cameraNodeHtml(ctx, key);
  });
  // pipeline nodes: brighten the dot when at least one wired camera is live
  const liveKeys = new Set(Object.keys(ctx.runtime.cams).filter(k => ctx.runtime.cams[k].ok));
  document.querySelectorAll('.node.pipeline').forEach(el => {
    const dot = el.querySelector('[data-role="pipe-dot"]');
    const txt = el.querySelector('[data-role="pipe-state"]');
    if (!dot) return;
    const wiredCams = document.querySelectorAll(`.node.camera[data-pipe-key="${CSS.escape(el.dataset.key)}"]`);
    const anyLive = [...wiredCams].some(c => liveKeys.has(c.dataset.key));
    dot.classList.toggle('ok', anyLive);
    txt.textContent = anyLive ? 'processing' : (wiredCams.length ? 'camera down' : 'unwired');
  });
}

export function boot() {
  const ctx = buildCtx();

  // globals the inline template onclick= handlers call
  window.resetLayout = () => resetLayout(ctx);
  window.applyTemplate = name => palette.applyTemplate(ctx, name);
  window.loadState = force => loadState(ctx, force);
  window.fitToScreen = () => canvas.fitToScreen(ctx);
  window.zoomAroundCenter = f => canvas.zoomAroundCenter(ctx, f);
  window.setZoom = z => canvas.setZoom(ctx, z);
  window.addCameraNode = key => renderer.addCameraNode(ctx, key);
  window.addNewCamera = () => palette.addNewCamera(ctx);
  window.addPipelineNodeFromPicker = () => palette.openPipelinePicker(ctx);
  window.pickPipelineType = name => palette.pickPipelineType(ctx, name);
  window.editNode = key => inspector.editNode(ctx, key);
  window.deleteCameraNode = key => renderer.deleteCameraNode(ctx, key);
  window.removePipelineNode = key => renderer.removePipelineNode(ctx, key);
  window.duplicateNode = key => selection.duplicateNode(ctx, key);
  window.toggleAddon = (name, type, enable) => palette.toggleAddon(ctx, name, type, enable);
  window.editAddon = (type, name) => inspector.editAddon(ctx, type, name);
  window.closePickType = () => palette.closePipelinePicker();

  canvas.applyView(ctx);
  canvas.initPan(ctx);
  canvas.initWheel(ctx);
  connections.initPortDrag(ctx);
  selection.initKeyboard(ctx);
  contextMenu.init(ctx);
  palette.init(ctx);
  inspector.init(ctx);
  window.addEventListener('resize', () => ctx.renderWires());

  loadState(ctx, true);
  refreshRuntime(ctx);
  setInterval(() => refreshRuntime(ctx), 2000);
}