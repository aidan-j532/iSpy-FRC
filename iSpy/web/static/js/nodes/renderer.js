// -------- node DOM lifecycle. wires/config stay in ctx.graph; this module
// just keeps the #graph-layer divs in sync. uses ctx.wireCamera/ctx.renderPalette. --------
import { api } from './api.js';
import { $, nodeEl, esc, jsStr } from './dom.js';
import { addPipe, nextPos, removeCamera, removePipe } from './graph.js';
import { saveLayout } from './storage.js';
import { makeDraggable } from './selection.js';

export function cameraNodeHtml(ctx, key) {
  const c = ctx.graph.cameras[key] || {};
  const pipe = c.pipeline || 'object_detection';
  const live = ctx.runtime && ctx.runtime.cams && ctx.runtime.cams[key];
  const dot = live ? (live.ok ? 'ok' : (live.stale ? 'stale' : 'down')) : 'down';
  const liveLine = live
    ? `${esc(live.resolution || '?')} &middot; ${live.frame_age_ms != null ? live.frame_age_ms + 'ms' : 'no frame'}`
    : 'not running';
  return `<div class="node camera" data-kind="camera" data-key="${jsStr(key)}">
    <div class="node-head">
      <span class="node-title" title="${esc(key)}">${esc(c.name || key)}</span>
      <button class="node-btn" title="settings" onclick="editNode('${jsStr(key)}')">&#9881;</button>
      <button class="node-btn del" title="delete camera" onclick="deleteCameraNode('${jsStr(key)}')">&times;</button>
    </div>
    <div class="node-type-label">pipeline: ${esc(pipe)}</div>
    <div class="node-body">${esc(c.camera_type || 'opencv')} &middot; ${esc(String(c.source ?? '-'))}</div>
    <div class="node-status"><span class="node-dot ${dot}"></span>${liveLine}</div>
    <span class="port out" data-edge="out" data-node="${jsStr(key)}"></span>
  </div>`;
}

export function pipelineNodeHtml(key, name) {
  return `<div class="node pipeline" data-kind="pipeline" data-key="${jsStr(key)}">
    <div class="node-head">
      <span class="node-title">${esc(name)}</span>
      <button class="node-btn" title="settings" onclick="editNode('${jsStr(key)}')">&#9881;</button>
      <button class="node-btn del" title="remove node" onclick="removePipelineNode('${jsStr(key)}')">&times;</button>
    </div>
    <div class="node-body">vision pipeline</div>
    <div class="node-status"><span class="node-dot" data-role="pipe-dot"></span><span data-role="pipe-state">unwired</span></div>
    <span class="port in" data-edge="in" data-node="${jsStr(key)}"></span>
  </div>`;
}

export function addCameraNode(ctx, key, opts) {
  if (nodeEl(key)) return;
  const pos = ctx.graph.layout[key] || nextPos(ctx.graph);
  ctx.graph.layout[key] = pos;
  const div = document.createElement('div');
  div.id = 'node-' + key;
  div.innerHTML = cameraNodeHtml(ctx, key);
  div.style.left = pos.x + 'px';
  div.style.top = pos.y + 'px';
  $('#graph-layer').appendChild(div);
  makeDraggable(ctx, div, key);
  if (!opts || opts.autowire !== false) ctx.wireCamera(key);
  saveLayout(ctx.graph.layout);
}

export function addPipelineNode(ctx, name, pos) {
  const key = addPipe(ctx.graph, name);
  const div = document.createElement('div');
  div.id = 'node-' + key;
  div.innerHTML = pipelineNodeHtml(key, name);
  const p = pos || nextPos(ctx.graph);
  ctx.graph.layout[key] = p;
  div.style.left = p.x + 'px';
  div.style.top = p.y + 'px';
  $('#graph-layer').appendChild(div);
  makeDraggable(ctx, div, key);
  return key;
}

export async function deleteCameraNode(ctx, key) {
  if (!confirm(`delete camera '${key}'?`)) return;
  try {
    await api.deleteCamera(key);
    const el = nodeEl(key);
    if (el) el.remove();
    removeCamera(ctx.graph, key);
    ctx.selected.delete(key);
    saveLayout(ctx.graph.layout);
    ctx.renderPalette();
    prunePipelineNodes(ctx);
    ctx.renderWires();
    showToast('camera deleted', 'success');
  } catch (e) {
    showToast(e.message || 'request failed', 'error');
  }
}

export function removePipelineNode(ctx, key) {
  const el = nodeEl(key);
  if (el) el.remove();
  const wiredCams = Object.keys(ctx.graph.wires).filter(camKey => ctx.graph.wires[camKey] === key);
  removePipe(ctx.graph, key);
  ctx.selected.delete(key);
  // unwire any camera pointing at this node (config is untouched)
  wiredCams.forEach(camKey => {
    const c = nodeEl(camKey);
    if (c) {
      c.dataset.pipeKey = '';
      const l = c.querySelector('.node-type-label');
      if (l) l.textContent = 'pipeline: not wired';
    }
  });
  ctx.renderWires();
}

export function prunePipelineNodes(ctx) {
  const used = new Set(Object.values(ctx.graph.wires));
  Object.keys(ctx.graph.pipes).forEach(k => {
    if (used.has(k)) return;
    const el = nodeEl(k);
    if (el) el.remove();
    removePipe(ctx.graph, k);
  });
}

// rebuild the canvas from server state: all configured cameras + wires
export function syncCamerasToCanvas(ctx) {
  document.querySelectorAll('#graph-layer .node.camera').forEach(el => {
    if (!(el.dataset.key in ctx.graph.cameras)) { el.remove(); delete ctx.graph.layout[el.dataset.key]; }
  });
  Object.keys(ctx.graph.cameras).forEach(key => { addCameraNode(ctx, key, { autowire: false }); });
  Object.keys(ctx.graph.cameras).forEach(key => { ctx.wireCamera(key); });
  prunePipelineNodes(ctx);
  saveLayout(ctx.graph.layout);
}