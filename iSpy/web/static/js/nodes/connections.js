// -------- wiring: wires svg, port drag, camera->pipeline assignment.
// uses ctx.screenToWorld / ctx.addPipelineNode / ctx.renderPalette / ctx.renderWires. --------
import { api } from './api.js';
import { $, nodeEl, jsStr } from './dom.js';
import { cameraPipelineName, connect, disconnectCamera, findFreePipe } from './graph.js';
import { wirePath, nodeEdgePos as worldEdge } from './graphUtils.js';
import { prunePipelineNodes } from './renderer.js';

export function renderWires(ctx) {
  const svg = $('#wire-svg');
  const wires = [];
  Object.keys(ctx.graph.wires).forEach(camKey => {
    const pipeKey = ctx.graph.wires[camKey];
    const node = nodeEl(camKey), pipeNode = nodeEl(pipeKey);
    if (!node || !pipeNode) return;
    const out = worldEdge(ctx.graph.layout, camKey, 'out', node.offsetWidth, node.offsetHeight);
    const inn = worldEdge(ctx.graph.layout, pipeKey, 'in', pipeNode.offsetWidth, pipeNode.offsetHeight);
    wires.push(`<path d="${wirePath(out.x, out.y, inn.x, inn.y)}" class="wire" data-from="${jsStr(camKey)}" data-to="${jsStr(pipeKey)}"></path>`);
  });
  svg.innerHTML = wires.join('');
  if (ctx.wireTmp) {
    svg.innerHTML += `<path d="${wirePath(ctx.wireTmp.x1, ctx.wireTmp.y1, ctx.wireTmp.x2, ctx.wireTmp.y2)}" class="wire candidate"/>`;
  }
}

export function connectWire(ctx, camKey, pipeKey, opts) {
  const camEl = nodeEl(camKey);
  if (!camEl) return;
  camEl.dataset.pipeKey = pipeKey;
  connect(ctx.graph, camKey, pipeKey);
  if (!opts || !opts.silent) { renderWires(ctx); }
}

export function removeWiresForCamera(ctx, camKey) {
  const el = nodeEl(camKey);
  if (el) el.dataset.pipeKey = '';
  disconnectCamera(ctx.graph, camKey);
}

// wire a camera: reuse a free pipeline node of that type, else make one
export function wireCamera(ctx, camKey, forcePipe) {
  const target = forcePipe || cameraPipelineName(ctx.graph, camKey);
  let pipeKey = findFreePipe(ctx.graph, target);
  if (!pipeKey) pipeKey = ctx.addPipelineNode(target);
  connectWire(ctx, camKey, pipeKey, { silent: true });
  applyCameraPipeline(ctx, camKey, target, pipeKey, { silent: true });
  if (forcePipe) saveCameraPipeline(ctx, camKey, target);
}

export async function wireCameraToPipeline(ctx, camKey, pipeKey, noSave) {
  const pipe = ctx.graph.pipes[pipeKey];
  if (!pipe) return;
  wireCamera(ctx, camKey);
  const apply = await applyCameraPipeline(ctx, camKey, pipe.name, pipeKey, { silent: true });
  if (apply === false) return;
  if (!noSave) await saveCameraPipeline(ctx, camKey, pipe.name);
  prunePipelineNodes(ctx);
  renderWires(ctx);
  ctx.renderPalette();
}

export function applyCameraPipeline(ctx, camKey, pipeName, pipeKey, opts) {
  connectWire(ctx, camKey, pipeKey, opts);
  const cam = ctx.graph.cameras[camKey];
  if (cam) cam.pipeline = pipeName;
  const el = nodeEl(camKey);
  const labelEl = el && el.querySelector('.node-type-label');
  if (labelEl) labelEl.textContent = 'pipeline: ' + pipeName;
  return true;
}

export async function saveCameraPipeline(ctx, camKey, pipeName) {
  try {
    await api.putCamera(camKey, { pipeline: { name: pipeName } });
    showToast('pipeline set to ' + pipeName, 'success');
    return true;
  } catch (e) {
    showToast(e.message || 'pipeline save failed', 'error');
    return false;
  }
}

// port dragging: start a prospective wire from a port, drop it on another node
export function initPortDrag(ctx) {
  $('#graph-layer').addEventListener('mousedown', ev => {
    const port = ev.target.closest('.port');
    if (!port) return;
    ev.preventDefault();
    const node = port.closest('.node');
    const start = worldEdge(ctx.graph.layout, node.dataset.key, port.dataset.edge === 'in' ? 'in' : 'out', node.offsetWidth, node.offsetHeight);
    ctx.wireTmp = { x1: start.x, y1: start.y, x2: start.x, y2: start.y };
    renderWires(ctx);
    const move = ev2 => {
      const p = ctx.screenToWorld(ev2.clientX, ev2.clientY);
      ctx.wireTmp.x2 = p.x; ctx.wireTmp.y2 = p.y;
      renderWires(ctx);
    };
    const up = ev2 => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
      ctx.wireTmp = null;
      renderWires(ctx);
      const endEl = document.elementFromPoint(ev2.clientX, ev2.clientY);
      const endNode = endEl && endEl.closest('.node');
      if (!endNode || endNode === node) return;
      const from = node.dataset.kind;
      const to = endNode.dataset.kind;
      // camera(out) -> pipeline(in) is basically the only legal combo here
      if (from === 'camera' && to === 'pipeline') {
        wireCameraToPipeline(ctx, node.dataset.key, endNode.dataset.key, ev2.shiftKey);
      } else if (from === 'pipeline' && to === 'camera') {
        wireCameraToPipeline(ctx, endNode.dataset.key, node.dataset.key, ev2.shiftKey);
      }
    };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
  });
}