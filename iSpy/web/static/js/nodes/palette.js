// -------- left palette + right add-ons panel. wires/templates live here. --------
import { api } from './api.js';
import { $, nodeEl, esc, jsStr } from './dom.js';
import { TEMPLATES } from './nodeDefinitions.js';
import { wireCameraToPipeline } from './connections.js';

export function renderPalette(ctx) {
  const pal = $('#palette');
  let html = '<div class="palette-card"><div class="pal-title">Cameras</div>';
  Object.values(ctx.graph.cameras).forEach(c => {
    html += `<button class="pal-item" onclick="addCameraNode('${jsStr(c.key)}')"><span>&#9679;</span> ${esc(c.name)}</button>`;
  });
  html += '<button class="pal-item" onclick="addNewCamera()"><span>+</span> add camera</button></div>';
  html += '<div class="palette-card"><div class="pal-title">Vision Pipelines</div>';
  html += '<button class="pal-item" onclick="addPipelineNodeFromPicker()"><span>+</span> add pipeline</button>';
  html += '</div><div class="palette-card"><div class="pal-title">Templates</div>';
  TEMPLATES.forEach(t => {
    html += `<button class="pal-item" onclick="applyTemplate('${jsStr(t.pipe)}')">${esc(t.label)}</button>`;
  });
  html += '</div>';
  pal.innerHTML = html;
}

// stock templates: wire every configured camera onto one pipeline so people
// get a pull-the-cord-once starting point. config stays the source of truth.
export function applyTemplate(ctx, pipeName) {
  const keys = Object.keys(ctx.graph.cameras);
  if (!keys.length) { showToast('add a camera first', 'error'); return; }
  keys.forEach(key => {
    if (!nodeEl(key)) ctx.addCameraNode(key, { autowire: false });
    ctx.wireCamera(key, pipeName);
  });
  ctx.renderWires();
  showToast('template applied to ' + keys.length + ' camera(s)', 'success');
}

export async function addNewCamera(ctx) {
  const name = prompt('camera name:');
  if (!name) return;
  const source = prompt('video source (index or path):');
  if (source == null) return;
  try {
    await api.addCamera({ name, source, camera_type: 'opencv' });
    showToast('camera added', 'success');
    ctx.loadState(true);
  } catch (e) {
    showToast(e.message || 'request failed', 'error');
  }
}

// -------- pipeline type picker (modal) --------
export function openPipelinePicker(ctx) {
  $('#picktype-body').innerHTML = Object.values(ctx.graph.pipelines).map(p =>
    `<button class="pal-item" style="margin:4px 0;" onclick="pickPipelineType('${jsStr(p.name)}')">${esc(p.name)}</button>`).join('');
  $('#picktype-modal').style.display = 'flex';
}

export function closePipelinePicker() { $('#picktype-modal').style.display = 'none'; }

export function pickPipelineType(ctx, name) {
  closePipelinePicker();
  ctx.pushUndo();
  const key = ctx.addPipelineNode(name);
  // wire to the currently selected camera if any, else first camera
  const sel = document.querySelector('.node.camera.selected');
  const camKey = (sel && sel.dataset.key) || Object.keys(ctx.graph.cameras)[0];
  if (camKey) {
    wireCameraToPipeline(ctx, camKey, key);
  } else {
    showToast('no camera to wire, add one from the palette', 'error');
  }
}

// -------- global add-ons panel --------
export function renderAddons(ctx) {
  const list = $('#addons-list');
  if (!ctx.graph.addonList.length) { list.innerHTML = '<span class="text-dim text-sm">no add-ons available</span>'; return; }
  list.innerHTML = ctx.graph.addonList.map(p => {
    const hasSettings = p.enabled && Object.keys(p.config_schema || {}).length > 0;
    return `<div class="addon-row-mini">
      <label class="switch" title="${p.enabled ? 'disable' : 'enable'}">
        <input type="checkbox" ${p.enabled ? 'checked' : ''} onchange="toggleAddon('${jsStr(p.name)}','${p.type}',this.checked)">
        <span class="slider"></span>
      </label>
      <span class="name">${esc(p.name)}</span>
      <span class="text-muted" style="font-size:0.65rem;">${esc(p.type.replace('_',' '))}</span>
      ${hasSettings ? `<button class="btn-sm" onclick="editAddon('${p.type}','${jsStr(p.name)}')">gear</button>` : ''}
    </div>`;
  }).join('');
}

export async function toggleAddon(ctx, name, type, enable) {
  try {
    await api.toggleAddon(name, type, enable);
    showToast(`${name} ${enable ? 'enabled' : 'disabled'}`, 'success');
  } catch (e) {
    showToast((e && e.message) || 'toggle failed', 'error');
  }
  ctx.loadState(true);
}

export function init(ctx) {
  $('#picktype-modal').addEventListener('click', e => { if (e.target === e.currentTarget) closePipelinePicker(); });
}