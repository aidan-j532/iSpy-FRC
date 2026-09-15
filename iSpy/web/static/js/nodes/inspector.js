// -------- settings popup (schema rendered) + add-on settings --------
import { api } from './api.js';
import { $, esc } from './dom.js';
import { ownerCameraOf } from './graph.js';
import { nodeEl } from './dom.js';

export function editNode(ctx, key) {
  const el = nodeEl(key);
  if (!el) return;
  if (el.dataset.kind === 'camera') editCamera(ctx, key);
  else if (el.dataset.kind === 'pipeline') editPipeline(ctx, key);
}

function renderPipeOptions(ctx, cam) {
  return Object.values(ctx.graph.pipelines)
    .map(p => `<option value="${esc(p.name)}" ${p.name === cam.pipeline ? 'selected' : ''}>${esc(p.name)}</option>`)
    .join('');
}

export function editCamera(ctx, key) {
  const cam = ctx.graph.cameras[key] || {};
  const schema = ctx.graph.cameraSchemas[cam.camera_type || 'opencv'] || {};
  const entry = (cam && cam.entry) || {};
  const vals = {};
  Object.keys(schema).forEach(fk => { vals[fk] = (fk in entry) ? entry[fk] : cam[fk]; });
  ctx.currentEdit = {
    kind: 'camera',
    key, schema, values: vals, cam,
    pre: `<div class="form-row">
        <span class="form-label">Name</span>
        <input class="form-input" data-key="__name" data-type="text" value="${esc(cam.name || key)}">
      </div>
      <div class="form-row">
        <span class="form-label">Pipeline</span>
        <select class="form-input" data-key="__pipeline" data-type="pipeline">${renderPipeOptions(ctx, cam)}</select>
      </div>`,
    save: async body => {
      const payload = {};
      if (body.__pipeline) payload.pipeline = { name: body.__pipeline };
      const name = String(body.__name || '').trim();
      if (name && name !== (cam.name || key)) payload.name = name;
      delete body.__name; delete body.__pipeline;
      Object.keys(body).forEach(k => { payload[k] = body[k]; });
      await api.putCamera(key, payload); // also persists pipeline settings fields routed by the backend
      ctx.loadState(true);
    },
  };
  openSettingsModal(ctx, `camera: ${cam.name || key}`, renderSchemaFields(ctx, schema, vals));
}

export function editPipeline(ctx, key) {
  const pipe = ctx.graph.pipes[key];
  if (!pipe) { showToast('pipeline node gone', 'error'); return; }
  const payload = ctx.graph.pipelines[pipe.name];
  const schema = (payload && payload.config_schema) || {};
  // pipe settings live on the wired camera (backend stores them per camera)
  const ownerCam = ownerCameraOf(ctx.graph, key);
  if (!ownerCam) { showToast('wire this pipeline to a camera first', 'error'); return; }
  const camState = ctx.graph.cameras[ownerCam] || {};
  const vals = {};
  Object.keys(schema).forEach(fk => {
    vals[fk] = (camState.pipeline_settings || {})[fk];
  });
  ctx.currentEdit = {
    kind: 'pipeline', key, schema, values: vals, ownerCam, pipe,
    pre: `<div class="form-hint">settings on camera '${esc(camState.name || ownerCam)}'</div>`,
    save: async body => {
      // a 'model' pick goes through its own endpoint (writes every model-backed cam)
      const modelName = body.vision_model;
      delete body.vision_model;
      if (modelName) await api.selectModel(modelName);
      await api.putCamera(ownerCam, { pipeline: { name: pipe.name, settings: body } });
      const cs = ctx.graph.cameras[ownerCam];
      if (cs) cs.pipeline_settings = Object.assign(cs.pipeline_settings || {}, body);
      ctx.loadState(true);
    },
  };
  openSettingsModal(ctx, `pipeline: ${pipe.name}`, renderSchemaFields(ctx, schema, vals));
}

export function editAddon(ctx, type, name) {
  const addon = ctx.graph.addonList.find(a => a.type === type && a.name === name);
  if (!addon) return;
  const schema = addon.config_schema || {};
  const vals = Object.assign({}, addon.settings || {});
  ctx.currentEdit = {
    kind: 'addon', key: name, schema, values: vals,
    pre: '',
    save: async body => {
      await api.saveAddon(name, type, body);
    },
  };
  openSettingsModal(ctx, `add-on: ${name}`, renderSchemaFields(ctx, schema, vals));
}

function currentModelName(schema, vals) {
  if (!schema.vision_model) return '';
  const vm = vals.vision_model;
  if (vm && typeof vm === 'object' && vm.file_path) return String(vm.file_path).split('/').pop();
  if (vm && typeof vm === 'string') return String(vm).split('/').pop();
  return '';
}

export function renderSchemaFields(ctx, schema, vals) {
  const keys = Object.keys(schema);
  if (!keys.length) return '<p class="text-dim text-sm">no settings</p>';
  return keys.map(key => {
    const def = schema[key] || {};
    const label = esc(def.label || key);
    const hint = def.help ? `<small class="text-dim">${esc(def.help)}</small>` : '';
    const value = (key in vals) ? vals[key] : def.default;
    if (def.type === 'toggle') {
      return `<label class="form-row settings-toggle-row">
        <input type="checkbox" class="toggle-input" data-key="${key}" data-type="toggle" ${value ? 'checked' : ''}>
        <span><strong>${label}</strong> ${hint}</span>
      </label>`;
    }
    if (def.type === 'select') {
      const opts = (def.options || []).map(o =>
        `<option value="${esc(o)}" ${String(o) === String(value) ? 'selected' : ''}>${esc(o)}</option>`).join('');
      return `<div class="form-row">
        <span class="form-label">${label}</span>
        <select class="form-input" data-key="${key}" data-type="select">${opts}</select>${hint}</div>`;
    }
    if (def.type === 'model') {
      const current = currentModelName(schema, vals);
      const opts = ctx.graph.models.map(m =>
        `<option value="${esc(m.name)}" ${m.name === current ? 'selected' : ''}>${esc(m.name)}${m.active ? ' *' : ''}</option>`).join('');
      return `<div class="form-row">
        <span class="form-label">${label}</span>
        <select class="form-input" data-key="${key}" data-type="model">
          <option value="">keep current</option>${opts}
        </select>${hint}</div>`;
    }
    if (def.type === 'list') {
      const text = JSON.stringify(Array.isArray(value) ? value : [], null, 1) || '[]';
      return `<div class="form-row">
        <span class="form-label">${label}</span>
        <textarea class="form-input" rows="3" data-key="${key}" data-type="list" style="font-family:monospace;font-size:0.72rem;">${esc(text)}</textarea>${hint}
        <small class="text-dim">one JSON array of objects</small>
      </div>`;
    }
    const inputType = def.type === 'number' ? 'number' : 'text';
    return `<div class="form-row">
      <span class="form-label">${label}</span>
      <input class="form-input" type="${inputType}" data-key="${key}" data-type="${def.type || 'text'}" value="${esc(value == null ? '' : String(value))}" step="any">${hint}
    </div>`;
  }).join('');
}

function openSettingsModal(ctx, title, fieldsHtml) {
  $('#settings-title').textContent = title;
  $('#settings-body').innerHTML = (ctx.currentEdit.pre || '') + fieldsHtml;
  $('#settings-modal').style.display = 'flex';
}

export function closeSettings(ctx) {
  $('#settings-modal').style.display = 'none';
  ctx.currentEdit = null;
}

async function saveSettingsPopup(ctx) {
  if (!ctx.currentEdit) { closeSettings(ctx); return; }
  const body = {};
  let invalid = false;
  $('#settings-body').querySelectorAll('[data-key]').forEach(el => {
    const k = el.dataset.key;
    const t = el.dataset.type;
    if (t === 'toggle') body[k] = el.checked;
    else if (t === 'pipeline') body[k] = el.value;
    else if (t === 'number') {
      const v = el.value.trim();
      if (v === '') body[k] = '';
      else { const n = Number(v); if (Number.isFinite(n)) body[k] = n; else { invalid = true; showToast(`'${k}' must be a number`, 'error'); } }
    } else if (t === 'source') {
      // device indices come back as numbers; keep paths/urls as strings
      const v = el.value.trim();
      body[k] = (v !== '' && !isNaN(Number(v))) ? Number(v) : v;
    } else if (t === 'model') {
      body[k] = el.value;
    } else if (t === 'list') {
      try { body[k] = JSON.parse(el.value); }
      catch { invalid = true; showToast(`'${k}' is not valid JSON`, 'error'); }
    } else body[k] = el.value;
  });
  if (invalid) return;
  try {
    await ctx.currentEdit.save(body);
    showToast('saved - restart vision to apply', 'success');
    Object.assign(ctx.currentEdit.values, body);
  } catch (e) {
    showToast((e && e.message) || 'save failed', 'error');
    return;
  }
  closeSettings(ctx);
}

export function init(ctx) {
  $('#settings-modal').addEventListener('click', e => { if (e.target === e.currentTarget) closeSettings(ctx); });
  window.saveSettingsPopup = () => saveSettingsPopup(ctx);
  window.closeSettings = () => closeSettings(ctx);
}