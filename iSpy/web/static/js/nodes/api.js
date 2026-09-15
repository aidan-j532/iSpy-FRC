// -------- thin fetch wrappers. endpoints stay the source of truth for config. --------
const j = r => r.json().catch(() => ({}));

async function ok(r) {
  const body = await j(r);
  if (!r.ok) throw new Error(body.error || 'request failed');
  return body;
}

export const api = {
  graph: () => fetch('/api/nodes/graph').then(ok),
  plugins: () => fetch('/api/plugins/available').then(ok),
  models: () => fetch('/api/models').then(j).catch(() => ({ models: [] })),
  status: () => fetch('/api/status').then(ok),

  addCamera(body) {
    return fetch('/api/cameras/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(ok);
  },
  putCamera(key, body) {
    return fetch('/api/cameras/config/' + encodeURIComponent(key), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(ok);
  },
  deleteCamera(key) {
    return fetch('/api/cameras/config/' + encodeURIComponent(key), { method: 'DELETE' }).then(ok);
  },
  selectModel(name) {
    return fetch('/api/models/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_path: 'YoloModels/pytorch/' + name }),
    }).then(ok);
  },
  saveAddon(name, type, settings) {
    return fetch('/api/plugins/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, type, settings }),
    }).then(ok);
  },
  toggleAddon(name, type, enable) {
    return fetch('/api/plugins/toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, type, enable }),
    }).then(ok);
  },
};