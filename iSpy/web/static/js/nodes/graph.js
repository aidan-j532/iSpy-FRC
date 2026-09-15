// -------- data model for the node editor. no DOM here so it unit-tests in node. --------
import { DEFAULT_PIPELINE, pipeNodeId } from './nodeDefinitions.js';

export function createGraph() {
  return {
    cameras: {},       // key -> camera payload from /api/nodes/graph
    pipelines: {},     // name -> pipeline payload (config_schema ...)
    cameraSchemas: {}, // camera_type -> schema
    addonList: [],
    models: [],
    layout: {},        // key -> {x,y}
    pipes: {},         // pipe:<key> -> {name}; pipeline nodes are virtual, a camera wire is what persists
    wires: {},         // camKey -> pipeKey
  };
}

export function loadServerGraph(g, payload) {
  g.cameras = {};
  (payload.cameras || []).forEach(c => g.cameras[c.key] = c);
  g.pipelines = {};
  (payload.pipelines || []).forEach(p => g.pipelines[p.name] = p);
  g.cameraSchemas = payload.camera_schemas || {};
}

export function addCamera(g, cam) { g.cameras[cam.key] = cam; }

export function removeCamera(g, key) {
  delete g.cameras[key];
  delete g.layout[key];
  delete g.wires[key];
}

// pipeline node ids are just pipe:name:count, nothing smarter needed
export function addPipe(g, name) {
  const key = pipeNodeId(name, Object.keys(g.pipes).length + 1);
  g.pipes[key] = { name };
  return key;
}

export function removePipe(g, key) {
  delete g.pipes[key];
  delete g.layout[key];
  // unwire any camera pointing at this node (config is untouched)
  Object.keys(g.wires).forEach(camKey => {
    if (g.wires[camKey] === key) delete g.wires[camKey];
  });
}

export function prunePipes(g) {
  const used = new Set(Object.values(g.wires));
  Object.keys(g.pipes).forEach(k => {
    if (!used.has(k)) { delete g.pipes[k]; delete g.layout[k]; }
  });
}

export function connect(g, camKey, pipeKey) { g.wires[camKey] = pipeKey; }
export function disconnectCamera(g, camKey) { delete g.wires[camKey]; }

export function cameraWiredTo(g, pipeKey) { return Object.values(g.wires).includes(pipeKey); }

export function findFreePipe(g, name) {
  for (const k in g.pipes) {
    if (g.pipes[k].name === name && !cameraWiredTo(g, k)) return k;
  }
  return null;
}

// first (and only) camera wired onto a pipeline node
export function ownerCameraOf(g, pipeKey) {
  for (const c in g.wires) if (g.wires[c] === pipeKey) return c;
  return null;
}

export function cameraPipelineName(g, camKey) {
  const cam = g.cameras[camKey];
  return (cam && cam.pipeline) || DEFAULT_PIPELINE;
}

export function nextPos(g) {
  const used = Object.keys(g.layout).filter(k => g.layout[k]).length;
  return { x: 60 + (used % 5) * 80, y: 50 + (used % 4) * 80 };
}