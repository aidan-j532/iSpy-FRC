import test from "node:test";
import assert from "node:assert/strict";

import {
  addGate,
  chainReady,
  completion,
  connectionError,
  deriveEdges,
  edgePath,
  edgeState,
  executionOrder,
  fitViewport,
  gateActive,
  nextOrder,
  nodePayload,
  removalPlan,
  ringPath,
  settingIsRequired,
  snap,
  uniqueCameraName,
} from "../iSpy/web/static/js/editor_core.js";

const camera = (settings = {}) => ({
  id: "camera:cam",
  type: "camera",
  label: "cam",
  camera: "cam",
  x: 80,
  y: 40,
  schema: { source: { type: "source", label: "Source" } },
  settings: { source: "0", ...settings },
  completion: { complete: true },
});
const processor = (name, order, settings = {}) => ({
  id: `frame_processor:cam:${name}`,
  type: "frame_processor",
  label: name,
  camera: "cam",
  plugin: name,
  order,
  schema: {},
  settings,
  completion: { complete: true },
});
const pipeline = (name = "yolo", complete = true) => ({
  id: "pipeline:cam",
  type: "pipeline",
  label: "cam pipeline",
  camera: "cam",
  plugin: name,
  x: 700,
  y: 400,
  schema: {},
  settings: {},
  completion: { complete },
});
const tracker = (name, order, supported = []) => ({
  id: `tracker:${name}`,
  type: "tracker",
  label: name,
  plugin: name,
  order,
  supported_pipelines: supported,
  schema: {},
  settings: {},
  completion: { complete: true },
});
const utility = (name, order, supported = []) => ({
  id: `utility:${name}`,
  type: "utility",
  label: name,
  plugin: name,
  order,
  supported_pipelines: supported,
  schema: {},
  settings: {},
  completion: { complete: true },
});

test("required fields follow the shared backend rule", () => {
  const schema = {
    source: { type: "source" },
    model: { type: "model" },
    browse: { type: "browse" },
    optional_default: { type: "text", default: "x" },
    optional_nullable: { type: "text", nullable: true },
    explicitly_optional: { type: "text", required: false },
    explicitly_required: { type: "text", default: "x", required: true },
  };
  assert.equal(settingIsRequired(schema, {}, "source"), true);
  assert.equal(settingIsRequired(schema, {}, "model"), true);
  assert.equal(settingIsRequired(schema, {}, "browse"), true);
  assert.equal(settingIsRequired(schema, {}, "optional_default"), false);
  assert.equal(settingIsRequired(schema, {}, "optional_nullable"), false);
  assert.equal(settingIsRequired(schema, {}, "explicitly_optional"), false);
  assert.equal(settingIsRequired(schema, {}, "explicitly_required"), true);

  assert.equal(completion({}, {}).complete, true);
  assert.equal(completion({ a: { type: "text" } }, {}).complete, false);
  assert.equal(completion({ a: { type: "text" } }, { a: "ok" }).complete, true);
  assert.equal(
    completion({ a: { type: "model" } }, { a: { source_pt: "" } }).complete,
    false,
  );
  assert.equal(
    completion({ a: { type: "number" } }, { a: "" }).complete,
    false,
  );
  assert.equal(
    completion(
      { mode: { type: "select" }, regions: { type: "list" } },
      { mode: "", regions: [] },
    ).complete,
    false,
  );
  assert.equal(
    completion({ a: { type: "text", required: null } }, {}).complete,
    false,
  );
});

test("gated fields are only required when their gate is enabled", () => {
  const schema = {
    use_gpu: { type: "toggle", default: false },
    device: { type: "text", gated_by: "use_gpu" },
  };
  assert.equal(gateActive(schema, {}, schema.device), false);
  assert.equal(settingIsRequired(schema, {}, "device"), false);
  assert.equal(settingIsRequired(schema, { use_gpu: true }, "device"), true);
  assert.equal(completion(schema, { use_gpu: false }).complete, true);
  assert.equal(completion(schema, { use_gpu: true }).complete, false);
});

test("connection validation is type, camera, support, and cycle aware", () => {
  const cam = camera();
  const fp = processor("depth", 0);
  const pipe = pipeline();
  const track = tracker("bytetrack", 0, ["yolo"]);
  const util = utility("annotate", 0, ["yolo"]);
  const other = { ...camera(), id: "camera:other", camera: "other", label: "other" };
  const otherPipe = {
    ...pipeline("other-pipeline"),
    id: "pipeline:other",
    camera: "other",
  };
  const nodes = [cam, fp, pipe, track, util, other, otherPipe];

  assert.equal(connectionError(cam, fp, nodes, []), "");
  assert.equal(connectionError(fp, pipe, nodes, []), "");
  const limited = {
    ...fp,
    id: "frame_processor:cam:limited",
    plugin: "limited",
    supported_pipelines: ["other-pipeline"],
  };
  assert.match(
    connectionError(limited, pipe, [...nodes, limited], []),
    /does not support pipeline/,
  );
  assert.equal(connectionError(cam, track, nodes, []), "Can't connect camera 'cam' to tracker 'bytetrack'.");
  assert.equal(
    connectionError(cam, otherPipe, nodes, []),
    "A frame-processor chain cannot cross cameras.",
  );
  assert.match(
    connectionError(otherPipe, track, nodes, []),
    /does not support pipeline/,
  );
  assert.match(
    connectionError(cam, fp, nodes, [{ from: cam.id, to: fp.id }]),
    /already connected/,
  );
  const firstTracker = tracker("t0", 0);
  const secondTracker = tracker("t1", 1);
  assert.match(
    connectionError(secondTracker, firstTracker, nodes, [
      { from: firstTracker.id, to: secondTracker.id },
    ]),
    /cycle/,
  );
});

test("derived edges preserve camera and global execution order", () => {
  const nodes = [
    camera(),
    processor("first", 0),
    processor("second", 1),
    pipeline(),
    tracker("t1", 1, ["yolo"]),
    tracker("t0", 0, ["yolo"]),
    utility("u0", 0, ["yolo"]),
    utility("u1", 1, ["yolo"]),
  ];
  const edges = deriveEdges(nodes);
  const ids = edges.map((edge) => edge.id);
  assert.deepEqual(ids.slice(0, 4), [
    "e:camera:cam->frame_processor:cam:first",
    "e:frame_processor:cam:first->frame_processor:cam:second",
    "e:frame_processor:cam:second->pipeline:cam",
    "e:pipeline:cam->tracker:t0",
  ]);
  assert.ok(ids.includes("e:tracker:t0->tracker:t1"));
  assert.ok(ids.includes("e:tracker:t1->utility:u0"));
  assert.ok(ids.includes("e:tracker:t1->utility:u1"));
  assert.ok(!ids.includes("e:tracker:t1->tracker:t0"));
});

test("execution order uses horizontal position and stays stable on ties", () => {
  const left = { ...tracker("left", 2), x: 100 };
  const middle = { ...tracker("middle", 1), x: 300 };
  const right = { ...tracker("right", 0), x: 300 };
  const ordered = executionOrder([right, left, middle]);
  assert.deepEqual(
    ordered.map((node) => node.plugin),
    ["left", "right", "middle"],
  );

  // Equal x positions fall back to the previous order, then id.
  const tie = { ...tracker("tie", 1), x: 300 };
  const tied = executionOrder([right, tie, middle]);
  assert.deepEqual(
    tied.map((node) => node.plugin),
    ["right", "middle", "tie"],
  );
});

test("adding nodes is gated by completed predecessor rows", () => {
  const cam = camera();
  const fp = processor("first", 0);
  const pipe = pipeline();
  const nodes = [cam, fp, pipe];

  assert.equal(chainReady([]), false);
  assert.equal(chainReady([cam]), false);
  assert.equal(chainReady(nodes), true);
  assert.equal(
    chainReady([cam, fp, { ...pipe, completion: { complete: false } }]),
    false,
  );
  assert.equal(
    chainReady([cam, fp, pipe, { ...pipe, id: "pipeline:cam:extra" }]),
    false,
  );

  assert.deepEqual(addGate([], "camera"), { allowed: true, reason: "" });
  assert.match(
    addGate([], "pipeline", {}).reason,
    /Select a camera/,
  );
  const incompleteCamera = { ...cam, completion: { complete: false } };
  assert.match(
    addGate([incompleteCamera], "frame_processor", {
      camera: incompleteCamera,
    }).reason,
    /Complete the camera/,
  );
  assert.match(
    addGate(nodes, "pipeline", { camera: cam }).reason,
    /exactly one pipeline/,
  );
  assert.match(
    addGate(
      [cam, { ...fp, completion: { complete: false } }],
      "pipeline",
      { camera: cam },
    ).reason,
    /Complete every frame processor/,
  );
  assert.match(
    addGate(nodes, "frame_processor", {
      camera: cam,
      plugin: "first",
    }).reason,
    /already assigned/,
  );
  assert.equal(
    addGate(nodes, "tracker", { plugin: "bytetrack" }).allowed,
    true,
  );
  assert.match(
    addGate([cam, fp], "utility", { plugin: "annotate" }).reason,
    /Complete every camera chain/,
  );
  const track = tracker("bytetrack", 0, ["yolo"]);
  assert.match(
    addGate([...nodes, track], "tracker", { plugin: "bytetrack" }).reason,
    /already in the graph/,
  );
});

test("removal plans cascade camera chains and protect the last camera", () => {
  const cam = camera();
  const fp = processor("first", 0);
  const pipe = pipeline();
  const otherCamera = {
    ...camera(),
    id: "camera:other",
    label: "other",
    camera: "other",
  };
  const otherPipe = {
    ...pipeline(),
    id: "pipeline:other",
    camera: "other",
  };
  const nodes = [cam, fp, pipe, otherCamera, otherPipe];
  const cameraPlan = removalPlan(nodes, [cam.id]);
  assert.deepEqual(cameraPlan.removeIds, [cam.id, fp.id, pipe.id]);
  assert.deepEqual(cameraPlan.remainingCameras, ["other"]);
  assert.equal(cameraPlan.blocked, false);
  assert.deepEqual(cameraPlan.orphanedCameras, []);

  const pipelinePlan = removalPlan(nodes, [pipe.id]);
  assert.deepEqual(pipelinePlan.removeIds, [pipe.id]);
  assert.deepEqual(pipelinePlan.orphanedCameras, ["cam"]);
  assert.equal(pipelinePlan.blocked, false);

  const allCameras = removalPlan(nodes, [cam.id, otherCamera.id]);
  assert.equal(allCameras.blocked, true);
  assert.match(allCameras.reason, /at least one camera/);

  const onlyCamera = removalPlan([cam, pipe], [cam.id]);
  assert.equal(onlyCamera.blocked, true);
  assert.deepEqual(removalPlan(nodes, ["missing"]).selectedIds, []);
});

test("edge states animate incomplete, half, complete, and invalid", () => {
  const source = { ...camera(), completion: { complete: false } };
  const target = { ...pipeline(), completion: { complete: false } };
  assert.equal(edgeState(source, target), "dashed");
  assert.equal(
    edgeState({ ...source, completion: { complete: true } }, target),
    "half",
  );
  assert.equal(
    edgeState(
      { ...source, completion: { complete: true } },
      { ...target, completion: { complete: true } },
    ),
    "complete",
  );
  assert.equal(
    edgeState(
      { ...source, completion: { complete: true } },
      { ...target, completion: { complete: true } },
      { warning: "unsupported" },
    ),
    "invalid",
  );
});

test("layout, geometry, naming, and save metadata stay deterministic", () => {
  assert.equal(snap(33, 20), 40);
  assert.equal(snap(-13, 20), -20);
  assert.equal(uniqueCameraName([camera()], "cam"), "cam_2");
  assert.equal(nextOrder([tracker("a", 3)], (node) => node.type === "tracker"), 4);
  assert.notEqual(ringPath("camera", 206, 128), ringPath("pipeline", 214, 116));
  assert.match(edgePath(camera(), pipeline()), /^M /);

  const viewport = fitViewport([camera(), pipeline()], 1200, 800);
  assert.ok(viewport.zoom > 0.25 && viewport.zoom <= 1.15);
  assert.ok(Number.isFinite(viewport.x) && Number.isFinite(viewport.y));

  const payload = nodePayload(
    [camera(), pipeline()],
    [{ id: "e:x", from: "camera:cam", to: "pipeline:cam" }],
    { x: 12, y: 22, zoom: 1.25 },
    20,
  );
  assert.equal(payload.layout.version, 1);
  assert.equal(payload.layout.grid, 20);
  assert.equal(payload.layout.viewport.zoom, 1.25);
  assert.equal(payload.layout.nodes["camera:cam"].y, 40);
  assert.equal(payload.layout.nodes["pipeline:cam"].y, 400);
  assert.deepEqual(payload.edges, [
    { from: "camera:cam", to: "pipeline:cam" },
  ]);
});
