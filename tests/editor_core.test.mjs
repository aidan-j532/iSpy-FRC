import test from "node:test";
import assert from "node:assert/strict";

import {
  COLUMN_X,
  applyDefaults,
  cameraNameError,
  completion,
  connectedFlow,
  connectionError,
  deriveEdges,
  edgePath,
  firstEmptyCell,
  fitViewport,
  inputPorts,
  laneNodes,
  layoutNodes,
  nodePayload,
  removalPlan,
  renameCameraNodes,
  settingIsRequired,
  utilitySourceChange,
  validateGraph,
} from "../iSpy/web/static/js/editor_core.js";

const schema = {
  source: { type: "source", label: "Source", default: "" },
  device_id: { type: "text", label: "Device ID", default: "" },
  distance_threshold: {
    type: "number",
    label: "Distance",
    default: 0.5,
  },
  nullable_note: { type: "text", label: "Note", nullable: true },
};

function camera(name, lane, extra = {}) {
  return {
    id: `camera:${name}`,
    type: "camera",
    column: "camera",
    camera: name,
    label: name,
    lane,
    settings: { source: "0" },
    schema,
    ...extra,
  };
}

function lane(name, laneIndex, { utility = "util", processors = 1 } = {}) {
  const nodes = [camera(name, laneIndex)];
  for (let index = 0; index < processors; index += 1) {
    nodes.push({
      id: `frame_processor:${name}:passthrough${index || ""}`,
      type: "frame_processor",
      camera: name,
      plugin: `passthrough${index || ""}`,
      label: "Pass-through",
      lane: laneIndex,
      settings: {},
      schema: {},
    });
  }
  nodes.push({
    id: `pipeline:${name}`,
    type: "pipeline",
    camera: name,
    plugin: "object_detection",
    label: "Object Detection",
    lane: laneIndex,
    settings: {},
    schema: {},
  });
  nodes.push({
    id: `tracker:${name}:object_tracker`,
    type: "tracker",
    camera: name,
    plugin: "object_tracker",
    label: "Object Tracker",
    lane: laneIndex,
    settings: {},
    schema: {},
  });
  if (utility) {
    const existing = nodes.find(
      (node) => node.type === "utility" && node.plugin === utility,
    );
    if (!existing) {
      nodes.push({
        id: `utility:${utility}`,
        type: "utility",
        plugin: utility,
        label: "Network Tables",
        lane: null,
        settings: {},
        schema: {},
        sources: [name],
      });
    } else {
      existing.sources.push(name);
    }
  }
  return nodes;
}

function graph(twoLanes = false) {
  const nodes = lane("front", 0);
  if (twoLanes) {
    for (const node of lane("back", 1)) {
      const found = nodes.find((item) => item.id === node.id);
      if (found?.type === "utility") found.sources.push("back");
      else nodes.push(node);
    }
  }
  return nodes;
}

test("layout locks nodes to lane/column and utilities to their own order", () => {
  const nodes = graph(true);
  nodes[4].utility_index = 0;
  layoutNodes(nodes);
  const frontCamera = nodes.find((node) => node.id === "camera:front");
  const backTracker = nodes.find(
    (node) => node.id === "tracker:back:object_tracker",
  );
  const utility = nodes.find((node) => node.type === "utility");
  assert.equal(frontCamera.x, COLUMN_X.camera);
  assert.equal(backTracker.x, COLUMN_X.tracker);
  assert.ok(backTracker.y > frontCamera.y);
  assert.equal(utility.x, COLUMN_X.utility);
  assert.equal(utility.lane, null);
});

test("deriveEdges builds the five-column chain and utility branches", () => {
  const nodes = graph(true);
  layoutNodes(nodes);
  const edges = deriveEdges(nodes);
  assert.equal(edges.length, 8); // 2 x 3 internal + 2 utility fan-in
  assert.ok(
    edges.some(
      (edge) =>
        edge.from === "camera:front" &&
        edge.to === "frame_processor:front:passthrough",
    ),
  );
  assert.equal(
    edges.filter(
      (edge) => edge.to === "utility:util" && edge.kind === "tracker_utility",
    ).length,
    2,
  );
});

test("completion only counts explicit non-blank values; defaults are ghosts", () => {
  const values = { source: "0" };
  assert.equal(settingIsRequired(schema, values, "distance_threshold"), true);
  assert.equal(settingIsRequired(schema, values, "device_id"), false);
  assert.equal(settingIsRequired(schema, values, "nullable_note"), false);
  assert.equal(completion(schema, values).complete, false);
  assert.deepEqual(completion(schema, values).missing, ["distance_threshold"]);
  const confirmed = applyDefaults({
    schema,
    settings: values,
    defaults: { distance_threshold: 0.5, device_id: "" },
  });
  assert.equal(confirmed.settings.distance_threshold, 0.5);
  assert.equal(confirmed.settings.device_id, "");
  assert.equal(confirmed.completion.complete, true);
});

test("validateGraph reports lane-specific errors", () => {
  const nodes = graph();
  const missing = validateGraph(
    nodes.filter((node) => node.type !== "tracker" && node.type !== "utility"),
    [],
  );
  assert.ok(
    missing.some(
      (error) =>
        error.code === "lane_missing_tracker" &&
        error.message.includes("Lane 'front'"),
    ),
  );
  const zeroInput = nodes.map((node) =>
    node.type === "utility" ? { ...node, sources: [] } : node,
  );
  const errors = validateGraph(zeroInput, []);
  assert.ok(
    errors.some(
      (error) =>
        error.code === "utility_no_inputs" &&
        error.message === "Utility 'util' needs at least one tracker input.",
    ),
  );
  assert.ok(
    errors.some(
      (error) =>
        error.code === "tracker_without_utility" &&
        error.message.includes("Tracker 'object_tracker' on lane 'front'"),
    ),
  );
});

test("validateGraph flags duplicate cells and unsupported processors", () => {
  const nodes = graph();
  const tracker = nodes.find((node) => node.type === "tracker");
  nodes.push({ ...tracker, id: "tracker:front:second", plugin: "second" });
  const processor = nodes.find((node) => node.type === "frame_processor");
  nodes.find((node) => node.type === "pipeline").plugin = "apriltag";
  processor.supported_pipelines = ["object_detection"];
  const errors = validateGraph(nodes, []);
  assert.ok(errors.some((error) => error.code === "duplicate_column"));
  assert.ok(errors.some((error) => error.code === "edge_pipeline_unsupported"));
});

test("connectionError mirrors lane, direction, skip, support and cycle rules", () => {
  const nodes = graph(true);
  layoutNodes(nodes);
  const pipeline = nodes.find((node) => node.id === "pipeline:front");
  const tracker = nodes.find((node) => node.id === "tracker:front:object_tracker");
  const otherProcessor = nodes.find(
    (node) => node.id === "frame_processor:back:passthrough",
  );
  const utility = nodes.find((node) => node.type === "utility");
  const camera = nodes.find((node) => node.id === "camera:front");
  assert.equal(
    connectionError(camera, pipeline, nodes, []),
    "Edges cannot skip a column. Connect Camera directly to the next column.",
  );
  assert.equal(
    connectionError(tracker, pipeline, nodes, []),
    "Edges must move left to right to the next column.",
  );
  assert.equal(
    connectionError(camera, otherProcessor, nodes, []),
    "Edges may only connect nodes in the same camera lane, except a Tracker -> Utility connection.",
  );
  assert.equal(
    connectionError(tracker, utility, nodes, []),
    "",
  );
  assert.equal(
    connectionError(tracker, camera, nodes, []),
    "Edges must move left to right to the next column.",
  );
  assert.equal(
    connectionError(utility, tracker, nodes, []),
    "Utility nodes have no output. Edges must move left to right.",
  );
});

test("edgePath is always a forward horizontal cubic bezier", () => {
  const source = { type: "camera", x: 10, y: 20 };
  const target = { type: "frame_processor", x: 300, y: 240 };
  const path = edgePath(source, target);
  assert.match(path, /^M 234 72 C /);
  assert.ok(path.includes("300"));
  const backwards = edgePath(
    { type: "tracker", x: 600, y: 20 },
    { type: "utility", x: 300, y: 20 },
  );
  assert.match(backwards, /C /);
  assert.ok(backwards.includes("C "));
});

test("connectedFlow keeps other lanes into the same utility untouched", () => {
  const nodes = graph(true);
  layoutNodes(nodes);
  const edges = deriveEdges(nodes);
  const frontEdge = edges.find((edge) => edge.from === "camera:front");
  const flow = connectedFlow(edges, frontEdge.id);
  assert.ok(
    flow.has(
      "edge:tracker:front:object_tracker->utility:util",
    ),
  );
  assert.ok(
    !flow.has("edge:tracker:back:object_tracker->utility:util"),
  );
  assert.equal(flow.size, 4);
});

test("utility input ports are N evenly spaced left-edge dots", () => {
  const nodes = graph(true);
  layoutNodes(nodes);
  const utility = nodes.find((node) => node.type === "utility");
  const ports = inputPorts(utility, nodes);
  assert.equal(ports.length, 2);
  assert.deepEqual(
    ports.map((port) => port.camera),
    ["front", "back"],
  );
  assert.ok(ports[0].y < ports[1].y);
  assert.equal(ports[0].x, utility.x);
  assert.deepEqual(inputPorts(nodes.find((node) => node.type === "camera")), []);
});

test("nodePayload strips coordinates and keeps explicit settings only", () => {
  const nodes = graph();
  layoutNodes(nodes);
  const edges = deriveEdges(nodes);
  const payload = nodePayload(nodes, edges, { x: 12, y: 34, zoom: 0.8 });
  assert.equal(payload.layout.viewport.zoom, 0.8);
  for (const node of payload.nodes) {
    assert.equal("x" in node, false);
    assert.equal("y" in node, false);
  }
  assert.equal(payload.nodes[0].settings.source, "0");
  assert.equal("device_id" in payload.nodes[0].settings, false);
  assert.deepEqual(payload.layout.lane_order, ["front"]);
  assert.deepEqual(payload.layout.utility_order, ["util"]);
});

test("lane helpers add into the selected/first empty lane and utility once", () => {
  const nodes = graph();
  layoutNodes(nodes);
  assert.deepEqual(firstEmptyCell(nodes, "tracker", 0), null);
  const withoutTracker = nodes.filter((node) => node.type !== "tracker");
  assert.deepEqual(firstEmptyCell(withoutTracker, "tracker", 0), {
    lane: 0,
    camera: "front",
  });
  assert.equal(firstEmptyCell(nodes, "utility"), null);
  const withoutUtility = nodes.filter((node) => node.type !== "utility");
  assert.deepEqual(firstEmptyCell(withoutUtility, "utility"), {
    lane: null,
    camera: null,
  });
});

test("camera rename updates lane nodes and utility sources", () => {
  const nodes = graph();
  assert.equal(cameraNameError("front", nodes), "Camera 'front' already exists.");
  const renamed = renameCameraNodes(nodes, "front", "left");
  assert.ok(renamed.some((node) => node.id === "camera:left"));
  assert.ok(renamed.some((node) => node.id === "pipeline:left"));
  assert.deepEqual(
    renamed.find((node) => node.type === "utility").sources,
    ["left"],
  );
});

test("removing a camera removes its lane and utility source", () => {
  const nodes = graph(true);
  layoutNodes(nodes);
  const plan = removalPlan(nodes, "camera:back");
  assert.ok(plan.remove.includes("tracker:back:object_tracker"));
  const utility = plan.nodes.find((node) => node.type === "utility");
  assert.deepEqual(utility.sources, ["front"]);
});

test("utilitySourceChange filters unknown cameras and dedupes", () => {
  const nodes = graph(true);
  const changed = utilitySourceChange(nodes, "util", [
    "back",
    "ghost",
    "front",
    "back",
  ]);
  assert.deepEqual(
    changed.find((node) => node.type === "utility").sources,
    ["back", "front"],
  );
});

test("fitViewport centers the fixed canvas without free node positions", () => {
  const viewport = fitViewport(
    { width: 1400, height: 500 },
    { width: 900, height: 700 },
  );
  assert.ok(viewport.zoom > 0.25 && viewport.zoom <= 1.4);
  assert.ok(viewport.x <= 48);
  assert.ok(viewport.y >= 0);
  const cameraNodes = laneNodes(graph(), "camera");
  assert.equal(cameraNodes.length, 1);
});
