import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

# the node-editor's pure modules (graph/logic, no DOM) are unit-tested by
# running them in node. browser-only modules (dom/canvas/etc) are covered
# manually via the /nodes page.

PURE = ["graph.js", "graphUtils.js", "history.js", "nodeDefinitions.js"]

MODULES_DIR = (
    Path(__file__).resolve().parents[1]
    / "iSpy"
    / "web"
    / "static"
    / "js"
    / "nodes"
)

RUNNER = r"""
import * as graph from './graph.mjs';
import * as utils from './graphUtils.mjs';
import { createHistory } from './history.mjs';
import { DEFAULT_PIPELINE, TEMPLATES, pipeNodeId } from './nodeDefinitions.mjs';

const assert = (cond, msg) => { if (!cond) throw new Error('assert failed: ' + msg); };
const eq = (a, b, msg) => {
  if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`assert failed: ${msg} (${JSON.stringify(a)} != ${JSON.stringify(b)})`);
  return true;
};

// ---- default pipeline + templates ----
assert(DEFAULT_PIPELINE === 'object_detection', 'default pipeline');
assert(TEMPLATES.length === 2, 'templates count');
assert(pipeNodeId('object_detection', 3) === 'pipe:object_detection:3', 'pipe id');

// ---- graph data model ----
let g = graph.createGraph();
assert(g && g.cameras && g.pipes && g.wires && g.layout, 'graph shape');
graph.loadServerGraph(g, {
  cameras: [{ key: 'cam_0', name: 'Front Cam', pipeline: 'object_detection', camera_type: 'opencv', source: 0 }],
  pipelines: [{ name: 'object_detection', config_schema: { min_conf: { label: 'min conf' } } }],
  camera_schemas: { opencv: { fps_cap: {} } },
});
assert(g.cameras['cam_0'].name === 'Front Cam', 'camera loaded');
assert(g.pipelines['object_detection'].config_schema.min_conf, 'pipeline schema loaded');
assert(g.cameraSchemas.opencv.fps_cap, 'camera schema loaded');
assert(graph.cameraPipelineName(g, 'cam_0') === 'object_detection', 'camera pipeline name');

// ---- pipeline nodes + wiring ----
const p1 = graph.addPipe(g, 'object_detection');
assert(p1 === 'pipe:object_detection:1', 'first pipe key');
const p2 = graph.addPipe(g, 'object_detection');
assert(p2 === 'pipe:object_detection:2', 'second pipe key');
graph.connect(g, 'cam_0', p1);
assert(graph.ownerCameraOf(g, p1) === 'cam_0', 'owner camel lookup');
assert(!graph.ownerCameraOf(g, p2), 'no owner on free pipe');
assert(graph.cameraWiredTo(g, p1), 'cameraWiredTo used pipe');
assert(graph.findFreePipe(g, 'object_detection') === p2, 'findFreePipe skips wired');

// ---- remove/unwire semantics ----
graph.disconnectCamera(g, 'cam_0');
assert(!graph.ownerCameraOf(g, p1), 'disconnect clears owner');
graph.connect(g, 'cam_0', p1);
graph.removePipe(g, p1);
assert(!g.pipes[p1], 'removePipe drops the node');
assert(!g.wires['cam_0'], 'removePipe unwires the camera');

// ---- prune keeps only wired pipes ----
const p3 = graph.addPipe(g, 'april_tag');
graph.connect(g, 'cam_0', p2);
graph.prunePipes(g);
assert(g.pipes[p2] && !g.pipes[p3], 'prune keeps wired, drops free');

// ---- removeCamera cleans layout + wires ----
g.layout = { cam_0: { x: 1, y: 2 }, [p2]: { x: 3, y: 4 } };
graph.removeCamera(g, 'cam_0');
assert(!g.cameras['cam_0'] && !g.layout['cam_0'] && !g.wires['cam_0'], 'removeCamera cleans everything');

// ---- nextPos grid ----
g.layout = {};
assert(eq(graph.nextPos(g), { x: 60, y: 50 }, 'nextPos first'));
g.layout.a = { x: 1, y: 1 }; g.layout.b = { x: 1, y: 1 }; g.layout.c = { x: 1, y: 1 }; g.layout.d = { x: 1, y: 1 };
assert(eq(graph.nextPos(g), { x: 60 + 4 * 80, y: 50 + 0 * 80 }, 'nextPos wraps row'));

// ---- nodeEdgePos + wirePath ----
const el = { x: 100, y: 200 };
assert(eq(utils.nodeEdgePos({ k: el }, 'k', 'out', 150, 70), { x: 250, y: 235 }, 'out port edge'));
assert(eq(utils.nodeEdgePos({ k: el }, 'k', 'in', 150, 70), { x: 100, y: 235 }, 'in port edge'));
assert(utils.wirePath(0, 0, 100, 100).startsWith('M0,0 C'), 'wire path start');

// ---- validation ----
const v = graph.createGraph();
graph.loadServerGraph(v, {
  cameras: [{ key: 'cam_0', name: 'C', pipeline: 'object_detection' }],
  pipelines: [{ name: 'object_detection' }],
});
const issues = utils.validateGraph(v);
assert(issues.some(i => i.level === 'info' && i.key === 'cam_0'), 'flags unwired camera');
const pipe = graph.addPipe(v, 'nope');
graph.connect(v, 'cam_0', pipe);
assert(utils.validateGraph(v).some(i => i.level === 'warn'), 'flags unknown pipeline');
graph.connect(v, 'cam_0', 'pipe:ghost:9');
assert(utils.validateGraph(v).some(i => i.level === 'error'), 'flags ghost wire');

// ---- history undo/redo ----
const h = createHistory();
let gg = graph.createGraph();
gg.layout = { a: { x: 0, y: 0 } };
h.push(gg);
gg.layout.a = { x: 10, y: 0 };
h.push(gg);
assert(h.canUndo() && !h.canRedo(), 'history flags after fresh pushes');
let snap = h.undo();
assert(snap.layout.a.x === 0, 'undo restores first layout');
assert(h.canRedo(), 'redo available after undo');
assert(h.undo() === null, 'no undo below bottom');
snap = h.redo();
assert(snap.layout.a.x === 10, 'redo restores second layout');
h.push(gg); // new branch after redo clears nothing ahead
assert(h.length === 3, 'history length');
const h2 = createHistory();
gg.pipes = {};
h2.push(gg);
for (let i = 0; i < 70; i++) { gg.layout['k' + i] = { x: i, y: i }; h2.push(gg); }
assert(h2.length === 60, 'history capped at 60');

console.log('nodes_graph: ok');
"""


class NodesGraphJsTests(unittest.TestCase):
    def test_pure_graph_modules_pass_in_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in PURE:
                data = (MODULES_DIR / name).read_text(encoding="utf-8")
                data = data.replace("./graph.js", "./graph.mjs").replace("./graphUtils.js", "./graphUtils.mjs").replace("./history.js", "./history.mjs").replace("./nodeDefinitions.js", "./nodeDefinitions.mjs")
                (tmp / name.replace(".js", ".mjs")).write_text(data, encoding="utf-8")
            (tmp / "runner.mjs").write_text(RUNNER, encoding="utf-8")
            r = subprocess.run(
                ["node", "runner.mjs"],
                cwd=str(tmp),
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, "node failed:\n" + r.stdout + r.stderr)
            self.assertIn("nodes_graph: ok", r.stdout)


if __name__ == "__main__":
    unittest.main()