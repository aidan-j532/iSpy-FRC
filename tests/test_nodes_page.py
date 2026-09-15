import unittest
from pathlib import Path

from iSpy.config.iSpyConfig import iSpyConfig
from iSpy.web.Backend.WebApp import create_app

TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "iSpy"
    / "web"
    / "templates"
    / "nodes.html"
)

STATIC_JS = (
    Path(__file__).resolve().parents[1]
    / "iSpy"
    / "web"
    / "static"
    / "js"
    / "nodes"
)


def _app():
    cfg = iSpyConfig()
    cfg.config["camera_configs"] = {
        "cam_0": {
            "name": "Front Cam",
            "source": 0,
            "camera_type": "opencv",
            "device_id": "dev_0",
            "pipeline": {
                "name": "object_detection",
                "settings": {"min_conf": 0.5},
            },
        }
    }
    web_app = create_app(cameras=[], config=cfg)
    return web_app.flask_app.test_client()


class NodesPageTests(unittest.TestCase):
    def test_page_renders_flow_graph_editor(self):
        client = _app()
        r = client.get("/nodes")
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        for needle in (
            "Flow Graph",
            "node-canvas",
            "graph-layer",
            "wire-svg",
            "ctx-menu",
        ):
            self.assertIn(needle, html, f"page missing {needle!r}")

    def test_graph_endpoint_ships_cameras_schemas_and_pipelines(self):
        client = _app()
        r = client.get("/api/nodes/graph")
        self.assertEqual(r.status_code, 200)
        payload = r.get_json()

        cams = {c["key"]: c for c in payload["cameras"]}
        self.assertIn("cam_0", cams)
        cam = cams["cam_0"]
        self.assertEqual(cam["name"], "Front Cam")
        self.assertEqual(cam["camera_type"], "opencv")
        self.assertEqual(cam["pipeline"], "object_detection")

        # the full config entry rides along so the inspector can edit
        # every real key, not just the short display list
        entry = cam["entry"]
        self.assertEqual(entry["source"], 0)
        self.assertEqual(entry["pipeline"]["name"], "object_detection")
        self.assertEqual(cam["pipeline_settings"]["min_conf"], 0.5)
        self.assertEqual(entry["device_id"], "dev_0")

        self.assertIn("opencv", payload["camera_schemas"])
        self.assertIn("fps_cap", payload["camera_schemas"]["opencv"])
        names = [p["name"] for p in payload["pipelines"]]
        self.assertIn("object_detection", names)


class NodesTemplateShapeTests(unittest.TestCase):
    def test_editor_markers_present(self):
        html = TEMPLATE.read_text(encoding="utf-8")
        for needle in (
            "id=\"node-canvas\"",
            "id=\"graph-layer\"",
            "id=\"wire-svg\"",
            "id=\"ctx-menu\"",
            "id=\"settings-modal\"",
            "id=\"palette\"",
        ):
            self.assertIn(needle, html, f"template missing {needle!r}")

    def test_wires_use_world_coord_helper(self):
        # wires are drawn in world coords off the node box, so pan/zoom
        # and node drags never desync the svg from the nodes. the helper
        # moved into the modules during the editor rewrite - assert the
        # behavior there, not in the template.
        graph_utils = (STATIC_JS / "graphUtils.js").read_text(encoding="utf-8")
        self.assertIn("export function nodeEdgePos(", graph_utils)
        self.assertIn("layout[key] || { x: 0, y: 0 }", graph_utils)
        self.assertNotIn("portWorldPos", graph_utils)

        connections = (STATIC_JS / "connections.js").read_text(encoding="utf-8")
        self.assertIn("'out'", connections)
        self.assertIn("worldEdge(", connections)
        self.assertNotIn("portWorldPos", connections)

    def test_editor_ships_as_modules(self):
        # the editor is modularised into esm files; the template should
        # only import the entry point, not carry a giant inline script
        html = TEMPLATE.read_text(encoding="utf-8")
        self.assertIn('type="module"', html)
        self.assertIn("/static/js/nodes/editor.js", html)
        for name in (
            "graph.js",
            "graphUtils.js",
            "history.js",
            "nodeDefinitions.js",
            "canvas.js",
            "connections.js",
            "renderer.js",
            "selection.js",
            "palette.js",
            "inspector.js",
            "contextMenu.js",
            "editor.js",
        ):
            self.assertTrue((STATIC_JS / name).is_file(), f"missing module {name}")


if __name__ == "__main__":
    unittest.main()