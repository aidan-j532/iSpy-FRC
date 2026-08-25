import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "iSpy" / "web" / "templates"
STATIC = ROOT / "iSpy" / "web" / "static"


def read(name: str) -> str:
    return (TEMPLATES / name).read_text(encoding="utf-8")


class TestGlobalChrome(unittest.TestCase):
    def test_base_has_favicon_and_theme_color(self):
        html = read("base.html")
        self.assertIn('rel="icon"', html)
        self.assertIn("/static/favicon.svg", html)
        self.assertIn('<meta name="theme-color" content="#0d1117">', html)

    def test_favicon_exists(self):
        svg = (STATIC / "favicon.svg").read_text(encoding="utf-8")
        self.assertIn("#0d1117", svg)  # dark bg
        self.assertIn("#2f81f7", svg)  # brand accent glyph

    def test_design_css_shared_pieces(self):
        css = (STATIC / "css" / "design.css").read_text(encoding="utf-8")
        for needle in (
            ".skeleton",
            ".skeleton-block",
            ".conn-banner",
            ".sidebar-footer",
            ".chart-waiting",
            ".empty-state.has-icon",
            ".plugin-pill.pill-tracker",
            "a.active::before",
        ):
            self.assertIn(needle, css)


class TestSetupWizardPage(unittest.TestCase):
    SOURCE = (ROOT / "iSpy" / "web" / "Backend" / "SetupWizard.py").read_text(encoding="utf-8")

    def test_links_design_system_and_favicon(self):
        self.assertIn("/static/css/design.css", self.SOURCE)
        self.assertIn("/static/favicon.svg", self.SOURCE)

    def test_no_raw_hex_feedback_colors(self):
        self.assertNotIn("#2c7", self.SOURCE)
        self.assertNotIn("#f44", self.SOURCE)

    def test_uses_design_system_classes(self):
        for cls in ("btn-ok", "setup-alert", "form-row", "form-label", "settings-section"):
            self.assertIn(cls, self.SOURCE)

    def test_submit_has_loading_state(self):
        self.assertIn("Saving", self.SOURCE)
        self.assertIn("disabled = true", self.SOURCE)


class TestInlineStyleCleanup(unittest.TestCase):
    def test_no_border_background_inline_buttons(self):
        offenders = []
        for tpl in TEMPLATES.glob("*.html"):
            if 'style="background:var(--border);"' in tpl.read_text(encoding="utf-8"):
                offenders.append(tpl.name)
        self.assertEqual(offenders, [])

    def test_no_raw_status_hex_in_templates(self):
        offenders = []
        for tpl in TEMPLATES.glob("*.html"):
            text = tpl.read_text(encoding="utf-8")
            for hexcode in ("'#2c7'", "'#f44'", "'#f66'"):
                if hexcode in text:
                    offenders.append(f"{tpl.name}:{hexcode}")
        self.assertEqual(offenders, [])


class TestLoadingAndConnectionStates(unittest.TestCase):
    CASES = {
        "dashboard.html": ["stat-value skeleton", "connLost()", "connOk()"],
        "health.html": ["stat-value skeleton", "skeleton-block", "connTracker(2)"],
        "metrics.html": ["chart-waiting", "markHasData", "connTracker(2)"],
        "logs.html": ["connTracker(2)"],
        "models.html": ["skeleton-block", "has-icon"],
        "datasets.html": ["skeleton-block", "has-icon"],
        "cameras.html": ["camerasConn = connTracker(2)", "has-icon"],
    }

    def test_pages_wired(self):
        for name, needles in self.CASES.items():
            html = read(name)
            for needle in needles:
                self.assertIn(needle, html, f"{name} missing {needle!r}")


class TestVersionEndpoint(unittest.TestCase):
    def test_api_version_serves_package_version(self):
        from iSpy import __version__
        from iSpy.config.iSpyConfig import iSpyConfig
        from iSpy.web.Backend.WebApp import create_app

        web_app = create_app(cameras=[], config=iSpyConfig())
        client = web_app.flask_app.test_client()
        r = client.get("/api/version")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["version"], __version__)


if __name__ == "__main__":
    unittest.main()
