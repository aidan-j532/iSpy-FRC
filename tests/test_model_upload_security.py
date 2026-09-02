import io
import os
import pickle
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import flask

from iSpy.config.iSpyConfig import iSpyConfig
from iSpy.vision.genericYolo import torch_load
from iSpy.vision.metadata import metadata_from_pt
from iSpy.web.modules.models import ModelsModule


_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_DETECT_PT = _REPO / "iSpy" / "assets" / "_default_detect.pt"
_DEFAULT_POSE_PT = _REPO / "iSpy" / "assets" / "_default_pose.pt"


class _EvilPickle:
    """__reduce__ gadget that runs os.system if the pickle is executed."""

    def __init__(self, marker: Path):
        self._marker = str(marker)

    def __reduce__(self):
        return (os.system, (f"echo pwned >> {self._marker}",))


class RestrictedTorchLoadTests(unittest.TestCase):
    """BUG 4: torch_load(trusted=False) must never run pickle gadgets."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ispy_up_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.marker = self.tmp / "pwned.txt"

    def _evil_bytes(self) -> bytes:
        return pickle.dumps(_EvilPickle(self.marker))

    def assertNoSideEffect(self):
        self.assertFalse(self.marker.exists())

    def test_plain_pickle_gadget_refused(self):
        evil = self.tmp / "evil.pt"
        evil.write_bytes(self._evil_bytes())
        with self.assertRaises(Exception):
            torch_load(evil, trusted=False)
        self.assertNoSideEffect()

    def test_torch_zip_with_evil_dataload_refused(self):
        evil = self.tmp / "evil_zip.pt"
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("payload/data.pkl", self._evil_bytes())
            zf.writestr("payload/byteorder", "little")
        with self.assertRaises(Exception):
            torch_load(evil, trusted=False)
        self.assertNoSideEffect()

    def test_direct_os_system_global_refused(self):
        evil = self.tmp / "evil_os.pt"
        evil.write_bytes(pickle.dumps((os.system, os.getcwd)))
        with self.assertRaises(Exception):
            torch_load(evil, trusted=False)
        self.assertNoSideEffect()

    def test_genuine_detect_checkpoint_loads_restricted(self):
        meta = metadata_from_pt(_DEFAULT_DETECT_PT, trusted=False)
        self.assertEqual(meta["task"], "detect")
        self.assertEqual(meta["nc"], 80)
        self.assertIn(0, meta["names"])

    def test_genuine_pose_checkpoint_loads_restricted(self):
        meta = metadata_from_pt(_DEFAULT_POSE_PT, trusted=False)
        self.assertEqual(meta["task"], "pose")

    def test_trusted_default_path_unchanged(self):
        meta = metadata_from_pt(_DEFAULT_DETECT_PT)
        self.assertEqual(meta["task"], "detect")


class ModelUploadAccessTests(unittest.TestCase):
    """BUG 4: the upload route is local/token-only and never pickles hostile bytes."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ispy_uproute_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.marker = self.tmp / "pwned.txt"
        self.pytorch_dir = self.tmp / "models" / "pytorch"
        cfg = iSpyConfig(file_path=str(self.tmp / "config.json"))
        cfg.config["app_mode"] = False
        self.app = flask.Flask(__name__)
        self.mod = ModelsModule({"config": cfg})
        self.mod.pytorch_dir = self.pytorch_dir
        self.pytorch_dir.mkdir(parents=True, exist_ok=True)
        self.mod.register_routes(self.app)
        self.client = self.app.test_client()

    def _evil_file(self) -> tuple:
        return (io.BytesIO(pickle.dumps(_EvilPickle(self.marker))), "evil.pt")

    def test_remote_without_token_rejected(self):
        r = self.client.post(
            "/api/models/upload",
            data={"file": self._evil_file()},
            content_type="multipart/form-data",
            environ_base={"REMOTE_ADDR": "8.8.8.8"},
        )
        self.assertEqual(r.status_code, 403)
        self.assertFalse(self.marker.exists())
        self.assertFalse(any(self.pytorch_dir.glob("*")))

    def test_remote_with_wrong_token_rejected(self):
        with mock.patch.dict(os.environ, {"ISPY_ADMIN_TOKEN": "sekrit"}):
            r = self.client.post(
                "/api/models/upload",
                data={"file": self._evil_file()},
                content_type="multipart/form-data",
                headers={"X-iSpy-Admin-Token": "nope"},
                environ_base={"REMOTE_ADDR": "8.8.8.8"},
            )
        self.assertEqual(r.status_code, 403)
        self.assertFalse(self.marker.exists())

    def test_local_upload_evil_bytes_rejected_without_side_effect(self):
        r = self.client.post(
            "/api/models/upload",
            data={"file": self._evil_file()},
            content_type="multipart/form-data",
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("Metadata generation failed", r.json["error"])
        self.assertFalse(self.marker.exists())
        self.assertFalse(any(self.pytorch_dir.glob("*")))

    def test_remote_with_valid_token_still_rejects_evil_bytes(self):
        with mock.patch.dict(os.environ, {"ISPY_ADMIN_TOKEN": "sekrit"}):
            r = self.client.post(
                "/api/models/upload",
                data={"file": self._evil_file()},
                content_type="multipart/form-data",
                headers={"X-iSpy-Admin-Token": "sekrit"},
                environ_base={"REMOTE_ADDR": "8.8.8.8"},
            )
        self.assertEqual(r.status_code, 400)
        self.assertFalse(self.marker.exists())

    def test_local_upload_genuine_model_succeeds(self):
        raw = _DEFAULT_DETECT_PT.read_bytes()
        r = self.client.post(
            "/api/models/upload",
            data={"file": (io.BytesIO(raw), "_default_detect.pt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(r.status_code, 200, r.json)
        self.assertTrue(r.json["success"])
        dest = self.pytorch_dir / "default_detect.pt"
        self.assertTrue(dest.exists())
        self.assertTrue((self.pytorch_dir / "default_detect_metadata.yaml").exists())
        self.assertFalse(any(self.pytorch_dir.glob("*.pt.uploading")))


if __name__ == "__main__":
    unittest.main()