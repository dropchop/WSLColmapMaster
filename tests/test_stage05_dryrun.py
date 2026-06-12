"""Stage 5 (MAtCha) dry-run preflight.

The real MAtCha pass needs a GPU and real footage, so it is never executed in
the suite. This test exercises 05's input validation + command construction via
--dry-run and asserts it clearly reports the stub. If the `matcha` env or the
MAtCha repo aren't present on this box, the test SKIPS with a loud reason
(reported by unittest — not a silent pass).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import _fixtures as fx
from _fixtures import SCRIPTS


def _has(cmd: str) -> bool:
    return shutil.which(cmd) is not None


class TestStage05DryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = fx.make_config(self.tmp)
        # Stage 5 preflight requires Stage 4 outputs (cameras/images.bin).
        import yaml
        cfg = yaml.safe_load(self.cfg.read_text())
        sparse = Path(cfg["session_dir"]) / "processed" / cfg["scene_name"] / "sparse" / "0"
        fx.fabricate_sparse(sparse)

    def test_dry_run_reports_stub(self):
        if not _has("conda"):
            self.skipTest("conda not on PATH — cannot exercise 05 env switch")
        matcha_repo = SCRIPTS.parent.parent / "MAtCha"
        if not (matcha_repo / "train.py").is_file():
            self.skipTest(f"MAtCha repo/train.py not found at {matcha_repo} — "
                          "05 preflight can't pass on this box")
        r = subprocess.run(["bash", str(SCRIPTS / "05_mast3r_points.sh"),
                            "--config", str(self.cfg), "--dry-run"],
                           capture_output=True, text=True)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("STUBBED", out)
        self.assertIn("--sfm_only", out)


if __name__ == "__main__":
    unittest.main()
