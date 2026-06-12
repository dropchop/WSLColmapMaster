"""Tests for the shared extractor core + the import_extract CLI."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import _fixtures as fx
from _fixtures import SCRIPTS
import _extract


def _tiny_clip(path: Path, seconds: float = 1.0, rate: int = 10):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"testsrc=d={seconds}:s=160x120:rate={rate}",
                    "-pix_fmt", "yuv420p", str(path)], check=True)


def _run_import(input_root: Path, out: Path, *extra):
    cmd = [sys.executable, str(SCRIPTS / "import_extract.py"), str(input_root),
           "--out", str(out), *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


class TestNamingSchemes(unittest.TestCase):
    def test_cam4c4d_pattern(self):
        n = _extract.NAMING_SCHEMES["cam4c4d"](_extract.ShotContext(out_root=Path("/o"), cam_index=3))
        self.assertEqual(n.example(), "cam03_0000.png")
        self.assertEqual(n.out_dir, Path("/o"))

    def test_actioncam_pattern(self):
        n = _extract.NAMING_SCHEMES["actioncam"](_extract.ShotContext(
            out_root=Path("/o"), cam_id="0042", date="20260529", shot_stem="GX010042"))
        self.assertEqual(n.example(), "0042_20260529_000000.png")
        self.assertEqual(n.out_dir, Path("/o/GX010042"))

    def test_actioncam_requires_fields(self):
        with self.assertRaises(ValueError):
            _extract.NAMING_SCHEMES["actioncam"](_extract.ShotContext(out_root=Path("/o")))


class TestRecordingDate(unittest.TestCase):
    def test_fallback_to_mtime_or_today(self):
        tmp = Path(tempfile.mkdtemp())
        v = tmp / "ActionCam_1/clip.mp4"
        _tiny_clip(v)
        # testsrc clips have no creation_time tag -> must fall back, not crash
        date, src = _extract.recording_date("ffprobe", v)
        self.assertRegex(date, r"^\d{8}$")
        self.assertIn(src, {"mtime", "today"})


class TestImportExtract(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.inp = self.tmp / "in"
        (self.inp / "ActionCam_0042").mkdir(parents=True)
        (self.inp / "ActionCam_0099").mkdir(parents=True)
        (self.inp / "JunkFolder").mkdir(parents=True)
        _tiny_clip(self.inp / "ActionCam_0042" / "GX010042.MP4")
        _tiny_clip(self.inp / "ActionCam_0042" / "GX020042.MP4")
        _tiny_clip(self.inp / "ActionCam_0099" / "GX010099.MP4")
        (self.inp / "JunkFolder" / "notes.txt").write_text("ignore me")

    def test_actioncam_per_shot_folders(self):
        out = self.tmp / "out"
        r = _run_import(self.inp, out, "--naming", "actioncam", "--date", "20260529")
        self.assertEqual(r.returncode, 0, r.stderr)
        # each shot in its OWN folder
        self.assertTrue((out / "GX010042").is_dir())
        self.assertTrue((out / "GX020042").is_dir())
        self.assertTrue((out / "GX010099").is_dir())
        # naming: <ID>_<DATE>_<frame>.png
        f0 = out / "GX010042" / "0042_20260529_000000.png"
        self.assertTrue(f0.is_file())
        self.assertEqual(len(list((out / "GX010099").glob("0099_20260529_*.png"))), 10)
        # non-matching folder skipped (no output created for it)
        self.assertFalse((out / "JunkFolder").exists())

    def test_manifest_written(self):
        out = self.tmp / "out"
        _run_import(self.inp, out, "--naming", "actioncam")
        manifest = json.loads((out / "import_manifest.json").read_text())
        self.assertEqual(manifest["naming"], "actioncam")
        self.assertEqual(manifest["ok"], 3)
        self.assertEqual(manifest["failed"], 0)

    def test_dry_run_writes_nothing(self):
        out = self.tmp / "out_dry"
        r = _run_import(self.inp, out, "--naming", "actioncam", "--dry-run")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(out.exists() and any(out.rglob("*.png")))

    def test_idempotent_skip(self):
        out = self.tmp / "out"
        _run_import(self.inp, out, "--naming", "actioncam")
        r2 = _run_import(self.inp, out, "--naming", "actioncam")
        self.assertIn("skip (exists)", r2.stdout + r2.stderr)

    def test_cam4c4d_flat(self):
        out = self.tmp / "out4c"
        r = _run_import(self.inp, out, "--naming", "cam4c4d")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((out / "cam00_0000.png").is_file())
        self.assertTrue((out / "cam02_0000.png").is_file())

    def test_error_isolation(self):
        # a corrupt "video" must not sink the batch
        bad = self.inp / "ActionCam_0007"
        bad.mkdir()
        (bad / "broken.mp4").write_bytes(b"not really a video")
        out = self.tmp / "out_err"
        r = _run_import(self.inp, out, "--naming", "actioncam")
        # returncode 1 signals some failures, but good shots still extracted
        self.assertTrue((out / "GX010042").is_dir())
        manifest = json.loads((out / "import_manifest.json").read_text())
        self.assertGreaterEqual(manifest["failed"], 1)
        self.assertGreaterEqual(manifest["ok"], 3)


if __name__ == "__main__":
    unittest.main()
