"""End-to-end pipeline run on synthetic data.

Really executed (no mocks): 01 ingest, 02 audio-sync, 03 frame extraction,
06 pack, 07 validate.

Stubbed (and logged as such): 04 runs against a FAKE `colmap` binary that
fabricates a faithful sparse model (so 04's staging/invocation/filtering logic
is real, but feature matching is not); 05/MAtCha is replaced by a fabricated
points3D.bin (no GPU available unattended).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

import _fixtures as fx
from _fixtures import N_CAMS, N_FRAMES, OFFSETS, W, H
import _colmap_io as cio


class TestPipelineE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="pipe_e2e_"))
        cls.cfg_path = fx.make_config(cls.tmp)
        cls.cfg = yaml.safe_load(cls.cfg_path.read_text())
        fx.make_raw_inputs(cls.tmp)
        fx.make_reference_stills(cls.tmp)
        cls.session = Path(cls.cfg["session_dir"])
        cls.processed = cls.session / "processed" / cls.cfg["scene_name"]
        cls.scene_out = (Path(cls.cfg["output_dataset_dir"]) /
                         cls.cfg["dataset_name"] / cls.cfg["scene_name"])

    def test_01_ingest(self):
        r = fx.run_stage("01_ingest.py", self.cfg_path)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for cam in range(N_CAMS):
            vids = list((self.session / "ingest" / f"cam{cam:02d}").glob("*.mp4"))
            self.assertEqual(len(vids), 1, f"cam{cam:02d} ingest: {vids}")

    def test_02_sync(self):
        r = fx.run_stage("02_sync.py", self.cfg_path, "--window-seconds", "2.5")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        sync = json.loads((self.session / "sync.json").read_text())
        self.assertEqual(sync["reference_cam"], 0)
        self.assertEqual(set(sync["lags"]), {"0", "1", "2", "3"})
        self.assertEqual(sync["lags"]["0"]["seconds"], 0.0)
        # cross-correlation should recover each baked-in offset (sign-agnostic)
        for cam in (1, 2, 3):
            detected = abs(sync["lags"][str(cam)]["seconds"])
            self.assertAlmostEqual(detected, OFFSETS[cam], delta=0.05,
                                   msg=f"cam{cam} lag {detected} vs {OFFSETS[cam]}")

    def test_03_extract(self):
        r = fx.run_stage("03_extract_frames.py", self.cfg_path)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        images = self.processed / "images"
        for cam in range(N_CAMS):
            got = sorted(images.glob(f"cam{cam:02d}_*.png"))
            self.assertEqual(len(got), N_FRAMES, f"cam{cam:02d}: {len(got)} frames")
        self.assertTrue((images / "cam00_0000.png").is_file())

    def test_04_colmap_with_fake_binary(self):
        bin_dir = self.tmp / "fakebin"
        fx.write_fake_colmap(bin_dir)
        print("\n[E2E] STUB: stage 04 run against FAKE colmap (no real SfM/GPU).")
        r = fx.run_stage("04_colmap_solve.py", self.cfg_path, extra_path=bin_dir)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        sparse = self.processed / "sparse" / "0"
        cams = cio.read_cameras_binary(sparse / "cameras.bin")
        imgs = cio.read_images_binary(sparse / "images.bin")
        self.assertEqual(len(imgs), N_CAMS)
        self.assertEqual(sorted(i.name for i in imgs.values()),
                         [f"cam{c:02d}_0000.png" for c in range(N_CAMS)])
        self.assertEqual(len(cams), N_CAMS)

    def test_05_matcha_stubbed(self):
        print("[E2E] STUB: stage 05/MAtCha NOT executed (no GPU). "
              "Fabricating points3D.bin to stand in for its output.")
        fx.fabricate_points3D(self.processed / "sparse" / "0", n_points=1200)
        self.assertTrue((self.processed / "sparse" / "0" / "points3D.bin").is_file())

    def test_06_pack(self):
        r = fx.run_stage("06_pack_dataset.py", self.cfg_path)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(list((self.scene_out / "images").glob("cam*_*.png"))),
                         N_CAMS * N_FRAMES)
        for fn in ("cameras.bin", "images.bin", "points3D.bin"):
            self.assertTrue((self.scene_out / "sparse" / "0" / fn).is_file())
        self.assertTrue((self.scene_out / "pipeline_manifest.json").is_file())

    def test_07_validate(self):
        r = fx.run_stage("07_validate_dataset.py", self.cfg_path)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("VALIDATION PASSED", r.stdout + r.stderr)


def load_tests(loader, tests, pattern):
    # Force ordered execution: stages depend on prior stages' artifacts.
    suite = unittest.TestSuite()
    for name in ("test_01_ingest", "test_02_sync", "test_03_extract",
                 "test_04_colmap_with_fake_binary", "test_05_matcha_stubbed",
                 "test_06_pack", "test_07_validate"):
        suite.addTest(TestPipelineE2E(name))
    return suite


if __name__ == "__main__":
    unittest.main()
