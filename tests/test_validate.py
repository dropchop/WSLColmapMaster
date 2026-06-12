"""Stage 7 validation: a good dataset passes; broken ones fail loudly."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image as PILImage

import _fixtures as fx
from _fixtures import N_CAMS, N_FRAMES, W, H


def build_dataset(tmp: Path, *, n_points=1200, drop_frame=False,
                  wrong_dims=False) -> Path:
    """Assemble a final <output>/<dataset>/<scene>/ layout and return cfg path."""
    cfg_path = fx.make_config(tmp)
    cfg = yaml.safe_load(cfg_path.read_text())
    scene = (Path(cfg["output_dataset_dir"]) / cfg["dataset_name"] / cfg["scene_name"])
    images = scene / "images"
    sparse = scene / "sparse" / "0"
    images.mkdir(parents=True, exist_ok=True)

    dims = (W + 4, H) if wrong_dims else (W, H)
    blank = PILImage.new("RGB", dims, (20, 40, 60))
    for cam in range(N_CAMS):
        for fr in range(N_FRAMES):
            if drop_frame and cam == 1 and fr == N_FRAMES - 1:
                continue  # intentionally missing frame
            blank.save(images / f"cam{cam:02d}_{fr:04d}.png")

    fx.fabricate_sparse(sparse, n_cameras=N_CAMS, width=W, height=H, n_points=n_points)
    return cfg_path


class TestValidate(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_good_dataset_passes(self):
        cfg = build_dataset(self.tmp)
        r = fx.run_stage("07_validate_dataset.py", cfg)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("VALIDATION PASSED", r.stdout + r.stderr)

    def test_missing_frame_fails(self):
        cfg = build_dataset(self.tmp, drop_frame=True)
        r = fx.run_stage("07_validate_dataset.py", cfg)
        self.assertEqual(r.returncode, 1)
        self.assertIn("frame(s) missing", r.stdout + r.stderr)

    def test_too_few_points_fails(self):
        cfg = build_dataset(self.tmp, n_points=10)
        r = fx.run_stage("07_validate_dataset.py", cfg)
        self.assertEqual(r.returncode, 1)
        self.assertIn("need >=", r.stdout + r.stderr)

    def test_wrong_dims_fails(self):
        cfg = build_dataset(self.tmp, wrong_dims=True)
        r = fx.run_stage("07_validate_dataset.py", cfg)
        self.assertEqual(r.returncode, 1)


if __name__ == "__main__":
    unittest.main()
