"""Round-trip tests for the COLMAP binary I/O."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import _fixtures  # noqa: F401  (puts scripts/ on sys.path)
import _colmap_io as cio


class TestColmapIO(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_cameras_round_trip(self):
        cams = {
            1: cio.Camera(1, "OPENCV", 1920, 1080, [1000., 1000., 960., 540., 0., 0., 0., 0.]),
            2: cio.Camera(2, "PINHOLE", 640, 480, [500., 500., 320., 240.]),
        }
        cio.write_cameras_binary(cams, self.tmp / "cameras.bin")
        got = cio.read_cameras_binary(self.tmp / "cameras.bin")
        self.assertEqual(len(got), 2)
        self.assertEqual(got[1].model, "OPENCV")
        self.assertEqual(got[1].params, cams[1].params)
        self.assertEqual(got[2].width, 640)

    def test_images_round_trip(self):
        imgs = {
            1: cio.Image(1, (1., 0., 0., 0.), (0.1, 0.2, 0.3), 1, "cam00_0000.png",
                         [(1.0, 2.0), (3.0, 4.0)], [10, -1]),
            2: cio.Image(2, (0., 1., 0., 0.), (1.0, 2.0, 3.0), 2, "cam01_0000.png", [], []),
        }
        cio.write_images_binary(imgs, self.tmp / "images.bin")
        got = cio.read_images_binary(self.tmp / "images.bin")
        self.assertEqual(set(got), {1, 2})
        self.assertEqual(got[1].name, "cam00_0000.png")
        self.assertEqual(got[1].xys, [(1.0, 2.0), (3.0, 4.0)])
        self.assertEqual(got[1].point3D_ids, [10, -1])

    def test_points_round_trip(self):
        pts = {k: cio.Point3D(k, (float(k), 0., 0.), (1, 2, 3), 0.5, [1, 2], [0, 1])
               for k in range(1, 51)}
        cio.write_points3D_binary(pts, self.tmp / "points3D.bin")
        got = cio.read_points3D_binary(self.tmp / "points3D.bin")
        self.assertEqual(len(got), 50)
        self.assertEqual(got[50].xyz, (50.0, 0.0, 0.0))
        self.assertEqual(got[1].image_ids, [1, 2])

    def test_empty_points(self):
        cio.write_points3D_binary({}, self.tmp / "p.bin")
        self.assertEqual(cio.read_points3D_binary(self.tmp / "p.bin"), {})

    def test_bad_param_count_raises(self):
        cams = {1: cio.Camera(1, "OPENCV", 100, 100, [1.0, 2.0])}  # OPENCV needs 8
        with self.assertRaises(ValueError):
            cio.write_cameras_binary(cams, self.tmp / "bad.bin")


if __name__ == "__main__":
    unittest.main()
