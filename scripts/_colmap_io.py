"""Minimal reader/writer for COLMAP's binary model files.

Implements the documented little-endian binary layout of `cameras.bin`,
`images.bin`, and `points3D.bin` (matching COLMAP's own
`scripts/python/read_write_model.py`). Used to:

  - filter a reconstruction down to the 4 video poses (stage 4),
  - copy/verify the model during packing (stage 6),
  - validate camera/image/point counts and intrinsics (stage 7),
  - fabricate synthetic models in the test suite.

Only the subset the pipeline needs is implemented, but the binary layout is
faithful so files round-trip and interoperate with real COLMAP output.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

# model_id -> (name, num_params)
CAMERA_MODEL_IDS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_NAMES = {name: (mid, n) for mid, (name, n) in CAMERA_MODEL_IDS.items()}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: list[float]


@dataclass
class Image:
    id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str
    xys: list[tuple[float, float]] = field(default_factory=list)
    point3D_ids: list[int] = field(default_factory=list)


@dataclass
class Point3D:
    id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float
    image_ids: list[int] = field(default_factory=list)
    point2D_idxs: list[int] = field(default_factory=list)


def _read(f, fmt: str):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(size))


# ----------------------------- cameras.bin -----------------------------------

def read_cameras_binary(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with open(path, "rb") as f:
        (num,) = _read(f, "<Q")
        for _ in range(num):
            cam_id, model_id, width, height = _read(f, "<iiQQ")
            name, nparams = CAMERA_MODEL_IDS[model_id]
            params = list(_read(f, "<" + "d" * nparams))
            cameras[cam_id] = Camera(cam_id, name, width, height, params)
    return cameras


def write_cameras_binary(cameras: dict[int, Camera], path: Path) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras.values():
            model_id, nparams = CAMERA_MODEL_NAMES[cam.model]
            if len(cam.params) != nparams:
                raise ValueError(f"camera {cam.id}: {cam.model} needs {nparams} "
                                 f"params, got {len(cam.params)}")
            f.write(struct.pack("<iiQQ", cam.id, model_id, cam.width, cam.height))
            f.write(struct.pack("<" + "d" * nparams, *cam.params))


# ----------------------------- images.bin ------------------------------------

def read_images_binary(path: Path) -> dict[int, Image]:
    images: dict[int, Image] = {}
    with open(path, "rb") as f:
        (num,) = _read(f, "<Q")
        for _ in range(num):
            image_id = _read(f, "<i")[0]
            qvec = _read(f, "<dddd")
            tvec = _read(f, "<ddd")
            camera_id = _read(f, "<i")[0]
            name_chars = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00" or c == b"":
                    break
                name_chars += c
            name = name_chars.decode("utf-8")
            (num_pts,) = _read(f, "<Q")
            xys: list[tuple[float, float]] = []
            p3d: list[int] = []
            for _ in range(num_pts):
                x, y, pid = _read(f, "<ddq")
                xys.append((x, y))
                p3d.append(pid)
            images[image_id] = Image(image_id, qvec, tvec, camera_id, name, xys, p3d)
    return images


def write_images_binary(images: dict[int, Image], path: Path) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img in images.values():
            f.write(struct.pack("<i", img.id))
            f.write(struct.pack("<dddd", *img.qvec))
            f.write(struct.pack("<ddd", *img.tvec))
            f.write(struct.pack("<i", img.camera_id))
            f.write(img.name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", len(img.xys)))
            for (x, y), pid in zip(img.xys, img.point3D_ids):
                f.write(struct.pack("<ddq", x, y, pid))


# ----------------------------- points3D.bin ----------------------------------

def read_points3D_binary(path: Path) -> dict[int, Point3D]:
    points: dict[int, Point3D] = {}
    with open(path, "rb") as f:
        (num,) = _read(f, "<Q")
        for _ in range(num):
            pid = _read(f, "<Q")[0]
            xyz = _read(f, "<ddd")
            rgb = _read(f, "<BBB")
            (error,) = _read(f, "<d")
            (track_len,) = _read(f, "<Q")
            image_ids, p2d_idxs = [], []
            for _ in range(track_len):
                iid, p2d = _read(f, "<ii")
                image_ids.append(iid)
                p2d_idxs.append(p2d)
            points[pid] = Point3D(pid, xyz, rgb, error, image_ids, p2d_idxs)
    return points


def write_points3D_binary(points: dict[int, Point3D], path: Path) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pt in points.values():
            f.write(struct.pack("<Q", pt.id))
            f.write(struct.pack("<ddd", *pt.xyz))
            f.write(struct.pack("<BBB", *pt.rgb))
            f.write(struct.pack("<d", pt.error))
            f.write(struct.pack("<Q", len(pt.image_ids)))
            for iid, p2d in zip(pt.image_ids, pt.point2D_idxs):
                f.write(struct.pack("<ii", iid, p2d))
