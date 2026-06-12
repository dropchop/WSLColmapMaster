#!/usr/bin/env python3
"""Stage 7 — validate the packed dataset before training handoff.

Non-destructive pre-training sanity checks on the final
`<output>/<dataset>/<scene>/` layout. Reports every problem found (doesn't stop
at the first) and exits non-zero if any check fails.

Checks:
  - all 4 x N_FRAMES frames exist with the cam{XX}_{YYYY}.png naming
  - cameras.bin has the expected camera count (n_cameras, or 1 if single_camera)
  - images.bin has exactly n_cameras poses, named for the camera first-frames
  - points3D.bin has >= validate.min_points3d points
  - each camera's frame resolution matches its cameras.bin intrinsics + config
  - no PNG exceeds validate.max_image_size_mb

Usage:
    python 07_validate_dataset.py [--config path]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image as PILImage

import _colmap_io as cio
from _utils import load_config, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the packed dataset.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    log = setup_logging("07_validate_dataset")
    cfg = load_config(args.config)

    images_dir = cfg.scene_images_dir()
    sparse_dir = cfg.scene_sparse_dir()
    validate = cfg.raw.get("validate", {})
    min_points = int(validate.get("min_points3d", 1000))
    max_mb = float(validate.get("max_image_size_mb", 10))
    single_camera = bool(cfg.raw.get("colmap_single_camera", False))

    errors: list[str] = []

    def check(cond: bool, ok_msg: str, err_msg: str):
        if cond:
            log.info("OK   : %s", ok_msg)
        else:
            log.error("FAIL : %s", err_msg)
            errors.append(err_msg)

    # --- frames present + correctly named ---
    if not images_dir.is_dir():
        errors.append(f"images dir missing: {images_dir}")
        log.error("FAIL : images dir missing: %s", images_dir)
    else:
        for cam in range(cfg.n_cameras):
            missing = [cfg.frame_name(cam, fr) for fr in range(cfg.n_frames)
                       if not (images_dir / cfg.frame_name(cam, fr)).is_file()]
            check(not missing,
                  f"cam{cam:02d}: all {cfg.n_frames} frames present",
                  f"cam{cam:02d}: {len(missing)} frame(s) missing "
                  f"(first: {missing[0] if missing else '-'})")

        # oversized PNGs
        big = [p.name for p in images_dir.glob("cam*_*.png")
               if p.stat().st_size > max_mb * 1e6]
        check(not big, f"no PNG exceeds {max_mb} MB",
              f"{len(big)} PNG(s) exceed {max_mb} MB (e.g. {big[:1]})")

    # --- sparse model ---
    cams_bin = sparse_dir / "cameras.bin"
    imgs_bin = sparse_dir / "images.bin"
    pts_bin = sparse_dir / "points3D.bin"
    cameras = images = points = None

    if cams_bin.is_file():
        cameras = cio.read_cameras_binary(cams_bin)
        expected = 1 if single_camera else cfg.n_cameras
        check(len(cameras) == expected,
              f"cameras.bin has {len(cameras)} camera(s)",
              f"cameras.bin has {len(cameras)} camera(s), expected {expected}")
    else:
        errors.append("cameras.bin missing")
        log.error("FAIL : cameras.bin missing: %s", cams_bin)

    if imgs_bin.is_file():
        images = cio.read_images_binary(imgs_bin)
        names = sorted(img.name for img in images.values())
        wanted = sorted(f"cam{c:02d}_0000.png" for c in range(cfg.n_cameras))
        check(len(images) == cfg.n_cameras,
              f"images.bin has {len(images)} pose(s)",
              f"images.bin has {len(images)} pose(s), expected {cfg.n_cameras}")
        check(names == wanted,
              "images.bin pose names match camera first-frames",
              f"images.bin pose names {names} != expected {wanted}")
    else:
        errors.append("images.bin missing")
        log.error("FAIL : images.bin missing: %s", imgs_bin)

    if pts_bin.is_file():
        points = cio.read_points3D_binary(pts_bin)
        check(len(points) >= min_points,
              f"points3D.bin has {len(points)} points (>= {min_points})",
              f"points3D.bin has {len(points)} points, need >= {min_points}")
    else:
        errors.append("points3D.bin missing")
        log.error("FAIL : points3D.bin missing: %s", pts_bin)

    # --- resolution agreement (PNG vs intrinsics vs config) ---
    if cameras and images and images_dir.is_dir():
        for img in images.values():
            png = images_dir / img.name
            if not png.is_file():
                continue
            with PILImage.open(png) as im:
                w, h = im.size
            cam = cameras.get(img.camera_id)
            cam_ok = cam is not None and cam.width == w and cam.height == h
            cfg_ok = (w, h) == (cfg.width, cfg.height)
            check(cam_ok and cfg_ok,
                  f"{img.name}: {w}x{h} matches intrinsics + config",
                  f"{img.name}: PNG {w}x{h} vs intrinsics "
                  f"{cam.width if cam else '?'}x{cam.height if cam else '?'} "
                  f"vs config {cfg.width}x{cfg.height}")

    if errors:
        log.error("VALIDATION FAILED with %d problem(s):", len(errors))
        for e in errors:
            log.error("  - %s", e)
        return 1
    log.info("VALIDATION PASSED — dataset is ready for 4C4D training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
