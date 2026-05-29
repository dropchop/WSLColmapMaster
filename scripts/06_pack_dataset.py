#!/usr/bin/env python3
"""Stage 6 — pack staged artifacts into the final 4C4D dataset layout.

Assembles:

    <output_dataset_dir>/<dataset_name>/<scene_name>/
    ├── images/   cam{XX}_{YYYY}.png   (from Stage 3)
    └── sparse/0/
        ├── cameras.bin                (from Stage 4)
        ├── images.bin                 (from Stage 4)
        └── points3D.bin               (from Stage 5)

and writes a `pipeline_manifest.json` recording counts + sources for traceability.

Usage:
    python 06_pack_dataset.py [--config path] [--move]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import _colmap_io as cio
from _utils import load_config, setup_logging, write_json

SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack the final 4C4D dataset.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--move", action="store_true",
                        help="move files instead of copying (saves disk)")
    args = parser.parse_args()

    log = setup_logging("06_pack_dataset")
    cfg = load_config(args.config)
    place = shutil.move if args.move else shutil.copy2

    src_images = cfg.processed_images_dir()
    src_sparse = cfg.processed_sparse_dir()
    if not src_images.is_dir():
        log.error("staged images not found: %s (run Stage 3)", src_images)
        return 2
    for fn in SPARSE_FILES:
        if not (src_sparse / fn).is_file():
            log.error("staged sparse file missing: %s (run Stages 4 & 5)", src_sparse / fn)
            return 2

    dst_images = cfg.scene_images_dir()
    dst_sparse = cfg.scene_sparse_dir()
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_sparse.mkdir(parents=True, exist_ok=True)

    frames = sorted(src_images.glob("cam*_*.png"))
    if not frames:
        log.error("no extracted frames in %s", src_images)
        return 2
    for f in frames:
        place(str(f), str(dst_images / f.name))
    log.info("placed %d frame(s) -> %s", len(frames), dst_images)

    for fn in SPARSE_FILES:
        place(str(src_sparse / fn), str(dst_sparse / fn))
    log.info("placed sparse model (%s) -> %s", ", ".join(SPARSE_FILES), dst_sparse)

    cameras = cio.read_cameras_binary(dst_sparse / "cameras.bin")
    images = cio.read_images_binary(dst_sparse / "images.bin")
    points = cio.read_points3D_binary(dst_sparse / "points3D.bin")

    manifest = {
        "tool": "06_pack_dataset",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": cfg.dataset_name,
        "scene_name": cfg.scene_name,
        "output_dir": str(cfg.scene_output_dir()),
        "n_cameras_config": cfg.n_cameras,
        "n_frames_config": cfg.n_frames,
        "counts": {
            "image_files": len(frames),
            "cameras": len(cameras),
            "poses": len(images),
            "points3D": len(points),
        },
    }
    write_json(cfg.scene_output_dir() / "pipeline_manifest.json", manifest)
    log.info("manifest written: %s", cfg.scene_output_dir() / "pipeline_manifest.json")
    log.info("Stage 6 complete: %d frames, %d cameras, %d poses, %d points -> %s",
             len(frames), len(cameras), len(images), len(points), cfg.scene_output_dir())
    return 0


if __name__ == "__main__":
    sys.exit(main())
