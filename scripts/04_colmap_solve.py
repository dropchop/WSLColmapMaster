#!/usr/bin/env python3
"""Stage 4 — solve camera poses with COLMAP (reference-stills technique).

Builds a pose-solve image set from the reference stills plus the first frame of
each camera, runs the COLMAP SfM pipeline, then filters the reconstruction down
to just the four video-source poses and exports `cameras.bin` + `images.bin`
into the scene's `sparse/0/`. (The dense `points3D.bin` comes from Stage 5.)

GPU note: the apt COLMAP on this box is a CPU-only build, so both SIFT GPU
flags are forced to 0. For a ~12-image solve that is plenty fast.

Usage:
    python 04_colmap_solve.py [--config path]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import _colmap_io as cio
from _utils import load_config, require_tool, run, setup_logging

# First frame of each camera, the poses we ultimately keep.
def first_frame_name(cam: int) -> str:
    return f"cam{cam:02d}_0000.png"


def stage_pose_images(cfg, log) -> Path:
    """Assemble reference stills + each camera's first frame into one folder."""
    pose_dir = cfg.colmap_workdir() / "images_for_pose"
    if pose_dir.exists():
        shutil.rmtree(pose_dir)
    pose_dir.mkdir(parents=True)

    n_stills = 0
    if cfg.reference_stills_dir.is_dir():
        for still in sorted(cfg.reference_stills_dir.iterdir()):
            if still.is_file() and still.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                shutil.copy2(still, pose_dir / f"ref_{still.name}")
                n_stills += 1
    else:
        log.warning("reference_stills_dir not found: %s (poses may be unstable "
                    "with only 4 images)", cfg.reference_stills_dir)

    images_dir = cfg.processed_images_dir()
    n_firsts = 0
    for cam in range(cfg.n_cameras):
        ff = images_dir / first_frame_name(cam)
        if not ff.is_file():
            log.error("missing first frame for cam%02d: %s (run Stage 3 first)", cam, ff)
            return None
        shutil.copy2(ff, pose_dir / first_frame_name(cam))
        n_firsts += 1

    log.info("pose image set: %d reference still(s) + %d camera first-frame(s) -> %s",
             n_stills, n_firsts, pose_dir)
    return pose_dir


def run_colmap(colmap: str, cfg, pose_dir: Path, log) -> Path:
    """Run feature_extractor -> exhaustive_matcher -> mapper. Returns sparse root."""
    work = cfg.colmap_workdir()
    db = work / "database.db"
    sparse = work / "sparse"
    if db.exists():
        db.unlink()
    sparse.mkdir(parents=True, exist_ok=True)

    single = "1" if bool(cfg.raw.get("colmap_single_camera", False)) else "0"
    model = cfg.raw.get("colmap_camera_model", "OPENCV")

    run([colmap, "feature_extractor",
         "--database_path", str(db),
         "--image_path", str(pose_dir),
         "--ImageReader.camera_model", model,
         "--ImageReader.single_camera", single,
         "--SiftExtraction.use_gpu", "0"], log, capture_output=True)

    run([colmap, "exhaustive_matcher",
         "--database_path", str(db),
         "--SiftMatching.use_gpu", "0"], log, capture_output=True)

    run([colmap, "mapper",
         "--database_path", str(db),
         "--image_path", str(pose_dir),
         "--output_path", str(sparse)], log, capture_output=True)
    return sparse


def pick_best_model(sparse_root: Path, log) -> Path:
    """COLMAP writes one or more models as numbered subdirs; pick the largest."""
    models = [d for d in sparse_root.iterdir() if d.is_dir() and (d / "images.bin").exists()]
    if not models:
        log.error("COLMAP produced no reconstruction under %s", sparse_root)
        return None
    best = max(models, key=lambda d: len(cio.read_images_binary(d / "images.bin")))
    log.info("selected reconstruction %s (%d registered images)",
             best.name, len(cio.read_images_binary(best / "images.bin")))
    return best


def filter_to_video_poses(model_dir: Path, cfg, log) -> bool:
    """Keep only the 4 camera first-frame poses; export to processed sparse/0/."""
    cameras = cio.read_cameras_binary(model_dir / "cameras.bin")
    images = cio.read_images_binary(model_dir / "images.bin")

    wanted = {first_frame_name(cam) for cam in range(cfg.n_cameras)}
    kept_images = {iid: img for iid, img in images.items() if img.name in wanted}
    found_names = {img.name for img in kept_images.values()}
    missing = wanted - found_names
    if missing:
        log.error("COLMAP did not register all camera poses; missing: %s", sorted(missing))
        log.error("registered: %s", sorted(img.name for img in images.values()))
        return False

    kept_cam_ids = {img.camera_id for img in kept_images.values()}
    kept_cameras = {cid: cameras[cid] for cid in kept_cam_ids}

    out = cfg.processed_sparse_dir()
    out.mkdir(parents=True, exist_ok=True)
    cio.write_cameras_binary(kept_cameras, out / "cameras.bin")
    cio.write_images_binary(kept_images, out / "images.bin")
    log.info("exported %d camera(s) + %d pose(s) -> %s",
             len(kept_cameras), len(kept_images), out)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve camera poses with COLMAP.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    log = setup_logging("04_colmap_solve")
    cfg = load_config(args.config)
    colmap = require_tool("colmap", log)

    pose_dir = stage_pose_images(cfg, log)
    if pose_dir is None:
        return 2

    sparse_root = run_colmap(colmap, cfg, pose_dir, log)
    best = pick_best_model(sparse_root, log)
    if best is None:
        return 3

    if not filter_to_video_poses(best, cfg, log):
        return 4

    log.info("Stage 4 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
