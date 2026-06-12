#!/usr/bin/env python3
"""Stage 3 — extract synchronized frames from the four ingested videos.

Reads the per-camera lags from `sync.json` (Stage 2) and extracts N_FRAMES
per camera at the configured fps, seeking each camera so all four land on a
common absolute moment. Output is the 4C4D dataloader layout:

    <session>/processed/<scene>/images/cam{XX}_{YYYY}.png

(cam XX = 00..n-1, frame YYYY = 0000..N_FRAMES-1, PNG, lossless).

This stage reuses the shared extractor (`_extract.py`) with the `cam4c4d`
naming scheme. The standalone `import_extract.py` uses the same core for the
`actioncam` convention.

sync.json convention (from Stage 2):
    lags[N].seconds > 0  =>  camN started LATER than the reference cam.
    So we seek camN to  T_ref - lags[N].seconds  in its own local timeline.

Usage:
    python 03_extract_frames.py [--config path] [--start-time SECONDS]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _extract import NAMING_SCHEMES, ShotContext, VIDEO_EXTS, extract_sequence
from _utils import load_config, read_json, require_tool, setup_logging


def find_combined_video(cam_ingest_dir: Path) -> Path:
    """Return the single combined video produced by Stage 1, or fail."""
    candidates = sorted(p for p in cam_ingest_dir.glob("*")
                        if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not candidates:
        raise FileNotFoundError(f"no ingested video in {cam_ingest_dir}")
    if len(candidates) > 1:
        raise ValueError(
            f"expected 1 video in {cam_ingest_dir}, found {len(candidates)}: "
            f"{[c.name for c in candidates]}. Multi-take sessions must be split.")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract synchronized frames.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--start-time", type=float, default=0.0,
                        help="reference-cam start time (s) of the action window")
    args = parser.parse_args()

    log = setup_logging("03_extract_frames")
    cfg = load_config(args.config)
    ffmpeg = require_tool("ffmpeg", log)

    sync_path = cfg.sync_json_path()
    if not sync_path.is_file():
        log.error("sync.json not found at %s — run Stage 2 first", sync_path)
        return 2
    sync = read_json(sync_path)
    lags = sync.get("lags", {})

    scheme = NAMING_SCHEMES["cam4c4d"]
    images_dir = cfg.processed_images_dir()
    images_dir.mkdir(parents=True, exist_ok=True)
    duration = cfg.n_frames / cfg.fps  # seconds of footage we need past the start

    total = 0
    for cam in range(cfg.n_cameras):
        cam_dir = cfg.cam_ingest_dir(cam)
        try:
            video = find_combined_video(cam_dir)
        except (FileNotFoundError, ValueError) as e:
            log.error("cam%02d: %s", cam, e)
            return 2

        lag = float(lags.get(str(cam), {}).get("seconds", 0.0))
        local_start = args.start_time - lag
        if local_start < 0:
            log.warning("cam%02d: computed start %.4fs < 0; clamping to 0 "
                        "(reference cam may need a later --start-time)", cam, local_start)
            local_start = 0.0

        log.info("=== cam%02d: %s @ local start %.4fs (lag %.4fs) ===",
                 cam, video.name, local_start, lag)
        naming = scheme(ShotContext(out_root=images_dir, cam_index=cam))
        try:
            n = extract_sequence(
                ffmpeg, video, naming, log,
                fps=cfg.fps, max_frames=cfg.n_frames,
                start_seconds=local_start, duration_seconds=duration + 1.0,
            )
        except RuntimeError as e:
            log.error("cam%02d: frame extraction failed: %s", cam, e)
            return 3
        if n < cfg.n_frames:
            log.warning("cam%02d: extracted %d frames, expected %d "
                        "(video may be too short past the start time)", cam, n, cfg.n_frames)
        total += n

    log.info("Stage 3 complete: %d frame(s) across %d cameras -> %s",
             total, cfg.n_cameras, images_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
