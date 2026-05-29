#!/usr/bin/env python3
"""Auto-import videos and extract them to PNG sequences.

One tool, selectable naming convention (`--naming`), built on the shared
extraction core in `_extract.py`. Two presets ship today:

  actioncam  (default) — input root holds `ActionCam_<ID>/` folders; every
             video becomes its OWN output folder (named by the source video
             stem) containing `<ID>_<YYYYMMDD>_<frame>.png`.
  cam4c4d              — flat output, `cam{XX}_{YYYY}.png`, assigning camera
             indices to discovered videos in sorted order. (The full 4C4D
             pipeline path with audio-sync seeking is `03_extract_frames.py`,
             which reuses the same core.)

Designed to be run unattended over a freshly-imported batch:
  - non-matching folders are skipped with a warning,
  - each video is extracted independently so one bad file can't sink the batch,
  - re-running is idempotent (skips shots whose PNGs already exist) unless
    --overwrite,
  - --dry-run prints the filenames it *would* produce,
  - a JSON manifest of everything done is written at the end.

Usage:
    python import_extract.py INPUT_ROOT --out OUT_DIR [--naming actioncam|cam4c4d]
        [--fps N] [--max-frames N] [--overwrite] [--dry-run] [--date YYYYMMDD]
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

from _extract import (NAMING_SCHEMES, ShotContext, VIDEO_EXTS, extract_sequence,
                      recording_date)
from _utils import require_tool, setup_logging, write_json

ACTIONCAM_RE = re.compile(r"^ActionCam_(.+)$", re.IGNORECASE)


def find_videos(folder: Path) -> list[Path]:
    """All video files directly inside `folder`, sorted by name."""
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def discover_actioncam(root: Path, log) -> list[tuple[str, Path]]:
    """Return (cam_id, video) pairs from ActionCam_<ID>/ subfolders.

    Folders that don't match the ActionCam_ pattern are skipped with a warning.
    """
    pairs: list[tuple[str, Path]] = []
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not subdirs:
        log.warning("no subfolders under %s", root)
    for d in subdirs:
        m = ACTIONCAM_RE.match(d.name)
        if not m:
            log.warning("skip (not ActionCam_*): %s", d.name)
            continue
        cam_id = m.group(1)
        vids = find_videos(d)
        if not vids:
            log.warning("skip (no videos): %s", d.name)
            continue
        for v in vids:
            pairs.append((cam_id, v))
    return pairs


def discover_cam4c4d(root: Path, log) -> list[tuple[int, Path]]:
    """Return (cam_index, video) pairs, indexing discovered videos in sorted order.

    Looks one level into subfolders if the root has none of its own videos.
    """
    vids = find_videos(root)
    if not vids:
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            vids.extend(find_videos(d))
    vids = sorted(vids)
    return list(enumerate(vids))


def main() -> int:
    parser = argparse.ArgumentParser(description="Import videos -> PNG sequences.")
    parser.add_argument("input_root", type=Path, help="folder of imported video folders")
    parser.add_argument("--out", type=Path, required=True, help="output root directory")
    parser.add_argument("--naming", choices=sorted(NAMING_SCHEMES), default="actioncam",
                        help="naming convention (default: actioncam)")
    parser.add_argument("--fps", type=float, default=None,
                        help="resample to this fps (default: every frame)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="cap frames per video")
    parser.add_argument("--date", default=None,
                        help="override recording date (YYYYMMDD) for actioncam naming")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-extract shots whose PNGs already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="print planned outputs without running ffmpeg")
    args = parser.parse_args()

    log = setup_logging("import_extract")
    if not args.input_root.is_dir():
        log.error("input root not found: %s", args.input_root)
        return 2

    scheme = NAMING_SCHEMES[args.naming]
    ffmpeg = require_tool("ffmpeg", log)
    ffprobe = require_tool("ffprobe", log)

    results: list[dict] = []
    ok = fail = 0

    if args.naming == "actioncam":
        pairs = discover_actioncam(args.input_root, log)
        for cam_id, video in pairs:
            try:
                if args.date:
                    date, src = args.date, "override"
                else:
                    date, src = recording_date(ffprobe, video)
                log.info("=== ActionCam_%s / %s (date %s via %s) ===",
                         cam_id, video.name, date, src)
                ctx = ShotContext(out_root=args.out, cam_id=cam_id, date=date,
                                  shot_stem=video.stem)
                naming = scheme(ctx)
                n = extract_sequence(ffmpeg, video, naming, log, fps=args.fps,
                                     max_frames=args.max_frames,
                                     overwrite=args.overwrite, dry_run=args.dry_run)
                results.append({"source": str(video), "cam_id": cam_id, "date": date,
                                "date_source": src, "out_dir": str(naming.out_dir),
                                "frames": n})
                ok += 1
            except Exception as e:  # one bad video must not sink the batch
                log.error("FAILED on %s: %s", video, e)
                results.append({"source": str(video), "error": str(e)})
                fail += 1
    else:  # cam4c4d
        pairs = discover_cam4c4d(args.input_root, log)
        for cam_index, video in pairs:
            try:
                log.info("=== cam%02d / %s ===", cam_index, video.name)
                ctx = ShotContext(out_root=args.out, cam_index=cam_index)
                naming = scheme(ctx)
                n = extract_sequence(ffmpeg, video, naming, log, fps=args.fps,
                                     max_frames=args.max_frames,
                                     overwrite=args.overwrite, dry_run=args.dry_run)
                results.append({"source": str(video), "cam_index": cam_index,
                                "out_dir": str(naming.out_dir), "frames": n})
                ok += 1
            except Exception as e:
                log.error("FAILED on %s: %s", video, e)
                results.append({"source": str(video), "error": str(e)})
                fail += 1

    if not args.dry_run:
        manifest = {
            "tool": "import_extract",
            "naming": args.naming,
            "input_root": str(args.input_root.resolve()),
            "out_root": str(args.out.resolve()),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "ok": ok, "failed": fail,
            "shots": results,
        }
        manifest_path = args.out / "import_manifest.json"
        write_json(manifest_path, manifest)
        log.info("manifest written: %s", manifest_path)

    log.info("import complete: %d ok, %d failed", ok, fail)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
