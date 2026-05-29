#!/usr/bin/env python3
"""Stage 1 — ingest from raw SD card dumps.

Expects you to have already copied SD card contents to raw_inputs/cam{XX}/.
This stage:
  1. Hashes every source file (SHA-256) and writes manifest.
  2. Concatenates GoPro chaptered files (GH01/GH02/...) losslessly with
     ffmpeg's concat demuxer.
  3. Produces ingest/cam{XX}/cam{XX}.mp4 — one combined file per camera.

GoPro chaptering: HERO 12 splits long recordings at ~4 GB boundaries into
GH01XXXX.MP4, GH02XXXX.MP4, etc. They share the same XXXX suffix per take.
We group by that suffix.

Usage:
    python 01_ingest.py [--config path/to/pipeline.yaml]
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from _utils import load_config, require_tool, run, setup_logging, sha256_file, write_json

# GoPro chaptered file pattern: GH<chapter><take>.MP4
# e.g. GH010001.MP4, GH020001.MP4 belong to take 0001, in chapter order.
CHAPTER_RE = re.compile(r"^GH(\d{2})(\d{4})\.MP4$", re.IGNORECASE)


def group_chapters(cam_dir: Path) -> dict[str, list[Path]]:
    """Group MP4 files by take ID, ordered by chapter number."""
    by_take: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for p in cam_dir.iterdir():
        if not p.is_file():
            continue
        m = CHAPTER_RE.match(p.name)
        if not m:
            continue
        chapter = int(m.group(1))
        take = m.group(2)
        by_take[take].append((chapter, p))
    return {take: [p for _, p in sorted(items)] for take, items in by_take.items()}


def concat_chapters(chapters: list[Path], out_path: Path, ffmpeg: str, log) -> None:
    """Lossless concat using ffmpeg concat demuxer. -c copy = no re-encode."""
    if len(chapters) == 1:
        # Single chapter — just copy bits over (still using ffmpeg so the
        # container is rewritten cleanly without leftover SD-card metadata).
        run([ffmpeg, "-y", "-i", str(chapters[0]), "-c", "copy", str(out_path)], log)
        return

    # Multi-chapter: write a temporary concat list file.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        for p in chapters:
            # ffmpeg concat demuxer requires `file 'path'` with escaped quotes.
            tf.write(f"file '{p.as_posix()}'\n")
        list_path = Path(tf.name)
    try:
        run([
            ffmpeg, "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(out_path),
        ], log)
    finally:
        list_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest raw GoPro footage.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-checksum", action="store_true",
                        help="Skip SHA-256 (faster, but no integrity record)")
    args = parser.parse_args()

    log = setup_logging("01_ingest")
    cfg = load_config(args.config)
    ffmpeg = require_tool("ffmpeg", log)

    if not cfg.raw_inputs_dir.is_dir():
        log.error("raw_inputs_dir not found: %s", cfg.raw_inputs_dir)
        return 2

    cfg.ingest_dir().mkdir(parents=True, exist_ok=True)
    manifest: dict = {"cameras": {}}

    for cam_idx in range(cfg.n_cameras):
        cam_src = cfg.raw_inputs_dir / f"cam{cam_idx:02d}"
        cam_dst = cfg.cam_ingest_dir(cam_idx)
        cam_dst.mkdir(parents=True, exist_ok=True)

        if not cam_src.is_dir():
            log.error("missing camera input dir: %s", cam_src)
            return 2

        log.info("=== cam%02d ===", cam_idx)
        chapters_by_take = group_chapters(cam_src)
        if not chapters_by_take:
            log.error("no GH*.MP4 files found in %s", cam_src)
            return 2

        if len(chapters_by_take) > 1:
            log.warning("multiple takes detected in %s: %s",
                        cam_src, sorted(chapters_by_take.keys()))
            log.warning("ingesting all takes; the pipeline assumes ONE take per session.")
            log.warning("if you have multiple takes, split them into separate sessions.")

        cam_record: dict = {"takes": {}}
        for take, chapters in sorted(chapters_by_take.items()):
            log.info("take %s — %d chapter(s)", take, len(chapters))

            # Hash sources
            chapter_records = []
            for c in chapters:
                size = c.stat().st_size
                digest = "skipped" if args.skip_checksum else sha256_file(c)
                chapter_records.append({
                    "filename": c.name,
                    "size_bytes": size,
                    "sha256": digest,
                })
                log.info("   %s  %.1f MB  %s", c.name, size / 1e6, digest[:12])

            out_path = cam_dst / f"cam{cam_idx:02d}_take{take}.mp4"
            concat_chapters(chapters, out_path, ffmpeg, log)
            log.info("   -> %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
            cam_record["takes"][take] = {
                "chapters": chapter_records,
                "combined_file": str(out_path.relative_to(cfg.session_dir)),
                "combined_size_bytes": out_path.stat().st_size,
            }

        manifest["cameras"][f"cam{cam_idx:02d}"] = cam_record

    manifest_path = cfg.session_dir / "ingest_manifest.json"
    write_json(manifest_path, manifest)
    log.info("manifest written: %s", manifest_path)
    log.info("Stage 1 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())