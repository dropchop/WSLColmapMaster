"""Shared frame-extraction core.

A single ffmpeg PNG-sequence extractor with a pluggable, extensible
*naming-scheme registry*. Two stages drive it:

  - `03_extract_frames.py` uses the `cam4c4d` scheme (the 4C4D dataloader's
    flat `cam{XX}_{YYYY}.png` layout).
  - `import_extract.py` (the ActionCam importer) uses the `actioncam` scheme
    (one folder per shot, `<ID>_<YYYYMMDD>_<frame>.png`).

Adding a new convention = register one more entry in NAMING_SCHEMES. Each
scheme is a callable `(ShotContext) -> FrameNaming`; the extraction primitive
is naming-agnostic.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from _utils import run

# Video containers we will try to extract from. Skip GoPro low-res proxies (.lrv)
# and thumbnails (.thm) on purpose.
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg"}


@dataclass
class FrameNaming:
    """Where a single video's PNG sequence lands on disk, and how it's named.

    The final ffmpeg output pattern is `out_dir/{prefix}_%0{digits}d.png`, so the
    first frame is `{prefix}_{start_number:0{digits}d}.png`.
    """
    out_dir: Path
    prefix: str
    digits: int = 6
    start_number: int = 0

    def output_pattern(self) -> Path:
        return self.out_dir / f"{self.prefix}_%0{self.digits}d.png"

    def glob(self) -> str:
        return f"{self.prefix}_*.png"

    def example(self) -> str:
        return f"{self.prefix}_{self.start_number:0{self.digits}d}.png"


@dataclass
class ShotContext:
    """Everything a naming scheme might need to decide a shot's output layout."""
    out_root: Path
    cam_index: int | None = None   # cam4c4d: which physical camera (0..n-1)
    cam_id: str | None = None      # actioncam: the bit after 'ActionCam_'
    date: str | None = None        # actioncam: recording date, YYYYMMDD
    shot_stem: str | None = None   # actioncam: source video filename stem


SchemeFn = Callable[[ShotContext], FrameNaming]


def _scheme_cam4c4d(ctx: ShotContext) -> FrameNaming:
    """Flat images dir, `cam{XX}_{YYYY}.png`, 4-digit frames from 0000.

    This is the exact layout the 4C4D dataloader expects.
    """
    if ctx.cam_index is None:
        raise ValueError("cam4c4d naming requires cam_index")
    return FrameNaming(out_dir=ctx.out_root, prefix=f"cam{ctx.cam_index:02d}",
                       digits=4, start_number=0)


def _scheme_actioncam(ctx: ShotContext) -> FrameNaming:
    """One folder per shot (named by source video stem), `<ID>_<DATE>_<frame>.png`.

    Each shot in its own folder keeps takes from the same camera cleanly separated.
    """
    if not (ctx.cam_id and ctx.date and ctx.shot_stem):
        raise ValueError("actioncam naming requires cam_id, date, and shot_stem")
    return FrameNaming(out_dir=ctx.out_root / ctx.shot_stem,
                       prefix=f"{ctx.cam_id}_{ctx.date}", digits=6, start_number=0)


# The registry. New conventions register here; the CLIs expose its keys.
NAMING_SCHEMES: dict[str, SchemeFn] = {
    "cam4c4d": _scheme_cam4c4d,
    "actioncam": _scheme_actioncam,
}


def probe_creation_date(ffprobe: str, video: Path) -> str | None:
    """Return the video's recording date as YYYYMMDD from container metadata, or None."""
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json",
             "-show_entries", "format_tags=creation_time", str(video)],
            check=False, capture_output=True, text=True, timeout=30,
        )
        tags = json.loads(proc.stdout or "{}").get("format", {}).get("tags", {})
        ct = tags.get("creation_time")
        if ct:
            return datetime.fromisoformat(ct.replace("Z", "+00:00")).strftime("%Y%m%d")
    except Exception:
        pass
    return None


def recording_date(ffprobe: str, video: Path) -> tuple[str, str]:
    """Resolve a recording date with a logged fallback chain.

    Returns (YYYYMMDD, source) where source is 'creation_time' | 'mtime' | 'today'.
    """
    d = probe_creation_date(ffprobe, video)
    if d:
        return d, "creation_time"
    try:
        return datetime.fromtimestamp(video.stat().st_mtime).strftime("%Y%m%d"), "mtime"
    except Exception:
        return datetime.now().strftime("%Y%m%d"), "today"


def extract_sequence(ffmpeg: str, video: Path, naming: FrameNaming, log,
                     *, fps: float | None = None, max_frames: int | None = None,
                     start_seconds: float | None = None, duration_seconds: float | None = None,
                     overwrite: bool = False, dry_run: bool = False) -> int:
    """Extract a PNG sequence from one video according to `naming`.

    Returns the number of frames written (or that would be written, in dry-run).
    Idempotent: if frames already exist and not `overwrite`, the shot is skipped.
    """
    out_pattern = naming.output_pattern()
    existing = sorted(naming.out_dir.glob(naming.glob())) if naming.out_dir.exists() else []

    if dry_run:
        log.info("[dry-run] %s -> %s (e.g. %s)", video.name, naming.out_dir, naming.example())
        return len(existing)

    if existing and not overwrite:
        log.info("skip (exists): %d frame(s) already in %s", len(existing), naming.out_dir)
        return len(existing)

    naming.out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    # -ss before -i is a fast (keyframe-accurate) seek; good enough for shot starts.
    if start_seconds is not None:
        cmd += ["-ss", f"{start_seconds:.6f}"]
    cmd += ["-i", str(video)]
    if duration_seconds is not None:
        cmd += ["-t", f"{duration_seconds:.6f}"]
    # fps filter resamples to the target rate; omit it to take every decoded frame.
    if fps is not None:
        cmd += ["-vf", f"fps={fps}"]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-start_number", str(naming.start_number), str(out_pattern)]

    run(cmd, log, capture_output=True)
    written = sorted(naming.out_dir.glob(naming.glob()))
    log.info("   wrote %d frame(s) -> %s", len(written), naming.out_dir)
    return len(written)
