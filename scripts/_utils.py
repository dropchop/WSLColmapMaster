"""Shared utilities for the 4C4D capture pipeline.

Loads pipeline.yaml and provides path/logging helpers used by every stage.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "pipeline.yaml"


@dataclass
class Config:
    """Typed view onto pipeline.yaml. Keep field names matching the YAML."""
    raw: dict[str, Any]

    @property
    def session_dir(self) -> Path:
        return Path(self.raw["session_dir"]).resolve()

    @property
    def raw_inputs_dir(self) -> Path:
        return Path(self.raw["raw_inputs_dir"]).resolve()

    @property
    def reference_stills_dir(self) -> Path:
        return Path(self.raw["reference_stills_dir"]).resolve()

    @property
    def output_dataset_dir(self) -> Path:
        return Path(self.raw["output_dataset_dir"]).resolve()

    @property
    def dataset_name(self) -> str:
        return self.raw["dataset_name"]

    @property
    def scene_name(self) -> str:
        return self.raw["scene_name"]

    @property
    def n_cameras(self) -> int:
        return int(self.raw["n_cameras"])

    @property
    def n_frames(self) -> int:
        return int(self.raw["n_frames"])

    @property
    def fps(self) -> int:
        return int(self.raw["fps"])

    @property
    def width(self) -> int:
        return int(self.raw["resolution"]["width"])

    @property
    def height(self) -> int:
        return int(self.raw["resolution"]["height"])

    # --- Derived paths ---
    def ingest_dir(self) -> Path:
        return self.session_dir / "ingest"

    def cam_ingest_dir(self, cam_idx: int) -> Path:
        return self.ingest_dir() / f"cam{cam_idx:02d}"

    def sync_json_path(self) -> Path:
        return self.session_dir / "sync.json"

    def scene_output_dir(self) -> Path:
        return self.output_dataset_dir / self.dataset_name / self.scene_name

    def scene_images_dir(self) -> Path:
        return self.scene_output_dir() / "images"

    def scene_sparse_dir(self) -> Path:
        return self.scene_output_dir() / "sparse" / "0"

    def colmap_workdir(self) -> Path:
        return self.session_dir / "colmap"

    # --- Staging paths (intermediate artifacts before the final pack in stage 6) ---
    def processed_dir(self) -> Path:
        return self.session_dir / "processed" / self.scene_name

    def processed_images_dir(self) -> Path:
        return self.processed_dir() / "images"

    def processed_sparse_dir(self) -> Path:
        return self.processed_dir() / "sparse" / "0"

    def frame_name(self, cam_idx: int, frame_idx: int) -> str:
        """4C4D dataloader frame name: cam{XX}_{YYYY}.png (cam 00-, frame 0000-)."""
        return f"cam{cam_idx:02d}_{frame_idx:04d}.png"


def load_config(path: Path | None = None) -> Config:
    """Load pipeline.yaml. If `path` is None, use the default location."""
    cfg_path = path or DEFAULT_CONFIG
    with open(cfg_path) as f:
        return Config(raw=yaml.safe_load(f))


def setup_logging(name: str) -> logging.Logger:
    """Stage scripts call this. Single-line, timestamped, INFO-level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 of a file. Streams via chunks; safe for multi-GB videos."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], log: logging.Logger, check: bool = True,
        capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run an external command with logging. Logs the command and exit code."""
    log.info("$ %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=capture_output, text=True)
    if check and result.returncode != 0:
        if capture_output:
            log.error("stderr: %s", result.stderr)
        log.error("command failed with exit code %d", result.returncode)
        sys.exit(result.returncode)
    return result


def require_tool(name: str, log: logging.Logger) -> str:
    """Resolve a tool on PATH; fail loudly if missing."""
    found = shutil.which(name)
    if not found:
        log.error("required tool %r not found on PATH", name)
        sys.exit(2)
    return found


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)