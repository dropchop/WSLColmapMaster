#!/usr/bin/env python3
"""Stage 0 — environment check.

Verifies that external tools and Python packages required by later stages
are actually available. Run this first on any new workstation.

Exits 0 on success, non-zero on the first missing dependency.

Usage:
    python 00_check_env.py [--config path/to/pipeline.yaml]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from _utils import load_config, setup_logging

REQUIRED_CLI_TOOLS = [
    ("ffmpeg", "--version", "ffmpeg with libx265 build"),
    ("ffprobe", "-version", "ffprobe (ships with ffmpeg)"),
    ("colmap", "-h", "COLMAP CLI; GPU build recommended"),
]

REQUIRED_PY_PACKAGES = [
    "numpy",
    "scipy",   # for signal.correlate in stage 2
    "yaml",    # PyYAML import name
    "PIL",     # Pillow
    "tqdm",
]


def check_cli(name: str, flag: str, desc: str, log) -> bool:
    path = shutil.which(name)
    if not path:
        log.error("MISSING: %s (%s)", name, desc)
        return False
    try:
        subprocess.run([name, flag], check=False, capture_output=True, timeout=10)
        log.info("OK     : %s -> %s", name, path)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.error("FOUND BUT FAILED: %s (%s): %s", name, desc, e)
        return False


def check_python_package(name: str, log) -> bool:
    try:
        __import__(name)
        log.info("OK     : python -m %s", name)
        return True
    except ImportError:
        log.error("MISSING: python package %r", name)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check pipeline environment.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    log = setup_logging("00_check_env")
    cfg = load_config(args.config)

    all_ok = True

    log.info("--- CLI tools ---")
    for name, flag, desc in REQUIRED_CLI_TOOLS:
        if not check_cli(name, flag, desc, log):
            all_ok = False

    log.info("--- Python packages (this env only) ---")
    for pkg in REQUIRED_PY_PACKAGES:
        if not check_python_package(pkg, log):
            all_ok = False

    log.info("--- Conda envs (informational, not enforced) ---")
    # We can't easily check arbitrary conda envs from inside Python without conda's
    # init machinery. Instead, just print the expected env names so the operator
    # can verify manually.
    for label, name in [
        ("pipeline (this script)", cfg.raw["conda_env_pipeline"]),
        ("4C4D training", cfg.raw["conda_env_4c4d"]),
        ("MAtCha", cfg.raw["conda_env_matcha"]),
    ]:
        log.info("   expected: %s -> %s", label, name)

    log.info("--- Paths from config ---")
    log.info("   session_dir            = %s", cfg.session_dir)
    log.info("   raw_inputs_dir         = %s (must exist before stage 1)", cfg.raw_inputs_dir)
    log.info("   reference_stills_dir   = %s (must exist before stage 4)", cfg.reference_stills_dir)
    log.info("   output_dataset_dir     = %s (will be created)", cfg.output_dataset_dir)

    log.info("--- Capture parameters ---")
    log.info("   %d cameras at %dx%d @ %d fps, %d frames per take",
             cfg.n_cameras, cfg.width, cfg.height, cfg.fps, cfg.n_frames)

    if all_ok:
        log.info("environment check PASSED")
        return 0
    log.error("environment check FAILED — install missing dependencies before proceeding")
    return 1


if __name__ == "__main__":
    sys.exit(main())