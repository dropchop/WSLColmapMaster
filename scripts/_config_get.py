#!/usr/bin/env python3
"""Print selected pipeline.yaml values as shell-eval `KEY=VALUE` lines.

Lets bash stages (e.g. 05_mast3r_points.sh) read the single source of truth
without parsing YAML themselves:

    eval "$(python _config_get.py --config pipeline.yaml matcha_repo scene_name)"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _utils import REPO_ROOT, load_config


def main() -> int:
    p = argparse.ArgumentParser(description="Emit config values for shell eval.")
    p.add_argument("--config", default=None)
    p.add_argument("keys", nargs="+")
    args = p.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)

    matcha_repo = Path(cfg.raw["matcha_repo"])
    if not matcha_repo.is_absolute():
        matcha_repo = (REPO_ROOT / matcha_repo).resolve()

    mapping = {
        "matcha_repo": str(matcha_repo),
        "matcha_env": str(cfg.raw.get("conda_env_matcha", "matcha")),
        "processed_dir": str(cfg.processed_dir()),
        "processed_images_dir": str(cfg.processed_images_dir()),
        "processed_sparse_dir": str(cfg.processed_sparse_dir()),
        "scene_name": str(cfg.scene_name),
        "dataset_name": str(cfg.dataset_name),
        "n_sparse": str(cfg.raw.get("n_sparse", cfg.n_cameras)),
    }
    for k in args.keys:
        if k not in mapping:
            sys.stderr.write(f"unknown config key: {k}\n")
            return 2
        # shell-safe single-quote
        print(f"{k}='{mapping[k]}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
