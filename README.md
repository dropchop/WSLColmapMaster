# 4-Camera 4D-GS Capture Pipeline

End-to-end data pipeline for replicating the 4C4D capture protocol (Zhou et al., CVPR 2026, arXiv:2604.04063): from four GoPro HERO 12 SD cards to a training-ready dataset for the 4C4D codebase.

**See `RUNBOOK.md` for the full procedure.** This README is just a quickstart.

## Layout

```
4cam_pipeline/
├── RUNBOOK.md                # ⬅ full design document, capture protocol, troubleshooting
├── README.md                 # this file
├── config/
│   └── pipeline.yaml         # single source of truth for paths and params
└── scripts/
    ├── _utils.py             # shared helpers
    ├── 00_check_env.py       # verify external deps
    ├── 01_ingest.py          # SD-card dump → concat takes → manifest
    ├── 02_sync.py            # audio cross-correlation for inter-cam alignment
    ├── 03_extract_frames.py  # ffmpeg → cam{XX}_{YYYY}.png
    ├── 04_colmap_solve.py    # COLMAP poses (reference-stills technique)
    ├── 05_mast3r_points.sh   # MAtCha/MASt3R dense point cloud (bash; conda env switch)
    ├── 06_pack_dataset.py    # final layout + manifest
    └── 07_validate_dataset.py # pre-training sanity checks
```

## Prerequisites

External tools on the workstation that processes the footage:
- Python 3.10+
- `ffmpeg` (with libx265)
- `colmap` (GPU build strongly recommended)
- A conda env named per `config/pipeline.yaml`: `matcha` (for the MAtCha/MASt3R wrapper)
- The 4C4D repo cloned somewhere (for the training side; this pipeline only produces data for it)
- The MAtCha repo cloned at the path set in `config/pipeline.yaml::matcha_repo`

Python deps for this pipeline's own scripts:
```
pip install numpy scipy pyyaml pillow tqdm
```

## Quickstart

1. Edit `config/pipeline.yaml` for your scene (paths, scene name, capture parameters).
2. Copy SD card contents into `raw_inputs/cam00/`, `raw_inputs/cam01/`, etc.
3. Drop reference stills (the 8 roving photos plus the 4 in-position stills per §4.2 of the runbook) into `reference_stills/`.
4. Run stages in order:
   ```
   python scripts/00_check_env.py
   python scripts/01_ingest.py
   python scripts/02_sync.py
   python scripts/03_extract_frames.py
   python scripts/04_colmap_solve.py
   bash   scripts/05_mast3r_points.sh
   python scripts/06_pack_dataset.py
   python scripts/07_validate_dataset.py
   ```
5. If 07 passes, the dataset is at `output_dataset/<dataset_name>/<scene_name>/` and is ready for `4C4D/train.py`.

## Honest disclosure

This pipeline was authored without access to a real 4-GoPro capture or to a working MAtCha install. The stages that interact with external tools (especially stage 5) reflect my best reading of the 4C4D paper, repo README, and standard practice — they have **not been end-to-end tested**. Treat the scripts as a starting framework: expect to iterate on COLMAP camera-model choice, MAtCha input layout, and similar dataset-specific details. Inferred decisions are flagged inline in code comments and in `RUNBOOK.md §9` (Known Gaps).