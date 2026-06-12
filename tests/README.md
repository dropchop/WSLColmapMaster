# 4cam_pipeline tests

Synthetic-fixture suite for the pre-training pipeline. No real capture footage
and no GPU required — everything runs unattended.

## Run

```bash
conda activate pipeline
cd tests
python -m unittest discover -p 'test_*.py' -v
```

## What's covered

| File | Scope |
|------|-------|
| `test_colmap_io.py` | round-trip of cameras/images/points3D `.bin` |
| `test_extract.py` | naming schemes, date fallback, import_extract (both presets, dry-run, idempotency, per-file error isolation, manifest) |
| `test_validate.py` | Stage 7 passes a good dataset; fails on missing frames, too few points, wrong dims |
| `test_pipeline_e2e.py` | ordered 01→02→03→04→05→06→07 on synthetic data |
| `test_stage05_dryrun.py` | Stage 5 preflight + command construction via `--dry-run` |

## Really executed vs stubbed

**Really executed** (on synthetic ffmpeg clips): Stage 01 ingest, Stage 02
audio cross-correlation sync (recovers the baked-in per-camera offsets), Stage
03 frame extraction, Stage 06 pack, Stage 07 validate. The shared extractor and
COLMAP binary I/O are exercised for real.

**Stubbed** (logged loudly at runtime, no silent skips):
- **Stage 04 / COLMAP** — run against a *fake* `colmap` binary that fabricates
  a faithful sparse model. Stage 04's own logic (image staging, invocation,
  model selection, filtering to the 4 video poses, `.bin` export) is real; only
  feature-matching/SfM is faked. Real CPU COLMAP on `testsrc` patterns wouldn't
  converge, and is unnecessary to test the pipeline plumbing.
- **Stage 05 / MAtCha** — never executed (needs a GPU + real footage). The e2e
  fabricates a `points3D.bin` to stand in for its output; `test_stage05_dryrun`
  separately validates the script's preflight + the exact `train.py` command it
  would run. If the `matcha` env or `~/MAtCha` are absent, that test SKIPS with
  a reported reason.
