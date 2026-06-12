# 4-Camera 4D-GS Capture Pipeline — Runbook

End-to-end procedure for going from four GoPro HERO 12 SD cards to a `data/<scene>/` directory that the 4C4D training code can ingest. This runbook is built to replicate **4C4D: 4 Camera 4D Gaussian Splatting** (Zhou et al., arXiv:2604.04063, CVPR 2026), and stops at the training boundary.

**Conventions used in this document.**
- *(paper §X)* — directly stated in the 4C4D paper
- *(repo)* — taken from the official code repository `yangzf-1023/4C4D`
- *(inferred)* — my judgment, not stated by the paper or repo; flagged so you can override

---

## 1. What this pipeline does and doesn't

**Does:**
- Captures 4 synchronized GoPro video streams plus reference stills for pose stability
- Sync-aligns the 4 streams in post (audio cross-correlation)
- Extracts frames in the exact naming convention 4C4D's dataloader expects *(repo)*
- Solves camera extrinsics with COLMAP using the reference-stills technique *(paper §A.2)*
- Initializes a dense point cloud with MASt3R via the MAtCha wrapper *(repo)*
- Assembles the `data/<scene>/{images,sparse/0}/` layout *(repo)*
- Validates the structure before handoff to training

**Does not:**
- Run 4C4D training. That is a separate concern owned by the 4C4D repo; this pipeline produces the data that repo consumes.
- Provide held-out test views. The paper notes that on their Dyn4Cam dataset *(paper §4.1)* "Dyn4Cam does not include held-out test views, we only provide qualitative comparisons" — meaning a 4-cam-only capture is for novel-view rendering at training-camera-adjacent trajectories, not for quantitative test-view evaluation. If you want quantitative evaluation, you need additional cameras at held-out positions.
- Color-grade, denoise, or stabilize footage. The paper does none of this.
- Handle moving cameras. The capture protocol assumes the 4 cameras are stationary during recording *(paper §A.2: "We position four cameras facing the region where the dynamic actions occur")*.

---

## 2. Codebase lineage — important note

The 4C4D repo's Acknowledgements section says the codebase builds on **"4DGS"** linking to `fudan-zvg.github.io/4d-gaussian-splatting/`. That is the Fudan 4DGS paper, which is **a different work** from the Wu et al. *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering* (arXiv:2310.08528) that uses a deformation field plus HexPlane encoder.

In 4C4D's own results tables, the two are distinguished:
- `4DGaussians [44]` = Wu et al. deformation-field approach (your project paper)
- `4DGS [47]` = Fudan native-4D-primitive approach (4C4D's parent codebase)

Both are baselines that 4C4D outperforms. If you intended to build on top of Wu et al.'s codebase rather than the Fudan one, this pipeline's data format (COLMAP-style `sparse/0/`) is still useful — Wu et al. uses a Neu3D-style format that is closely related — but you'd need to adapt the training-side glue yourself.

---

## 3. Hardware

Replicating *(paper §A.1)*:

| Item | Spec |
|---|---|
| Cameras | 4× GoPro HERO 12 (Black) |
| Resolution | 1920×1080 |
| Frame rate | 60 fps |
| Settings | Identical across all 4 cameras |
| Mounting | Custom brackets; rig + cameras < $1,500 USD total |

*(inferred)* recommendations the paper doesn't specify:
- **Lens mode:** Use **Linear** on all four. Wide/SuperView/HyperView apply different non-pinhole digital corrections; Linear gets you closest to a rectilinear projection COLMAP can handle. Verify it's the *same* setting on all four bodies.
- **Exposure / WB:** Lock both. Auto-exposure drift between cameras kills SfM feature matching and color consistency.
- **Codec & bitrate:** HEVC at the highest available bitrate. Don't enable HyperSmooth (digital stabilization changes the effective intrinsics frame-to-frame).
- **Audio:** Keep audio recording ON. We use it for sync.
- **GPS:** Off (saves battery, irrelevant here).
- **SD cards:** UHS-I V30 minimum; UHS-II V60 preferred for 1080p60 HEVC sustained writes.
- **Power:** Externally power all four cameras if possible. HERO 12 has known thermal throttling at sustained 1080p60.

*(paper §A.2)* on capture geometry: "The four viewpoints span an effective coverage of approximately 100–120 degrees, providing sufficient overlap to fully observe the scene without causing the severe difficulties associated with non-overlapping views." Interpretation *(inferred)*: spread the four cameras around the scene so adjacent cameras share substantial scene overlap; ~25–40° between adjacent cameras is a reasonable read of "100–120° total span across 4 cameras." Verify on a test shot before committing to a long session.

---

## 4. Capture protocol

### 4.1 Pre-shoot
1. Mount all 4 cameras rigidly. Movement during the session will invalidate the COLMAP poses.
2. Match all 4 settings (resolution, fps, lens mode, exposure, WB, audio).
3. Verify time-of-day is set identically on all four (helps with rough manual sync if audio-sync fails).
4. Battery > 80% on all four; SD cards formatted in-camera.

### 4.2 Reference-stills phase *(paper §A.2)*
> "Estimating stable poses from only four photos can often be unreliable. Therefore, we capture an additional set of eight images of the same scene to improve multi-view feature matching and enhance pose stability."

Procedure:
1. With cameras mounted in their final video positions, capture **one still per camera** in photo mode.
2. **Without moving the rig**, ask an assistant to walk around the scene with **one** of the four cameras temporarily removed from the rig, and take 8 additional stills from different positions and angles. Replace the camera in the rig when done.
3. *(inferred — paper is ambiguous here)* The paper says "the same four cameras are used to take photos from additional viewpoints." It's unclear whether this means 8 stills *per camera* removed-and-replaced, or 8 stills *total* using any of the four cameras as a roving camera. Reading the text literally, I read it as: 8 additional viewpoints total, captured using one or more of the four cameras temporarily relocated. The key invariant is that COLMAP gets ≥12 photos of the scene with diverse viewpoints, and the final 4 positions are recoverable. The 4 in-rig positions must be among the photos.

### 4.3 Video capture
1. Sync marker at start: a sharp audio event (clap, slate) visible/audible to all four cameras simultaneously.
2. Start all cameras. The paper doesn't specify a method; the GoPro Quik app can start multiple cameras via the network feature, but in practice press-each-camera works fine if you do a sync marker.
3. Record the dynamic action. Paper uses 300 frames at 60 fps = **5-second clips** *(paper §4.1)*.
4. Sync marker at end (optional but useful for validation).
5. Stop all cameras.

### 4.4 Shot log
For each take, record on paper or a tablet:
- Take ID, scene name, action category
- Start/stop timestamp
- Any anomalies (someone walked through frame, camera shifted, etc.)

---

## 5. Software stack

Install once on the workstation that will process the footage.

| Tool | Purpose | Notes |
|---|---|---|
| Python 3.10+ | Pipeline scripts | Conda recommended for env isolation |
| ffmpeg | Video decode, audio extraction, frame extraction | Build with libx265 |
| COLMAP | Camera pose estimation | GPU-accelerated build strongly recommended |
| 4C4D repo | Training-side code | `git clone github.com/yangzf-1023/4C4D` |
| MAtCha (MASt3R wrapper) | Dense point cloud init | `git clone github.com/anttwo/MAtCha`; `python install.py`; `python download_checkpoints.py` |
| `numpy`, `scipy`, `pyyaml`, `tqdm`, `pillow` | Pipeline scripts | See `requirements.txt` |

Two conda environments per the 4C4D README:
- `4c4d` (or `4dgs`) — 4C4D training repo env
- `matcha` — MAtCha env

This pipeline's scripts use a third lightweight env (`pipeline`) for ingest/sync/extract/pack — no GPU needed.

---

## 6. Pipeline stages

Each stage has a script under `scripts/`. The single source of truth for paths and parameters is `config/pipeline.yaml`. All scripts read it.

### Stage 0 — `00_check_env.py`
Sanity-checks that ffmpeg, COLMAP, and the conda envs exist before you waste an hour discovering they don't. Non-destructive.

### Stage 1 — `01_ingest.py`
Reads from `raw_inputs/cam{XX}/` (where you've already copied SD card contents) and produces `<session>/ingest/cam{XX}/`:
- Copies all `.MP4`/`.LRV`/`.THM` to the session folder
- SHA-256 manifest for integrity
- Detects chaptered files (`GH01XXXX.MP4`, `GH02XXXX.MP4`, ...) — GoPros split files at ~4 GB; this script concatenates them losslessly via `ffmpeg -c copy`

*(inferred)* I'm assuming you copy from SD card to disk manually before running this. Auto-detecting mounted SD cards reliably is platform-specific and brittle.

### Stage 2 — `02_sync.py`
Audio-based sync alignment.
1. Extract audio from each of the 4 streams as mono 48kHz WAV.
2. Cross-correlate audio from cam01..cam03 against cam00 (designated reference).
3. Output: per-camera offset in milliseconds, written to `<session>/sync.json`.
4. Optional verification: also detect the sharpest peak in the first 10 seconds (should be your clap) and confirm offsets are consistent.

**Sync accuracy expectations** *(inferred)*: audio cross-correlation typically nails alignment to ±1 audio sample = ±~21µs (at 48kHz). At 60fps a frame is 16.7ms, so this is sub-frame and adequate for 4D-GS. The paper says "perfect synchronization" without describing how; my read is they're claiming frame-level alignment which audio xcorr clears easily.

**Limitations:** rolling shutter within each frame means even "perfectly synced" GoPros still scan different parts of a moving subject at slightly different times. The paper does not address this and neither does this pipeline.

### Stage 3 — `03_extract_frames.py`
1. Read `sync.json` offsets.
2. For each cam, seek to the offset-adjusted start time.
3. Extract `N_FRAMES` (default 300, matching paper §A.1 and §4.1) at exactly 60 fps.
4. Write to `<session>/processed/<scene>/images/cam{XX}_{YYYY}.png` *(repo's exact naming)*.
5. Camera index `XX` is `00..03`. Frame index `YYYY` is `0000..0299` (zero-padded).
6. PNG, not JPEG. The repo's example data uses PNG and lossless is preferable for SfM features.

### Stage 4 — `04_colmap_solve.py`
Wraps COLMAP CLI:
1. Build an `images_for_pose/` directory containing the reference stills (Stage 4.2 from this runbook) and the first frame of each video.
2. `colmap feature_extractor` with the OPENCV camera model (4 distortion params).
3. `colmap exhaustive_matcher`.
4. `colmap mapper` to produce a sparse reconstruction.
5. Filter the reconstruction to only the 4 video-source poses (one per cam).
6. Export `cameras.bin`, `images.bin` to the scene's `sparse/0/`.

*(inferred)* Camera model choice: OPENCV (4 distortion coeffs) for Linear-mode GoPros. If you instead used Wide/HyperView, OPENCV_FISHEYE is closer. Paper does not specify the COLMAP camera model.

### Stage 5 — `05_mast3r_points.sh`
Wraps the MAtCha invocation per *(repo)*:
```
python train.py \
  -s data/<scene>/mast3r_4 \
  -o data/<scene>/mast3r_4 \
  --sfm_config posed --sfm_only
```
Inputs: the COLMAP-derived poses from Stage 4, plus the first frame of each camera. Output: `points3D.bin` written into the scene's `sparse/0/`.

*(repo)* MAtCha is what 4C4D actually invokes; the underlying engine is MASt3R but you call it through MAtCha's wrapper with `--sfm_config posed` (skip MAtCha's own pose solve, use ours) and `--sfm_only` (don't go on to do mesh reconstruction).

### Stage 6 — `06_pack_dataset.py`
Final assembly into the layout 4C4D's dataloader expects *(repo)*:
```
data/<dataset_name>/<scene>/
├── images/
│   ├── cam00_0000.png
│   ├── cam00_0001.png
│   ├── ...
│   └── cam03_0299.png
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

This stage copies/moves the staged files into final position and writes a `pipeline_manifest.json` for traceability.

### Stage 7 — `07_validate_dataset.py`
Pre-training sanity checks:
- All `4 × N_FRAMES` images exist with correct naming
- `cameras.bin` has exactly 4 cameras (or 1 if `SINGLE_CAMERA=true` — same model)
- `images.bin` has exactly 4 poses, matching expected cam indices
- `points3D.bin` exists and has ≥1000 points *(inferred threshold)*
- All four image resolutions match `cameras.bin` intrinsics

---

## 7. Training handoff

Once `07_validate_dataset.py` passes, the directory is ready for 4C4D. Per the repo:
```
python train.py \
  --config <path-to-4c4d-config> \
  --training_view 0,1,2,3 \
  --output_dir output/<scene>
```
Note `--training_view 0,1,2,3` reflects our zero-indexed `cam00..cam03` convention. Adjust if you use a different indexing.

This pipeline does not run the training command itself. You'll run it in the 4C4D conda env.

---

## 8. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `02_sync.py` reports offsets > 200ms variance | One camera started/stopped much earlier | Re-check shot log; trim source videos to common window manually |
| `04_colmap_solve.py` mapper produces <4 registered images | Too few reference photos, or non-overlapping views | Re-shoot reference stills with more overlap; verify Linear lens mode |
| MAtCha OOMs | Too many images or too high resolution | Reduce input image resolution or count |
| `07_validate_dataset.py` reports image dim ≠ intrinsic dim | Frame extraction at different size than COLMAP solve | Ensure both stages use the same source resolution (don't downsample one) |
| Hot GoPros dropping framerate mid-take | Thermal throttling | Use external power, ventilate, shorter takes |

---

## 9. Known gaps and open questions

These are honest places where the paper, the repo, or this pipeline leaves you on your own:

1. **Exact lens mode used in Dyn4Cam.** Paper says "GoPro HERO 12 ... 1920×1080 ... 60fps" but doesn't specify Linear vs Wide. Linear is my recommendation but other modes may work if COLMAP camera model is chosen accordingly.
2. **Reference-still capture procedure.** The "8 additional photos" detail in §A.2 is described in one sentence. My reading is documented in §4.2 of this runbook; you may need to iterate if COLMAP doesn't converge.
3. **Sync method.** Paper says "multi-view frame re-alignment" without specifics. Audio xcorr is my choice; GoPro Labs timecode is a more rigorous alternative if you want to go further.
4. **Scene scale / units.** COLMAP outputs poses in arbitrary scale. The 4C4D training code presumably handles this internally, but if you intersect with a measured environment, you'll need a known-scale reference object in the scene.
5. **Color consistency across cameras.** Not addressed by 4C4D. If you see color seams in renders, the fix is at capture time (manual WB, identical settings) — patching in post is hard.

---

## 10. References

- 4C4D paper: arXiv:2604.04063v1 (Zhou et al., CVPR 2026)
- 4C4D code: https://github.com/yangzf-1023/4C4D
- MAtCha (MASt3R wrapper): https://github.com/anttwo/MAtCha
- COLMAP: https://colmap.github.io/
- Wu et al. 4D-GS (project-file paper, related but distinct from 4C4D's parent codebase): arXiv:2310.08528v3