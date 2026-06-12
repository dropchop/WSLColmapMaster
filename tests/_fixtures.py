"""Shared test fixtures for the 4cam_pipeline suite.

Everything here is synthetic so the suite runs unattended with no real capture
footage and no GPU:
  - tiny ffmpeg-generated 4-cam clips with a shared (delayed) noise track so
    audio cross-correlation in Stage 2 has something real to lock onto,
  - a fake `colmap` executable that fabricates a faithful sparse model, so
    Stage 4's real plumbing (staging, invocation, model selection, filtering)
    is exercised without GPU/feature-matching,
  - fabricated COLMAP .bin models standing in for the MAtCha (Stage 5) output.

Anything stubbed is logged loudly by the tests that use it.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import _colmap_io as cio  # noqa: E402  (after sys.path tweak)

# Small, fast synthetic capture parameters.
W, H, FPS, N_FRAMES, N_CAMS = 160, 120, 10, 6, 4
CLIP_SECONDS = 3
# Per-camera audio offsets (seconds) baked into the synthetic clips.
OFFSETS = {0: 0.0, 1: 0.10, 2: 0.20, 3: 0.05}


def load_stage(filename: str):
    """Import a (possibly numerically-named) stage module from scripts/."""
    path = SCRIPTS / filename
    mod_name = "stage_" + filename.replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_config(tmp: Path, **overrides) -> Path:
    """Write a pipeline.yaml into tmp and return its path."""
    cfg = {
        "session_dir": str(tmp / "session_001"),
        "raw_inputs_dir": str(tmp / "raw_inputs"),
        "reference_stills_dir": str(tmp / "reference_stills"),
        "output_dataset_dir": str(tmp / "output_dataset"),
        "dataset_name": "dyn4cam_test",
        "scene_name": "take_test",
        "n_cameras": N_CAMS,
        "resolution": {"width": W, "height": H},
        "fps": FPS,
        "n_frames": N_FRAMES,
        "sync_reference_cam": 0,
        "sync_audio_sample_rate": 48000,
        "sync_max_offset_ms": 3000,
        "colmap_camera_model": "OPENCV",
        "colmap_single_camera": False,
        "matcha_repo": "../MAtCha",
        "n_sparse": 4,
        "conda_env_pipeline": "pipeline",
        "conda_env_4c4d": "4c4d",
        "conda_env_matcha": "matcha",
        "validate": {"min_points3d": 1000, "max_image_size_mb": 10},
    }
    cfg.update(overrides)
    p = tmp / "pipeline.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
                   check=True)


def make_master_noise(tmp: Path) -> Path:
    """One pink-noise WAV reused (delayed) across cameras so they correlate."""
    wav = tmp / "master_noise.wav"
    _ffmpeg(["-f", "lavfi", "-i", f"anoisesrc=d={CLIP_SECONDS}:c=pink:r=48000", str(wav)])
    return wav


def make_raw_inputs(tmp: Path) -> Path:
    """Build raw_inputs/cam0X/GH010001.MP4 with offset-shared audio."""
    master = make_master_noise(tmp)
    raw = tmp / "raw_inputs"
    for cam in range(N_CAMS):
        d = raw / f"cam{cam:02d}"
        d.mkdir(parents=True, exist_ok=True)
        delay_ms = int(OFFSETS[cam] * 1000)
        _ffmpeg([
            "-f", "lavfi", "-i", f"testsrc=d={CLIP_SECONDS}:s={W}x{H}:rate={FPS}",
            "-i", str(master),
            "-af", f"adelay={delay_ms}:all=1",
            "-t", str(CLIP_SECONDS), "-shortest", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(d / "GH010001.MP4"),
        ])
    return raw


def make_reference_stills(tmp: Path, n: int = 3) -> Path:
    """A few correctly-sized stills for the COLMAP pose solve."""
    d = tmp / "reference_stills"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        _ffmpeg(["-f", "lavfi", "-i", f"testsrc=d=0.1:s={W}x{H}:rate=1",
                 "-frames:v", "1", str(d / f"ref_{i:02d}.png")])
    return d


def write_fake_colmap(bin_dir: Path) -> Path:
    """Create a `colmap` executable that fabricates a faithful sparse model.

    feature_extractor / exhaustive_matcher are no-ops; mapper writes a model
    under <output_path>/0 with one OPENCV camera per image (sized to the PNG)
    and an identity-ish pose, named by the source PNG filename.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "colmap"
    # Pin the shebang to the running interpreter (not PATH's `python3`) so the
    # fake binary always has the same packages (e.g. PIL) as the test process.
    script.write_text(f'''#!{sys.executable}
import sys
from pathlib import Path
sys.path.insert(0, {str(SCRIPTS)!r})
import _colmap_io as cio
from PIL import Image as PILImage

argv = sys.argv[1:]
cmd = argv[0] if argv else ""
flags = {{}}
i = 1
while i < len(argv):
    if argv[i].startswith("--"):
        flags[argv[i]] = argv[i + 1] if i + 1 < len(argv) else ""
        i += 2
    else:
        i += 1

if cmd in ("feature_extractor", "exhaustive_matcher"):
    sys.exit(0)

if cmd == "mapper":
    image_path = Path(flags["--image_path"])
    out = Path(flags["--output_path"]) / "0"
    out.mkdir(parents=True, exist_ok=True)
    pngs = sorted(p for p in image_path.iterdir() if p.suffix.lower() == ".png")
    cameras, images = {{}}, {{}}
    for idx, png in enumerate(pngs, start=1):
        with PILImage.open(png) as im:
            w, h = im.size
        cameras[idx] = cio.Camera(idx, "OPENCV", w, h,
                                  [float(w), float(w), w / 2.0, h / 2.0, 0.0, 0.0, 0.0, 0.0])
        images[idx] = cio.Image(idx, (1.0, 0.0, 0.0, 0.0), (float(idx), 0.0, 0.0),
                                idx, png.name, [], [])
    cio.write_cameras_binary(cameras, out / "cameras.bin")
    cio.write_images_binary(images, out / "images.bin")
    cio.write_points3D_binary({{}}, out / "points3D.bin")
    sys.exit(0)

sys.exit(0)
''')
    script.chmod(0o755)
    return script


def fabricate_sparse(sparse_dir: Path, *, n_cameras: int = N_CAMS,
                     width: int = W, height: int = H, n_points: int = 1200) -> None:
    """Write a complete sparse/0 model (stands in for Stage 4 + 5 output)."""
    sparse_dir.mkdir(parents=True, exist_ok=True)
    cameras = {c + 1: cio.Camera(c + 1, "OPENCV", width, height,
                                 [float(width), float(width), width / 2.0, height / 2.0,
                                  0.0, 0.0, 0.0, 0.0]) for c in range(n_cameras)}
    images = {c + 1: cio.Image(c + 1, (1.0, 0.0, 0.0, 0.0), (float(c), 0.0, 0.0),
                               c + 1, f"cam{c:02d}_0000.png", [], [])
              for c in range(n_cameras)}
    points = {k: cio.Point3D(k, (float(k), 0.0, 0.0), (200, 100, 50), 0.4, [1], [0])
              for k in range(1, n_points + 1)}
    cio.write_cameras_binary(cameras, sparse_dir / "cameras.bin")
    cio.write_images_binary(images, sparse_dir / "images.bin")
    cio.write_points3D_binary(points, sparse_dir / "points3D.bin")


def fabricate_points3D(sparse_dir: Path, n_points: int = 1200) -> None:
    """Stand in for the MAtCha (Stage 5) points3D.bin output."""
    points = {k: cio.Point3D(k, (float(k), 1.0, 2.0), (10, 20, 30), 0.5, [1], [0])
              for k in range(1, n_points + 1)}
    cio.write_points3D_binary(points, sparse_dir / "points3D.bin")


def run_stage(script: str, cfg_path: Path, *extra, extra_path: Path | None = None):
    """Run a stage script as a subprocess in the current (pipeline) env."""
    env = dict(os.environ)
    if extra_path:
        env["PATH"] = f"{extra_path}{os.pathsep}{env['PATH']}"
    cmd = [sys.executable, str(SCRIPTS / script), "--config", str(cfg_path), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, env=env)
