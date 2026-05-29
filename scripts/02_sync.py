#!/usr/bin/env python3
"""Stage 2 — sync alignment by audio cross-correlation.

For each camera, extracts the first N seconds of audio as mono PCM, then
cross-correlates camN audio against the reference camera's audio to find
the inter-camera lag.

Output: <session>/sync.json

The convention used here:
  lag_seconds[N] = how much LATER camN started recording relative to cam{ref},
  in seconds. Positive value means camN started after cam{ref}.

Stage 3 uses these values to pick frame-extraction start times that align
the four cameras to a common absolute time.

The paper §A.2 calls this "multi-view frame re-alignment" but doesn't specify
a method. Cross-correlation on the recorded audio is a standard choice and
typically aligns sub-sample (≈21µs at 48 kHz), well under one frame at 60 fps.

Usage:
    python 02_sync.py [--config path/to/pipeline.yaml] [--window-seconds 30]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import correlate

from _utils import load_config, require_tool, run, setup_logging, write_json


def extract_audio_mono(video_path: Path, out_wav: Path, sample_rate: int,
                       duration_seconds: float, ffmpeg: str, log) -> None:
    """Extract the first `duration_seconds` of audio as 16-bit PCM mono WAV.

    -ac 1   : downmix to mono (GoPro records stereo; we only need mono)
    -ar SR  : resample to target rate
    -t SEC  : limit duration
    -vn     : no video
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run([
        ffmpeg, "-y",
        "-i", str(video_path),
        "-t", str(duration_seconds),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        str(out_wav),
    ], log, capture_output=True)


def load_wav_int16(wav_path: Path) -> tuple[np.ndarray, int]:
    """Minimal WAV reader for our PCM s16le mono file.

    Avoids a dependency on scipy.io.wavfile. Reads bytes, parses the canonical
    WAV header just enough to find the data chunk.
    """
    data = wav_path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"not a RIFF/WAVE file: {wav_path}")

    # Walk subchunks to find 'fmt ' and 'data'
    sample_rate = None
    pcm_data = None
    pos = 12
    while pos < len(data) - 8:
        chunk_id = data[pos:pos+4]
        chunk_size = int.from_bytes(data[pos+4:pos+8], "little")
        chunk_body = data[pos+8:pos+8+chunk_size]
        if chunk_id == b"fmt ":
            sample_rate = int.from_bytes(chunk_body[4:8], "little")
        elif chunk_id == b"data":
            pcm_data = chunk_body
            break
        pos += 8 + chunk_size + (chunk_size & 1)  # chunks pad to even
    if sample_rate is None or pcm_data is None:
        raise ValueError(f"could not find fmt/data chunks in {wav_path}")

    samples = np.frombuffer(pcm_data, dtype="<i2").astype(np.float32)
    return samples, sample_rate


def find_lag_samples(ref: np.ndarray, target: np.ndarray,
                     max_offset_samples: int) -> tuple[int, float, float]:
    """Cross-correlate. Return (lag, peak_value, prominence).

    lag > 0 means target's content appears delayed relative to ref —
    equivalently, target's camera started LATER in absolute time.

    Prominence = peak_value / mean(|correlation|), a crude confidence number.
    Higher is better; 10+ is a clear peak, 2-3 is marginal.
    """
    # Normalize to zero mean to avoid DC bias in correlation.
    ref = ref - ref.mean()
    target = target - target.mean()

    c = correlate(ref, target, mode="full", method="fft")
    center = len(target) - 1  # index of zero lag

    # Restrict argmax search to a bounded window around zero lag.
    lo = max(0, center - max_offset_samples)
    hi = min(len(c), center + max_offset_samples + 1)
    search_region = c[lo:hi]

    local_peak_idx = int(np.argmax(np.abs(search_region)))
    peak_idx = lo + local_peak_idx
    lag = peak_idx - center

    peak_value = float(c[peak_idx])
    mean_abs = float(np.mean(np.abs(search_region)) + 1e-9)
    prominence = abs(peak_value) / mean_abs
    return lag, peak_value, prominence


def find_ingest_video(cam_ingest_dir: Path) -> Path:
    """Return the single combined video file from stage 1, or fail."""
    candidates = sorted(cam_ingest_dir.glob("*.mp4"))
    if not candidates:
        raise FileNotFoundError(f"no .mp4 found in {cam_ingest_dir}")
    if len(candidates) > 1:
        raise ValueError(
            f"expected 1 video in {cam_ingest_dir}, found {len(candidates)}: "
            f"{[c.name for c in candidates]}. "
            f"Multi-take sessions need to be split."
        )
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync cameras via audio xcorr.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--window-seconds", type=float, default=30.0,
                        help="Seconds of audio from the start to use for sync.")
    args = parser.parse_args()

    log = setup_logging("02_sync")
    cfg = load_config(args.config)
    ffmpeg = require_tool("ffmpeg", log)

    ref_idx = int(cfg.raw["sync_reference_cam"])
    sr = int(cfg.raw["sync_audio_sample_rate"])
    max_offset_ms = float(cfg.raw["sync_max_offset_ms"])
    max_offset_samples = int(max_offset_ms * sr / 1000)

    audio_dir = cfg.session_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: extract a window of audio from each cam.
    audio_arrays: dict[int, np.ndarray] = {}
    for cam in range(cfg.n_cameras):
        video = find_ingest_video(cfg.cam_ingest_dir(cam))
        wav = audio_dir / f"cam{cam:02d}.wav"
        log.info("cam%02d: extracting %.1fs of audio from %s", cam, args.window_seconds, video.name)
        extract_audio_mono(video, wav, sr, args.window_seconds, ffmpeg, log)

        samples, file_sr = load_wav_int16(wav)
        if file_sr != sr:
            log.error("cam%02d wav sample rate is %d, expected %d", cam, file_sr, sr)
            return 2
        audio_arrays[cam] = samples
        log.info("cam%02d: %d samples loaded", cam, len(samples))

    # Step 2: cross-correlate.
    ref_audio = audio_arrays[ref_idx]
    lags: dict[str, dict] = {}

    for cam in range(cfg.n_cameras):
        if cam == ref_idx:
            lags[str(cam)] = {
                "samples": 0,
                "seconds": 0.0,
                "peak_value": float(np.dot(ref_audio, ref_audio)),
                "prominence": float("inf"),
                "is_reference": True,
            }
            continue
        target = audio_arrays[cam]
        lag, peak, prom = find_lag_samples(ref_audio, target, max_offset_samples)
        lag_seconds = lag / sr
        lags[str(cam)] = {
            "samples": int(lag),
            "seconds": float(lag_seconds),
            "peak_value": float(peak),
            "prominence": float(prom),
            "is_reference": False,
        }
        log.info(
            "cam%02d: lag = %+d samples = %+.4fs (prominence %.1f)",
            cam, lag, lag_seconds, prom,
        )
        if prom < 5.0:
            log.warning(
                "cam%02d: prominence is low (%.1f). The sync may be unreliable. "
                "Inspect the audio manually (e.g. in Audacity) and consider "
                "supplying a manual offset.", cam, prom
            )

    # Step 3: write sync.json.
    payload = {
        "reference_cam": ref_idx,
        "sample_rate": sr,
        "max_offset_searched_ms": max_offset_ms,
        "window_seconds": args.window_seconds,
        "lags": lags,
        "convention": (
            "lags.N.seconds > 0 means camN started LATER than the reference cam. "
            "Stage 3 extracts frames starting at "
            "T_ref - lags.N.seconds within camN's local timeline."
        ),
    }
    write_json(cfg.sync_json_path(), payload)
    log.info("wrote %s", cfg.sync_json_path())

    # Show worst-case span so the operator sees how much margin Stage 3 will need.
    all_seconds = [v["seconds"] for v in lags.values()]
    log.info(
        "sync span: min=%+.3fs, max=%+.3fs, range=%.3fs",
        min(all_seconds), max(all_seconds), max(all_seconds) - min(all_seconds),
    )
    log.info("Stage 2 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())