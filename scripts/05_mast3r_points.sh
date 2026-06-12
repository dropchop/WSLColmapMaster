#!/usr/bin/env bash
# Stage 5 — dense point-cloud init via MAtCha (MASt3R SfM), in the `matcha` env.
#
# Wraps:
#   python train.py -s <scene-staging> -o <out> --sfm_config posed --sfm_only
# which reads the COLMAP poses from Stage 4 and writes a points3D model. This
# script then copies the resulting points3D.bin into the scene's sparse/0/.
#
# Conda-env switch: stages 00-04/06/07 run in the `pipeline` env; MAtCha needs
# its own `matcha` env. We read config values via the pipeline env, then run
# train.py in the matcha env.
#
# Usage:
#   bash 05_mast3r_points.sh [--config path/to/pipeline.yaml] [--dry-run]
#
# --dry-run validates inputs and prints the exact train.py command WITHOUT
# launching MAtCha (used by the unattended test suite, which has no GPU).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "[05_mast3r] unknown arg: $1" >&2; exit 2 ;;
  esac
done

log() { echo "$(date +%H:%M:%S) [05_mast3r] $*"; }

# --- conda bootstrap ---------------------------------------------------------
if [[ -z "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="${HOME}/miniconda3"
else
  CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# --- read config (in the pipeline env so PyYAML is available) ----------------
CFG_ARG=()
[[ -n "$CONFIG" ]] && CFG_ARG=(--config "$CONFIG")
eval "$(conda run -n pipeline python "${SCRIPT_DIR}/_config_get.py" "${CFG_ARG[@]}" \
        matcha_repo matcha_env processed_dir processed_sparse_dir scene_name)"

log "matcha_repo        = ${matcha_repo}"
log "matcha_env         = ${matcha_env}"
log "scene staging dir  = ${processed_dir}"
log "output sparse/0    = ${processed_sparse_dir}"

# --- preflight ---------------------------------------------------------------
fail=0
if ! conda env list | awk '{print $1}' | grep -qx "${matcha_env}"; then
  log "ERROR: conda env '${matcha_env}' not found"; fail=1
fi
if [[ ! -d "${matcha_repo}" ]]; then
  log "ERROR: MAtCha repo not found at ${matcha_repo}"; fail=1
fi
if [[ ! -f "${matcha_repo}/train.py" ]]; then
  log "ERROR: ${matcha_repo}/train.py missing"; fail=1
fi
if [[ ! -f "${processed_sparse_dir}/images.bin" || ! -f "${processed_sparse_dir}/cameras.bin" ]]; then
  log "ERROR: Stage 4 output (cameras.bin/images.bin) missing in ${processed_sparse_dir}"; fail=1
fi
if [[ "$fail" -ne 0 ]]; then
  log "preflight failed"; exit 3
fi

OUT_DIR="${processed_dir}/mast3r_out"
TRAIN_CMD=(python train.py -s "${processed_dir}" -o "${OUT_DIR}" --sfm_config posed --sfm_only)

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "[dry-run] would run in env '${matcha_env}' (cwd ${matcha_repo}):"
  log "[dry-run]   ${TRAIN_CMD[*]}"
  log "[dry-run] then copy points3D.bin -> ${processed_sparse_dir}/points3D.bin"
  log "[dry-run] STUBBED — MAtCha not executed (no GPU / unattended run)."
  exit 0
fi

# --- run MAtCha --------------------------------------------------------------
mkdir -p "${OUT_DIR}"
log "launching MAtCha in env '${matcha_env}'..."
( cd "${matcha_repo}" && conda run -n "${matcha_env}" "${TRAIN_CMD[@]}" )

# --- collect output ----------------------------------------------------------
# MAtCha's exact output path can vary; search for the produced points3D.bin.
mapfile -t found < <(find "${OUT_DIR}" -name "points3D.bin" -type f 2>/dev/null | sort)
if [[ "${#found[@]}" -eq 0 ]]; then
  log "ERROR: no points3D.bin produced under ${OUT_DIR}"
  log "       inspect that directory and adjust this script's collection step."
  exit 4
fi
src="${found[0]}"
mkdir -p "${processed_sparse_dir}"
cp -f "${src}" "${processed_sparse_dir}/points3D.bin"
log "copied $(basename "$src") (${src}) -> ${processed_sparse_dir}/points3D.bin"
log "Stage 5 complete."
