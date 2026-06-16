#!/usr/bin/env bash
# PhysNet sur UBFC avec ROI Li, Elgendi & Menon (2024) — li2024_extended.
set -euo pipefail
set -o pipefail

export RPPG_ROI="${RPPG_ROI:-li2024}"
export UBFC_CACHED_PATH="${UBFC_CACHED_PATH:-/media/simeon/rPPP-Data/Datasets/UBFC-rPPG/Preprocessed_DL}"
export UBFC_DO_PREPROCESS="${UBFC_DO_PREPROCESS:-false}"
export UBFC_LOW_MEMORY="${UBFC_LOW_MEMORY:-1}"

exec bash "$(dirname "$0")/run_ubfc_dl_one.sh" Physnet_Li2024
