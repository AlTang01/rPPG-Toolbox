#!/usr/bin/env bash
# TS-CAN + ROI Bondarenko 2025 (bondarenko2025_top5).
set -euo pipefail
export RPPG_ROI="${RPPG_ROI:-bondarenko2025}"
export UBFC_DO_PREPROCESS="${UBFC_DO_PREPROCESS:-false}"
export UBFC_LOW_MEMORY="${UBFC_LOW_MEMORY:-1}"
exec bash "$(dirname "$0")/run_ubfc_dl_one.sh" Tscan_Bondarenko2025
