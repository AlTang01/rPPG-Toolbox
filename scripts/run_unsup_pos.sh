#!/usr/bin/env bash
# POS non supervisé + ROI. Usage: bash scripts/run_unsup_pos.sh <roi>
#   bash scripts/run_unsup_pos.sh bondarenko2025
#   bash scripts/run_unsup_pos.sh forehead
set -euo pipefail

ROI="${1:?Usage: $0 <roi>  (forehead | bondarenko2025 | zhao2024 | ...)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export RPPG_ROI="$ROI"
export UBFC_DO_PREPROCESS="${UBFC_DO_PREPROCESS:-false}"

bash scripts/run_with_results.sh "POS_${ROI}" configs/infer_configs/UBFC-rPPG_UNSUPERVISED_POS.yaml
