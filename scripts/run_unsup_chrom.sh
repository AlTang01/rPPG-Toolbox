#!/usr/bin/env bash
# CHROM non supervisé + ROI. Usage: bash scripts/run_unsup_chrom.sh <roi>
set -euo pipefail

ROI="${1:?Usage: $0 <roi>  (forehead | bondarenko2025 | ...)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export RPPG_ROI="$ROI"
export UBFC_DO_PREPROCESS="${UBFC_DO_PREPROCESS:-false}"

bash scripts/run_with_results.sh "CHROM_${ROI}" configs/infer_configs/UBFC-rPPG_UNSUPERVISED_CHROM.yaml
