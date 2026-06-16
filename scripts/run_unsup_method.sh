#!/usr/bin/env bash
# Méthode classique non supervisée + ROI.
# Usage: bash scripts/run_unsup_method.sh GREEN bondarenko2025
#        bash scripts/run_unsup_method.sh POS forehead
#        bash scripts/run_unsup_method.sh CHROM nose_upper
# ROI simples : forehead | cheeks | glabella | nose_upper
set -euo pipefail

METHOD="${1:?Usage: $0 <GREEN|LGI|PBV|OMIT|POS|CHROM|ICA> [roi]}"
ROI="${2:-bondarenko2025}"

CFG="configs/infer_configs/UBFC-rPPG_UNSUPERVISED_${METHOD}.yaml"
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG"
  exit 1
fi

export RPPG_ROI="$ROI"
export UBFC_DO_PREPROCESS="${UBFC_DO_PREPROCESS:-false}"

bash scripts/run_with_results.sh "${METHOD}_${ROI}" "$CFG"
