#!/usr/bin/env bash
# DL UBFC + ROI simple (forehead, cheeks, glabella, nose_upper).
# Usage: bash scripts/run_ubfc_dl_roi.sh DeepPhys forehead
#        bash scripts/run_ubfc_dl_roi.sh Tscan nose_upper
set -euo pipefail

MODEL_BASE="${1:?Usage: $0 <DeepPhys|Tscan|Physnet> <forehead|cheeks|glabella|nose_upper>}"
ROI_KEY="${2:?Usage: $0 <DeepPhys|Tscan|Physnet> <forehead|cheeks|glabella|nose_upper>}"

case "$MODEL_BASE" in
  DeepPhys|Tscan|Physnet) ;;
  *)
    echo "ERROR: model must be DeepPhys, Tscan, or Physnet (got: $MODEL_BASE)"
    exit 1
    ;;
esac

case "$ROI_KEY" in
  forehead|cheeks|glabella|nose_upper) ;;
  *)
    echo "ERROR: roi must be forehead, cheeks, glabella, or nose_upper (got: $ROI_KEY)"
    exit 1
    ;;
esac

ROI_SLUG="$(python3 - <<PY
key = "${ROI_KEY}"
print("_".join(p[:1].upper() + p[1:] for p in key.split("_")))
PY
)"

CFG="configs/train_configs/local/UBFC_DL_${MODEL_BASE}_${ROI_SLUG}.yaml"
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG"
  echo "Run: python scripts/generate_ubfc_dl_roi_configs.py"
  exit 1
fi

export RPPG_ROI="$ROI_KEY"
export UBFC_DO_PREPROCESS="${UBFC_DO_PREPROCESS:-false}"
export UBFC_LOW_MEMORY="${UBFC_LOW_MEMORY:-1}"
exec bash "$(dirname "$0")/run_ubfc_dl_one.sh" "${MODEL_BASE}_${ROI_SLUG}"
