#!/usr/bin/env bash
# Suite ROI générales (Selection_ROI.py) : forehead, cheeks, glabella, nose_upper
# Classiques : GREEN, LGI, PBV, OMIT, POS, CHROM
# DL : DeepPhys, Tscan, Physnet
#
# Usage:
#   bash scripts/run_general_rois_suite.sh              # les 4 ROI
#   bash scripts/run_general_rois_suite.sh forehead     # une seule ROI
set -euo pipefail

export UBFC_DATA_PATH="${UBFC_DATA_PATH:-/media/simeon/rPPP-Data/Datasets/UBFC-rPPG/RawData}"
export UBFC_CACHED_PATH="${UBFC_CACHED_PATH:-/media/simeon/rPPP-Data/Datasets/UBFC-rPPG/Preprocessed_DL}"
export UBFC_DO_PREPROCESS="${UBFC_DO_PREPROCESS:-false}"
export UBFC_LOW_MEMORY="${UBFC_LOW_MEMORY:-1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${1:-}" ]]; then
  ROIS=("$1")
else
  ROIS=(forehead cheeks glabella nose_upper)
fi

UNSUP_METHODS=(GREEN LGI PBV OMIT POS CHROM)
DL_MODELS=(DeepPhys Tscan Physnet)

for roi in "${ROIS[@]}"; do
  case "$roi" in
    forehead|cheeks|glabella|nose_upper) ;;
    *)
      echo "ERROR: unknown roi: $roi"
      exit 1
      ;;
  esac

  for m in "${UNSUP_METHODS[@]}"; do
    export RPPG_RUN_NAME=""
    echo ""
    echo "========== Classique $m + $roi =========="
    bash scripts/run_unsup_method.sh "$m" "$roi"
  done

  for dl in "${DL_MODELS[@]}"; do
    export RPPG_RUN_NAME=""
    echo ""
    echo "========== DL $dl + $roi =========="
    bash scripts/run_ubfc_dl_roi.sh "$dl" "$roi"
  done
done

echo ""
echo "Terminé. Voir runs/comparison/results.csv"
