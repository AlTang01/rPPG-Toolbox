#!/usr/bin/env bash
# Suite Bondarenko 2025 : GREEN, LGI, PBV, OMIT (classique) + DeepPhys (DL)
set -euo pipefail

export UBFC_DATA_PATH="${UBFC_DATA_PATH:-/media/simeon/rPPP-Data/Datasets/UBFC-rPPG/RawData}"
export UBFC_CACHED_PATH="${UBFC_CACHED_PATH:-/media/simeon/rPPP-Data/Datasets/UBFC-rPPG/Preprocessed_DL}"
export UBFC_DO_PREPROCESS="${UBFC_DO_PREPROCESS:-false}"
export UBFC_LOW_MEMORY="${UBFC_LOW_MEMORY:-1}"
export RPPG_ROI=bondarenko2025

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for m in GREEN LGI PBV OMIT; do
  export RPPG_RUN_NAME=""
  echo ""
  echo "========== Classique $m + bondarenko2025 =========="
  bash scripts/run_unsup_method.sh "$m" bondarenko2025
done

export RPPG_RUN_NAME=""
echo ""
echo "========== DeepPhys + bondarenko2025 =========="
bash scripts/run_deepphys_bondarenko2025.sh

echo ""
echo "Terminé. Voir runs/comparison/results.csv"
