#!/usr/bin/env bash
# DeepPhys + TS-CAN + PhysNet sur UBFC avec ROI Li, Elgendi & Menon (2024).
# Arrêt au premier échec.
set -euo pipefail
set -o pipefail

export RPPG_ROI="${RPPG_ROI:-li2024}"
export UBFC_CACHED_PATH="${UBFC_CACHED_PATH:-/media/simeon/rPPP-Data/Datasets/UBFC-rPPG/Preprocessed_DL}"
export UBFC_DO_PREPROCESS="${UBFC_DO_PREPROCESS:-false}"
export UBFC_LOW_MEMORY="${UBFC_LOW_MEMORY:-1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for script in run_deepphys_li2024.sh run_tscan_li2024.sh run_physnet_li2024.sh; do
  echo ""
  echo "========== $(date -Iseconds) START $script =========="
  bash "scripts/$script"
  echo "========== $(date -Iseconds) DONE $script =========="
done

echo ""
echo "All Li2024 DL models finished."
