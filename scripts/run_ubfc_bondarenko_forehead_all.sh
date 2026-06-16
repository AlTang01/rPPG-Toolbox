#!/usr/bin/env bash
# 8 runs : TS-CAN / PhysNet / POS / CHROM × bondarenko2025 + forehead
# Arrêt au premier échec.
set -euo pipefail

export UBFC_DO_PREPROCESS="${UBFC_DO_PREPROCESS:-false}"
export UBFC_LOW_MEMORY="${UBFC_LOW_MEMORY:-1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for script in \
  run_tscan_bondarenko2025.sh \
  run_tscan_forehead.sh \
  run_physnet_bondarenko2025.sh \
  run_physnet_forehead.sh; do
  export RPPG_RUN_NAME=""
  echo ""
  echo "========== $(date -Iseconds) $script =========="
  bash "scripts/$script"
done

for script_roi in \
  "run_unsup_pos.sh bondarenko2025" \
  "run_unsup_pos.sh forehead" \
  "run_unsup_chrom.sh bondarenko2025" \
  "run_unsup_chrom.sh forehead"; do
  echo ""
  echo "========== $(date -Iseconds) $script_roi =========="
  # shellcheck disable=SC2086
  bash scripts/$script_roi
done

echo ""
echo "All bondarenko2025 + forehead runs finished."
