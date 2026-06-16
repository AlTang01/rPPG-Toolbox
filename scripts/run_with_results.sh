#!/usr/bin/env bash
# Lance une expérience et enregistre un nom de run pour la comparaison.
#
# Usage:
#   bash scripts/run_with_results.sh POS_unsup configs/infer_configs/UBFC-rPPG_UNSUPERVISED.yaml
#   RPPG_ROI=forehead bash scripts/run_with_results.sh POS_forehead configs/infer_configs/UBFC-rPPG_UNSUPERVISED_POS.yaml
#
# Variables optionnelles: RPPG_ROI, RPPG_APPEND_TIMESTAMP=0 pour désactiver le suffixe horaire
set -euo pipefail
set -o pipefail

RUN_NAME_BASE="${1:?Usage: $0 <run_name> <config.yaml>}"
CONFIG="${2:?Usage: $0 <run_name> <config.yaml>}"
shift 2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export RPPG_ROOT="$ROOT"
# shellcheck source=scripts/rppg_traceability.sh
source "$ROOT/scripts/rppg_traceability.sh"

rppg_export_config_file "$CONFIG"
export RPPG_RUN_NAME="$(rppg_default_run_name "$RUN_NAME_BASE")"
rppg_print_traceability

source /home/simeon/miniconda3/etc/profile.d/conda.sh
conda activate rppg-toolbox

LOGDIR="runs/logs/${RPPG_RUN_NAME}"
mkdir -p "$LOGDIR"

echo "Log: $LOGDIR/run.log"

python main.py --config_file "$CONFIG" "$@" 2>&1 | tee "$LOGDIR/run.log"

echo ""
echo "Résumé CSV : runs/comparison/results.csv"
echo "Comparer   : python scripts/compare_runs.py"
