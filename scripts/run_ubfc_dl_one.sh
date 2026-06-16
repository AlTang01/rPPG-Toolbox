#!/usr/bin/env bash
# Run a single UBFC DL model (low-memory configs). Example:
#   bash scripts/run_ubfc_dl_one.sh DeepPhys
#   bash scripts/run_ubfc_dl_one.sh EfficientPhys
#   bash scripts/run_ubfc_dl_one.sh Tscan
set -euo pipefail
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export RPPG_ROOT="$ROOT"
# shellcheck source=scripts/rppg_traceability.sh
source "$ROOT/scripts/rppg_traceability.sh"

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 <ModelName>"
  echo "  ModelName: DeepPhys | Tscan | Physnet | EfficientPhys | PhysFormer | RhythmFormer | PhysMamba | FactorizePhys"
  exit 1
fi

export UBFC_DATA_PATH="${UBFC_DATA_PATH:-/media/simeon/rPPP-Data/Datasets/UBFC-rPPG/RawData}"
export UBFC_CACHED_PATH="${UBFC_CACHED_PATH:-/media/simeon/rPPP-Data/Datasets/UBFC-rPPG/Preprocessed_DL}"
export UBFC_LOG_PATH="${UBFC_LOG_PATH:-runs/exp_ubfc_dl}"
export UBFC_LOW_MEMORY="${UBFC_LOW_MEMORY:-1}"
export UBFC_DO_PREPROCESS="${UBFC_DO_PREPROCESS:-false}"
# ROI Zhao 2024 : export RPPG_ROI=zhao2024  (alias de zhao2024_motion_robust)
export RPPG_ROI="${RPPG_ROI:-}"

CFG="configs/train_configs/local/UBFC_DL_${MODEL}.yaml"
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG"
  echo "Run: python scripts/prepare_ubfc_dl_configs.py"
  exit 1
fi

# ---- Run traceability (comparison CSV/JSON) ----
rppg_export_config_file "$CFG"
ROI_LABEL="${RPPG_ROI:-full_face}"
if [[ -z "${RPPG_RUN_NAME:-}" ]]; then
  export RPPG_RUN_NAME="$(rppg_default_run_name "UBFC_DL_${MODEL}_${ROI_LABEL}")"
else
  export RPPG_RUN_NAME="$(rppg_default_run_name "$RPPG_RUN_NAME")"
fi
rppg_print_traceability

source /home/simeon/miniconda3/etc/profile.d/conda.sh
conda activate rppg-toolbox

python scripts/prepare_ubfc_dl_configs.py

if [[ ! -d "$UBFC_DATA_PATH" ]]; then
  echo "ERROR: dataset not found at $UBFC_DATA_PATH"
  exit 1
fi

LOGDIR="runs/logs/dl_one"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/${MODEL}_$(date +%Y%m%d_%H%M%S).log"

echo "Config: $CFG"
echo "Log: $LOG"
if [[ -n "$RPPG_ROI" ]]; then
  echo "ROI: $RPPG_ROI (via RPPG_ROI)"
fi
bash scripts/ensure_ubfc_cache.sh "$CFG"
python main.py --config_file "$CFG" 2>&1 | tee "$LOG"
