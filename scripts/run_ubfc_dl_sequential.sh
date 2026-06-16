#!/usr/bin/env bash
# Run multiple UBFC DL models one after another. Stops on first failure.
#
#   bash scripts/run_ubfc_dl_sequential.sh DeepPhys EfficientPhys Tscan
#   bash scripts/run_ubfc_dl_sequential.sh --all
#
# Recommended on ~2 GB GPU: run models individually via run_ubfc_dl_one.sh
set -euo pipefail
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export UBFC_DATA_PATH="${UBFC_DATA_PATH:-/media/simeon/rPPP-Data/Datasets/UBFC-rPPG/RawData}"
export UBFC_CACHED_PATH="${UBFC_CACHED_PATH:-/media/simeon/rPPP-Data/Datasets/UBFC-rPPG/Preprocessed_DL}"
export UBFC_LOG_PATH="${UBFC_LOG_PATH:-runs/exp_ubfc_dl}"
export UBFC_LOW_MEMORY="${UBFC_LOW_MEMORY:-1}"
export UBFC_DO_PREPROCESS="${UBFC_DO_PREPROCESS:-false}"

CONFIG_DIR="configs/train_configs/local"
LIGHT_MODELS=(DeepPhys EfficientPhys Tscan Physnet)

source /home/simeon/miniconda3/etc/profile.d/conda.sh
conda activate rppg-toolbox

python scripts/prepare_ubfc_dl_configs.py

if [[ ! -d "$UBFC_DATA_PATH" ]]; then
  echo "ERROR: dataset not found at $UBFC_DATA_PATH"
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 --all | --light | <ModelName> [ModelName ...]"
  echo "  --light  : DeepPhys, EfficientPhys, Tscan, Physnet (smaller VRAM)"
  echo "  --all    : all 8 models (heavy ones use CPU by default)"
  exit 1
fi

CONFIGS=()
if [[ "$1" == "--all" ]]; then
  mapfile -t CONFIGS < <(find "$CONFIG_DIR" -name 'UBFC_DL_*.yaml' | sort)
elif [[ "$1" == "--light" ]]; then
  for m in "${LIGHT_MODELS[@]}"; do
    CONFIGS+=("$CONFIG_DIR/UBFC_DL_${m}.yaml")
  done
else
  for m in "$@"; do
    CONFIGS+=("$CONFIG_DIR/UBFC_DL_${m}.yaml")
  done
fi

LOGDIR="runs/logs/dl_sequential_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
echo "Logs: $LOGDIR"
echo "Stop on first failure (set -e)"

for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "$cfg" ]]; then
    echo "ERROR: missing config $cfg"
    exit 1
  fi
  name="$(basename "$cfg" .yaml)"
  log="$LOGDIR/${name}.log"
  echo ""
  echo "========== $(date -Iseconds) START $name =========="
  python main.py --config_file "$cfg" 2>&1 | tee "$log"
  echo "========== $(date -Iseconds) DONE $name =========="
done

echo ""
echo "All requested models finished. Summary in $LOGDIR"
