#!/usr/bin/env bash
# Helpers for run traceability (source from other scripts).
# Usage: source "$(dirname "$0")/rppg_traceability.sh"

rppg_default_run_name() {
  # $1 = base name (e.g. POS_forehead or UBFC_DL_Tscan_Zhao2024)
  local base="${1:?base run name required}"
  if [[ "${RPPG_APPEND_TIMESTAMP:-1}" == "1" ]]; then
    if [[ "$base" =~ _[0-9]{8}_[0-9]{6}$ ]]; then
      echo "$base"
    else
      echo "${base}_$(date +%Y%m%d_%H%M%S)"
    fi
  else
    echo "$base"
  fi
}

rppg_export_config_file() {
  # $1 = config path (relative or absolute)
  local cfg="${1:?config path required}"
  local root="${RPPG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  if [[ "$cfg" != /* ]]; then
    cfg="$root/$cfg"
  fi
  export RPPG_CONFIG_FILE="$cfg"
}

rppg_print_traceability() {
  echo "Run traceability:"
  echo "  RPPG_RUN_NAME=${RPPG_RUN_NAME:-}"
  echo "  RPPG_CONFIG_FILE=${RPPG_CONFIG_FILE:-}"
  echo "  RPPG_ROI=${RPPG_ROI:-full_face}"
}
