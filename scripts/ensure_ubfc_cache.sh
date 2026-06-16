#!/usr/bin/env bash
# Ensure preprocessed cache exists for a UBFC DL config; run one-time preprocess if missing.
# Usage: ensure_ubfc_cache.sh <config.yaml>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${1:?config yaml required}"
CFG_ABS="$ROOT/$CFG"
[[ -f "$CFG_ABS" ]] || CFG_ABS="$CFG"
[[ -f "$CFG_ABS" ]] || { echo "ERROR: config not found: $CFG"; exit 1; }

FORCE="${UBFC_DO_PREPROCESS:-false}"
case "${FORCE,,}" in
  1|true|yes) FORCE=1 ;;
  *) FORCE=0 ;;
esac

CACHE_DIR="$(python "$ROOT/scripts/ubfc_train_cache_dir.py" "$CFG_ABS")"

if [[ "$FORCE" -eq 1 ]] || [[ ! -d "$CACHE_DIR" ]]; then
  if [[ "$FORCE" -eq 1 ]]; then
    echo "UBFC_DO_PREPROCESS=true — building cache at:"
  else
    echo "Preprocessed cache missing — one-time preprocess at:"
  fi
  echo "  $CACHE_DIR"

  LOGDIR="$ROOT/runs/logs/dl_one"
  mkdir -p "$LOGDIR"
  PRELOG="$LOGDIR/preprocess_$(basename "$CFG_ABS" .yaml)_$(date +%Y%m%d_%H%M%S).log"

  TMP="$(mktemp --suffix=.yaml)"
  sed -e 's/DO_PREPROCESS: False/DO_PREPROCESS: True/g' \
      -e 's/EPOCHS: 5/EPOCHS: 1/' "$CFG_ABS" > "$TMP"

  echo "Preprocess log: $PRELOG"
  python "$ROOT/main.py" --config_file "$TMP" 2>&1 | tee "$PRELOG"
  rm -f "$TMP"

  if [[ ! -d "$CACHE_DIR" ]]; then
    echo "ERROR: preprocess finished but cache still missing: $CACHE_DIR"
    exit 1
  fi
  echo "Cache ready."
else
  echo "Using existing cache: $CACHE_DIR"
fi
