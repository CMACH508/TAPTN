#!/bin/bash
# Dry-run of the TAPTN ICL tables (no LLM calls).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "${TAPTN_ASSETS:-}" ]]; then
  if [[ -d "$ROOT/../TAPTN_icl_repro_assets" ]]; then
    export TAPTN_ASSETS="$ROOT/../TAPTN_icl_repro_assets"
  elif [[ -d "$ROOT/assets" ]]; then
    export TAPTN_ASSETS="$ROOT/assets"
  else
    echo "Unpack the asset bundle and set TAPTN_ASSETS, e.g.:" >&2
    echo "  export TAPTN_ASSETS=/path/to/TAPTN_icl_repro_assets" >&2
    exit 1
  fi
fi
PYTHON="${PYTHON:-python}"
cd "$ROOT"
"$PYTHON" reproduce.py check
"$PYTHON" reproduce.py dry --table all --rescore --figures
