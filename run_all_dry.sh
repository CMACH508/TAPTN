#!/usr/bin/env bash
# Dry-run all three TAPTN chapter packages (no LLM calls, no training).
# Each package resolves its own sibling *_repro_assets/ directory.
set -euo pipefail
HUB="$(cd "$(dirname "$0")" && pwd)"
export PYTHON="${PYTHON:-python}"

# Do not leak a single TAPTN_ASSETS across packages.
unset TAPTN_ASSETS || true

fail=0
run_one() {
  local pkg="$1"
  echo
  echo "======== $pkg ========"
  if ! (cd "$HUB/$pkg" && bash run_dry.sh); then
    echo "FAILED: $pkg" >&2
    fail=1
  fi
}

run_one TAPTN_rewiring_repro
run_one TAPTN_icl_repro
run_one TAPTN_finetune_repro

if [[ "$fail" -ne 0 ]]; then
  echo "One or more packages failed." >&2
  exit 1
fi
echo
echo "All three dry-runs finished."
