#!/usr/bin/env bash
# LLM Mode A toy grid (scripted adversary + LLM org agents).
# Pair ids do not include the model, so RESULTS_ROOT must stay unique per model
# or an earlier run will be treated as already done / overwritten.
#
# Usage:
#   export OPENAI_API_KEY=...
#   bash contamination-marble/scripts/run_llm_levels.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# --- knobs ---
MODEL="${MODEL:-gpt-5.6-luna}"
LEVELS="${LEVELS:-1 2 3 4 5}"
TOPOLOGIES="${TOPOLOGIES:-graph}"
TRIGGERS="${TRIGGERS:-no_trigger}"
SEEDS="${SEEDS:-0 1 2}"
MAX_STEPS="${MAX_STEPS:-24}"
WORKERS="${WORKERS:-1}"
EMBED_BACKEND="${EMBED_BACKEND:-openai}"
# Isolated from results/pilot_llm (gpt-4o-mini).
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/contamination-marble/results/pilot_llm_${MODEL}}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set" >&2
  exit 1
fi

echo "model=$MODEL"
echo "levels=$LEVELS topologies=$TOPOLOGIES triggers=$TRIGGERS seeds=$SEEDS"
echo "results=$RESULTS_ROOT"
echo "max_steps=$MAX_STEPS workers=$WORKERS"

PYTHONPATH=contamination-marble python contamination-marble/runner/run_grid.py \
  --mode llm \
  --llm-backend openai \
  --llm-model "$MODEL" \
  --embed-backend "$EMBED_BACKEND" \
  --poison-levels $LEVELS \
  --topologies $TOPOLOGIES \
  --triggers $TRIGGERS \
  --seeds $SEEDS \
  --max-steps "$MAX_STEPS" \
  --workers "$WORKERS" \
  --results-root "$RESULTS_ROOT" \
  --aggregate
