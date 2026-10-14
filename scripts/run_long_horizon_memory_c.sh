#!/usr/bin/env bash
# Condition C v2 (explicit memory.write every turn) at depth 2 vs frozen B.
# Archived v1 optional-field cell: results/pilot_llm_${MODEL}_long_c/
# Do not run poisoned Luna C until mock benign save→reset→retrieve→use passes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-gpt-5.6-luna}"
LLM_BACKEND="${LLM_BACKEND:-mock}"
EPISODE_BUDGET="${EPISODE_BUDGET:-12}"
SEEDS="${SEEDS:-0 1 2}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/contamination-marble/results/pilot_llm_${MODEL}_long_c_v2}"

if [[ "$LLM_BACKEND" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set" >&2
  exit 1
fi

echo "model=$MODEL backend=$LLM_BACKEND budget=$EPISODE_BUDGET"
echo "C private memory depth=2 seeds=$SEEDS"
echo "results=$RESULTS_ROOT"

PYTHONPATH=contamination-marble python contamination-marble/runner/run_horizon.py \
  --experiment long_horizon_memory_c \
  --llm-backend "$LLM_BACKEND" \
  --llm-model "$MODEL" \
  --episode-budget "$EPISODE_BUDGET" \
  --seeds $SEEDS \
  --results-root "$RESULTS_ROOT"
