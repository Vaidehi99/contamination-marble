#!/usr/bin/env bash
# Long-horizon A/B smoke: 12 pairs / 24 trajectories, plus protocol.json and
# horizon_report.{json,md} (paired Y, Delta(c,d), seed-level rows).
# Isolated from the frozen 24-step Luna baseline (results/pilot_llm_${MODEL}/).
#
# Mock (no API key):
#   bash contamination-marble/scripts/run_long_horizon.sh
# Luna:
#   export OPENAI_API_KEY=...
#   LLM_BACKEND=openai bash contamination-marble/scripts/run_long_horizon.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-gpt-5.6-luna}"
LLM_BACKEND="${LLM_BACKEND:-mock}"
EPISODE_BUDGET="${EPISODE_BUDGET:-12}"
SEEDS="${SEEDS:-0 1 2}"
DEPTHS="${DEPTHS:-0 2}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/contamination-marble/results/pilot_llm_${MODEL}_long}"

if [[ "$LLM_BACKEND" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set" >&2
  exit 1
fi

echo "model=$MODEL backend=$LLM_BACKEND budget=$EPISODE_BUDGET"
echo "seeds=$SEEDS depths=$DEPTHS"
echo "results=$RESULTS_ROOT"

PYTHONPATH=contamination-marble python contamination-marble/runner/run_horizon.py \
  --experiment long_horizon \
  --llm-backend "$LLM_BACKEND" \
  --llm-model "$MODEL" \
  --episode-budget "$EPISODE_BUDGET" \
  --seeds $SEEDS \
  --depths $DEPTHS \
  --results-root "$RESULTS_ROOT"
