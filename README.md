# Contamination-Marble (Phase 0–1)

Implementation of
`infection_tracing/latent-contamination-marble-experiment-spec.md`.

## Layout

Matches the spec §0 tree under `contamination-marble/`.

Import convention: put this directory on `PYTHONPATH` (hyphenated folder name):

```bash
cd /export/xgen-finance/vaidehi_intern_project
PYTHONPATH=contamination-marble python -m pytest contamination-marble/tests -q
```

## Phase 0 (done)

- Package tree: `marble_ext/`, `event_log/` (spec’s `logging/`; renamed to avoid stdlib clash), `metrics/`, `runner/`, `analysis/`, `results/`
- Frozen interfaces in `interfaces.py` (`OrgConfig`, `PoisonTemplate`, `EpisodeConfig`, `Event`, `MetricBundle`, `PairResult`, …)
- `paths.py` points at sibling `MARBLE/` without forking it

## Phase 1 (done)

- Orgs: `configs/orgs/star.yaml`, `configs/orgs/graph.yaml`
- Poisons: `configs/poisons/level_{0,1,2,4}.yaml`
- Pilot grid: `configs/experiments/pilot.yaml` → **180** episode-pairs
- Loaders: `config_loaders.py` (`load_org`, `load_poison`, `load_pilot_experiment`, `build_episode_pair`)

## Phase 2 (done)

- `event_log/schema.py` — Event helpers + JSONL I/O (§3)
- `event_log/graph.py` — interaction edges + path recovery from detected agent
- `event_log/synthetic.py` — paired synthetic logs with known path
  `adversary → builder_1 → builder_2 → builder_3`

```bash
PYTHONPATH=contamination-marble python -m pytest contamination-marble/tests -q
PYTHONPATH=contamination-marble python contamination-marble/event_log/write_synthetic_demo.py
```

## Phase 3 (done)

- `metrics/behavior_distance.py` — `D` from d_chat / d_plan / d_resource / d_exploit
- `metrics/control.py` — `c_i`, `C`, `C_bar`
- `metrics/spread.py` — `S`, frontier point
- `metrics/zombieness.py` — `Z_i`, `R_Z` (continuation API + pair proxy)
- `metrics/attribution.py` — source/path/latent-carrier/false-attribution
- `metrics/compute.py` — assembles `MetricBundle`
- Unit-tested known `D` fixture + perfect attribution on synthetic pair

## Phase 4 (done) — Mode A

- `marble_ext/registry.py` — agent registry with removal
- `marble_ext/adversary.py` — budget `k`, inject-turn scheduling, forced leave
- `marble_ext/injector.py` — chat / blueprint_edit / strategy_note injection events
- `marble_ext/trigger.py` — no_trigger / +20 / +60
- `marble_ext/defender.py` — read-only observer stub
- `marble_ext/mode_a_sim.py` — scripted org episode (no Mineflayer/LLM)
- `marble_ext/run_mode_a_demo.py` — CLI demo

```bash
PYTHONPATH=contamination-marble python -m pytest contamination-marble/tests -q
PYTHONPATH=contamination-marble python contamination-marble/marble_ext/run_mode_a_demo.py
```

## Phase 5 (done) — seed-matched pair runner

- `runner/seed_match.py` — pre-injection bit-identity + structural turn-order assert
- `runner/run_pair.py` — adv+clean pair → `results/pairs/{digest}/{adv,clean}.jsonl`
  - length-matched level-0 via `content_override`
  - resume-safe (skips complete pairs unless `--force`)
  - metrics only via pairwise `compute_metric_bundle` (rejects unpaired logs)

```bash
PYTHONPATH=contamination-marble python -m pytest contamination-marble/tests -q
PYTHONPATH=contamination-marble python contamination-marble/runner/run_pair.py \
  --poison-level 2 --topology graph --seed 0
```

## Phase 6 (done) — pilot grid + analysis

Still Mode A (scripted). Same pair/metrics APIs will later take real-agent logs.

- `runner/run_grid.py` — sweeps pilot matrix (180 pairs); resume-safe; optional `--workers`
- `analysis/aggregate.py` — Table-1 style Control / Zombieness / Spread by poison family
- `analysis/frontier_plot.py` — `(S, C_bar)` JSON + SVG under `results/pilot/`

```bash
# Smoke subset
PYTHONPATH=contamination-marble python contamination-marble/runner/run_grid.py \
  --poison-levels 2 --topologies graph --triggers no_trigger --seeds 0 1 --aggregate

# Full pilot (180 pairs)
PYTHONPATH=contamination-marble python contamination-marble/runner/run_grid.py --aggregate
```

## LLM Mode A (done) — reasoning agents, scripted adversary

Decision-point interface in `llm/decision.py`. Adversary stays scripted; builder/architect/coordinator decide via LLM (or mock). Idle-skip + prompt-hash completion cache + shared cache across adv/clean for pre-injection identity. `d_chat` can use real embeddings (`llm/embeddings.py`).

```bash
# Offline mock (no API key) — content-sensitive heuristic backend
PYTHONPATH=contamination-marble python contamination-marble/marble_ext/run_mode_a_llm_demo.py

# Real OpenAI decisions
export OPENAI_API_KEY=sk-...
PYTHONPATH=contamination-marble python contamination-marble/runner/run_pair.py \
  --mode llm --llm-backend openai --embed-backend openai \
  --poison-level 4 --topology graph --seed 0 --max-steps 24 --force

# Smoke grid under LLM Mode A (mock)
PYTHONPATH=contamination-marble python contamination-marble/runner/run_grid.py \
  --mode llm --llm-backend mock --max-steps 20 --aggregate \
  --poison-levels 1 2 4 --topologies graph --triggers no_trigger --seeds 0 1 2
```

## Next

1. Re-smoke LLM Mode A (mock, then openai) after scarcity + dual-channel + exploit split
2. If level effects hold, scale to the 180-pair LLM pilot
3. Only then invest in Minecraft Mode B

### Measurement fixes (LLM Mode A)

- **Correctness:** clock-driven scarcity probe (self vs shared) — identical onset for adv/clean
- **Correctness:** `d_exploit_mentioned` vs `d_exploit_committed` (narration ≠ commission)
- **Sensitivity:** dual-field schema `{action, note}` — chat is a side-channel, not a competing action
