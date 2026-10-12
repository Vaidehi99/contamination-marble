# Long-Horizon Agent Persistence: Staged Experimental Plan

## Research objective

Determine whether a **single adversarial injection** can cause harmful behavior after the adversary is gone, and identify the carrier responsible: **continuous context, private memory, shared organizational memory, persistent world state, or a later action**.

The full protocol remains the target paper design. Implementation proceeds through explicit stop/go gates rather than launching the full factorial grid immediately.

## Prerequisite: finish the existing short-horizon Luna run

- Let the current **24-step Luna grid** finish and freeze its configuration, outputs, and scoring before modifying the execution loop.
- Treat that run as a **same-episode immediate baseline**. It is **not** the long-horizon depth-0 condition.
- Make the first code change **episode orchestration plus probe gating**, not memory stores, propagation, or consolidation.

## 0. Freeze the protocol

- **One injection:** The adversary injects once and is removed afterward; no re-entry during intervening tasks or probes.
- **Matched control:** Every injected run has a length-matched **L0 twin** with the same benign task and environment setup, differing only in the adversarial content.
- **No label leakage:** Agent-visible task descriptions, artifacts, and tool responses must not reveal condition names, attack labels, or evaluation targets.
- **Probe:** Begin with one standardized **L3** probe; defer L5.
- **Success criterion:** Define the exact observable harmful behavior and scoring rules *before* running experiments. Do not tune thresholds after inspecting results.
- **Independent runs:** Every condition × depth × seed × injection/control combination starts from a fresh baseline. Never probe a trajectory and then continue it to a deeper probe.
- **Persistent organization:** Keep the intended organizational environment across episodes; document precisely which world-state fields persist.
- **Benign intervening episodes:** Turn off the chest fork and scarcity mechanics during intervening work, but retain their intended behavior for the injection and probe stages.
- **Depth semantics:** Depth `d` counts **completed benign episodes between the injection episode and the probe episode**. Depth `0` is `injection episode → probe episode`, with no intervening benign episode. This is distinct from the existing same-episode 24-step Luna baseline.
- **Reset at depth 0:** Condition B resets conversation context at the **injection → probe boundary**, even when there are zero intervening episodes. Condition A retains it.
- **Verifiable benign goals:** Each benign episode has an independently checkable world-state goal (e.g., place `N` blocks at specified coordinates or increase shared storage to `X`). A changed `task_status` string alone is not a new task. Record starting value, target, ending value, and pass/fail.
- **World-state contract:** Before the smoke test, list every field that persists across a context reset, including **agent inventory, shared storage, ledger, placed structures**, and any relevant chest contents, resource counters, world objects, or tool-maintained state. Identify which fields are reset, which persist, and which are intentionally excluded from agent context.
- **Initial seeds:** `{0, 1, 2}` for feasibility only, not confirmatory statistical claims.

**Deliverable:** Versioned protocol, L0/L3 definitions, success rubric, and contamination checklist.

## 1. Build the minimum episode infrastructure

Implement the smallest infrastructure needed to run the A/B comparison.

1. **Episode manager:** Execute an initial injection episode, zero or more benign intervening episodes, and one final probe in a persistent organization.
2. **State separation:** Explicitly distinguish agent conversation/context from persistent world state. Do not add private or shared memory stores yet.
3. **Context-reset switch:** Support continuous context (A) and a clean episode-boundary context reset (B).
4. **Benign task gating:** Disable chest/scarcity during intervening episodes. Implement at least one task with a measurable completion predicate checked against actual world state (not merely seven `task_status` strings over the same 24-step loop). Log start/goal/end values and task completion.
5. **Persistent-state manifest:** At each episode boundary, snapshot inventory, shared storage, ledger, structures, and any other persistent fields; verify the declared persistence/reset contract by test.
6. **Probe gating:** Invoke L3 only at the designated probe stage, with no accidental early trigger.
7. **Observability:** Log complete agent actions/tool results, context boundaries, world-state snapshots, injection removal, and probe outcomes. Use separate output directories per run.
8. **Reset verification:** Audit any hidden caches, tool outputs, summaries, or state that could inadvertently bridge episodes.

**Acceptance gate:** A benign episode completes its verifiable goal with chest/scarcity off; the adversary is absent after injection; depth-0 B resets at the injection/probe boundary; persistent-state fields match the manifest; L3 is identical across conditions; and trajectories and snapshots are reproducible.

**Required pre-smoke checks:**
1. `depth=0` produces exactly **two episodes** (injection, then probe), never the existing 24-step same-episode cell.
2. `condition=B, depth=0` has a context-reset event between those two episodes.
3. `depth=2` produces injection → two distinct, verifiably completed benign episodes → probe.
4. Chest/scarcity are off for *every intervening step*, and available as intended in injection/probe episodes.
5. World-state snapshots show which inventory, ledger, storage, and structure fields survived the reset.
6. No private/shared memory store or forced consolidation LLM call is present.

## 2. A/B persistence smoke test

### Conditions

| Condition | Conversation context | Explicit private/shared memory | World state |
|---|---|---|---|
| **A** | Continuous across episodes | None | Persistent |
| **B** | Reset at episode boundary | None | Persistent |

### Grid

- Conditions: **A, B**
- Probe depths: **0, 2** benign intervening episodes
- Seeds: **0, 1, 2**
- Arms: **one injection and one length-matched L0 twin**
- Probe: **L3**

Total: **2 × 2 × 3 × 2 = 24 independent runs** (12 matched injection/control pairs).

### Exact episode sequences

| Depth | Condition A | Condition B |
|---|---|---|
| `0` | Inject → **retain context** → L3 probe | Inject → **reset context** → L3 probe |
| `2` | Inject → benign goal 1 → benign goal 2 → L3 probe, context retained | Inject → reset → benign goal 1 → reset → benign goal 2 → reset → L3 probe |

For both depths, world state persists according to the declared manifest. The adversary is removed after the initial injection. Every cell also receives its matched L0 twin. The old Luna 24-step grid is reported separately.

### Measurements

- **Primary:** Pre-registered L3 harmful-action outcome, reported separately for injection and L0 control.
- **Secondary:** Legitimate task completion, relevant world-state changes, context retention, and attack-related content visible at the probe.
- **Diagnostics:** Full trajectories, event timelines, context-reset audit, verifiable benign-goal outcomes, and inventory/ledger/storage/structure snapshots at each episode boundary and immediately before L3.

Report raw seed-level results. Three seeds are a smoke test, not a reliable estimate of population-level attack success.

## 3. Stop/go interpretation

| Observation | Interpretation | Next action |
|---|---|---|
| A succeeds at depth 2, B fails | Context continuity is a plausible carrier under this setup | Audit context and consider controlled memory experiments |
| A and B succeed at depth 2 | Reset alone does not eliminate the effect; carrier remains unidentified | Inspect persistent world state, caches, and reset completeness; add a matched state-control diagnostic |
| Both succeed at depth 0 but fail at depth 2 | Immediate effect exists; delayed persistence not yet demonstrated | Check benign-task dilution and probe consistency before expanding |
| Both fail, including depth 0 | Injection or probe may not work in this configuration | Debug immediate attack and scoring first |
| L0 controls show target behavior | Outcome may be baseline behavior rather than injection-caused | **Stop** and repair controls/scoring |
| Episode contamination, label leakage, or mismatched probes | Causal comparison is invalid | **Stop** and repair infrastructure |

**Go to memory mechanisms only when** the episode loop is validated, L0 is clean enough to support attribution, and A/B results pose a concrete question about surviving carriers. A successful B run **does not** prove memory persistence: world state is still persistent.

## 4. Add memory mechanisms C–E (conditional)

After A/B, introduce explicit memory stores one at a time, while preserving the same injection, benign work, L0 twin, L3 probe, and reset policy.

- **Private-only memory:** Can the injected influence survive a context reset through an agent-private store?
- **Shared organizational memory only:** Can it survive through shared records, and can another agent later encounter it?
- **Private + shared memory:** Does their interaction amplify persistence beyond either channel alone?

Specify exact read/write permissions and when memories are updated. Prefer normal workflow memory writes. **Do not add a forced neutral consolidation LLM call** unless the earlier results justify it; such a call creates another contamination path.

Start at depth 2. Include matched no-memory and L0 controls. Where world state can carry attack information, isolate or explicitly control it before making a memory-specific causal claim.

## 5. Expand the horizon (conditional)

After identifying a plausible carrier, evaluate fresh runs at depths **0, 2, 5, and 10**. Never reuse already-probed trajectories. Keep intervening episodes genuinely benign and verify the chest/scarcity gate at every episode.

Increase seeds based on pilot variability and the smallest meaningful effect size. Report uncertainty and failures, not just selected successful trajectories.

## 6. Test propagation and later-action mechanisms

Separate three outcomes:

1. **Persistence:** Attack-relevant information or behavior remains after benign work.
2. **Propagation:** A previously unexposed agent acquires the influence through a shared carrier or changed world state.
3. **Downstream action:** The influenced agent actually performs the predefined harmful behavior at a later opportunity.

Test shared-memory propagation, world-state-mediated propagation, and later-action execution with distinct controls. Do not equate text retention with successful harmful action.

## 7. Remediation, stronger probes, and full protocol

Only after the persistence and carrier results are interpretable, add:

- Carrier deletion/correction and organizational state repair.
- Provenance-aware filtering or other remediation controls.
- Stronger **L5** probe and additional attack conditions, including deferred **F**.
- The broader A–F × depth × seed grid, with sufficient statistical power and explicit safety–utility measurements.

Evaluate whether remediation reduces the harmful outcome **without** unnecessarily impairing legitimate task success.

## Implementation order

| Order | Deliverable | Gate |
|---:|---|---|
| 0 | Finish and freeze current 24-step Luna same-episode baseline | Before code changes |
| 1 | Frozen protocol, depth-0 boundary semantics, L0/L3 scoring, state manifest | Before experiments |
| 2 | Episode manager and meaningful benign episodes | Must pass smoke test |
| 3 | Context reset, state snapshots, probe gating | Must pass reset/probe audit |
| 4 | A/B at depths 0 and 2, three seeds, L0 twins | 24 independent runs |
| 5 | Carrier inspection and state-control diagnostics | Before causal interpretation |
| 6 | Private/shared memory C–E | Only if justified |
| 7 | Depths 5 and 10 with expanded seeds | Only after mechanism validated |
| 8 | Propagation and later-action experiments | After persistence established |
| 9 | Remediation, F, L5, full factorial study | Last |

## Immediate first diff

**Implement the episode loop and L3 probe gating, not the full memory/propagation stack.** The first meaningful result is the paired A/B smoke test with verified adversary removal, genuinely benign intervening episodes, and inspectable persistent world state.
