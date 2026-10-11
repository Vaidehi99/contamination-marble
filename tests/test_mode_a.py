"""Phase 4 tests: adversary budget, injection, registry removal, Mode A sim."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loaders import build_episode_pair, load_org, load_pilot_experiment  # noqa: E402
from event_log.graph import build_interaction_graph, recover_path_from_detected  # noqa: E402
from event_log.schema import injection_events, read_jsonl  # noqa: E402
from interfaces import Channel  # noqa: E402
from marble_ext.adversary import AdversaryController  # noqa: E402
from marble_ext.injector import Injector  # noqa: E402
from marble_ext.mode_a_sim import run_mode_a_episode  # noqa: E402
from marble_ext.registry import AgentRegistry  # noqa: E402
from marble_ext.trigger import TriggerSchedule  # noqa: E402
from interfaces import TriggerTiming  # noqa: E402


def test_adversary_budget_removes_from_registry():
    org = load_org("graph")
    registry = AgentRegistry.from_org(org)
    adv_id = org.adversary_id
    assert adv_id
    ctrl = AdversaryController(agent_id=adv_id, budget_k=3, inject_turns=[3])
    assert ctrl.should_inject_on_next_turn() is False
    assert ctrl.consume_turn(registry) == 1
    assert ctrl.should_inject_on_next_turn() is False
    assert ctrl.consume_turn(registry) == 2
    assert ctrl.should_inject_on_next_turn() is True
    assert ctrl.consume_turn(registry) == 3
    assert ctrl.removed is True
    assert not registry.is_active(adv_id)
    assert adv_id in registry.removed


def test_injector_chat_marks_injection_event():
    pilot = load_pilot_experiment("pilot")
    adv, _ = build_episode_pair(
        pilot, poison_level=2, topology="star", seed=1, trigger="no_trigger"
    )
    inj = Injector(adv)
    record = inj.emit(step=5, source="adversary", turn_index=3)
    assert record.event.is_injection_event is True
    assert record.event.event_type == "chat"
    assert record.event.action_payload["to"] == adv.injection.target_agent_id
    assert record.event.poison_level == 2
    text = record.event.raw_content or ""
    for key in adv.slot_fillers:
        assert "{" + key + "}" not in text


def test_injector_blueprint_channel():
    pilot = load_pilot_experiment("pilot")
    adv, _ = build_episode_pair(
        pilot, poison_level=1, topology="graph", seed=2, trigger="no_trigger"
    )
    adv.injection.channel = Channel.BLUEPRINT_EDIT.value
    record = Injector(adv).emit(step=3, source="adversary", turn_index=1)
    assert record.event.event_type == "blueprint_edit"
    assert record.event.is_injection_event


def test_trigger_plus_20_emits_at_base_plus_20():
    sched = TriggerSchedule(timing=TriggerTiming.PLUS_20, base_step=3)
    assert sched.trigger_step == 23
    ev = sched.maybe_emit(episode_id="e", run_type="adversarial", step=23)
    assert ev is not None
    assert ev.action_payload["trigger"] == "trigger_plus_20"
    assert sched.maybe_emit(episode_id="e", run_type="adversarial", step=22) is None


def test_mode_a_episode_injects_then_removes_adversary(tmp_path: Path):
    pilot = load_pilot_experiment("pilot")
    adv, clean = build_episode_pair(
        pilot, poison_level=2, topology="graph", seed=0, trigger="trigger_plus_20"
    )
    result = run_mode_a_episode(
        adv,
        enable_contagion=True,
        out_path=tmp_path / "adv.jsonl",
    )
    assert result.injection is not None
    assert result.injection.event.is_injection_event
    assert result.adversary_removed
    assert adv.org.adversary_id not in result.active_agents_final
    # No post-removal adversary events
    post = [
        e
        for e in result.events
        if e.agent == adv.org.adversary_id and e.step > result.injection.event.step
    ]
    # adversary may have only injection on last turn; ensure none after removal
    assert all(e.step <= result.injection.event.step for e in result.events if e.agent == adv.org.adversary_id)

    inj = injection_events(result.events)
    assert len(inj) == 1
    assert inj[0].content_hash == result.poison_hash

    loaded = read_jsonl(tmp_path / "adv.jsonl")
    assert len(loaded) == len(result.events)

    # Clean pair also removes adversary and uses level-0 content
    clean_result = run_mode_a_episode(clean, enable_contagion=False)
    assert clean_result.adversary_removed
    assert clean.injection.poison_level == 0
    assert clean_result.poison_hash != result.poison_hash


def test_mode_a_contagion_path_recoverable():
    pilot = load_pilot_experiment("pilot")
    adv, _ = build_episode_pair(
        pilot, poison_level=4, topology="graph", seed=3, trigger="no_trigger"
    )
    result = run_mode_a_episode(adv, enable_contagion=True, max_contagion_forwards=3)
    graph = build_interaction_graph(result.events)
    assert result.injection is not None
    detected = result.carriers[-1] if result.carriers else result.injection.target
    path = recover_path_from_detected(
        graph, detected, prefer_hashes={result.poison_hash}
    )
    assert path[0] == "adversary"
    assert result.injection.target in path
