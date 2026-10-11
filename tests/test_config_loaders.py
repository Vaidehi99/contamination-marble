"""Phase 0–1 tests: config loaders and pilot matrix size."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loaders import (  # noqa: E402
    build_episode_pair,
    choose_slot_fillers,
    default_injection_target,
    load_org,
    load_pilot_experiment,
    load_poison,
)
from interfaces import RunType, TriggerTiming  # noqa: E402


def test_load_star_org():
    org = load_org("star")
    assert org.topology == "star"
    assert org.adversary_id == "adversary"
    assert len(org.builder_ids) == 3
    assert org.coordinate_mode == "star"
    roles = {a.role for a in org.agents}
    assert roles == {"coordinator", "architect", "builder", "adversary"}


def test_load_graph_org():
    org = load_org("graph")
    assert org.topology == "graph"
    assert len(org.relationships) >= 10


def test_load_poisons_pilot_levels():
    for level, name in [
        (0, "benign"),
        (1, "goal_perturbation"),
        (2, "behavioral_prescription"),
        (3, "capability_exploitation"),
        (4, "recruitment"),
        (5, "autonomous_agenda"),
    ]:
        poison = load_poison(level)
        assert poison.level == level
        assert poison.name == name
        fillers = choose_slot_fillers(poison, seed=1)
        filled = poison.instantiate(fillers)
        for key in fillers:
            assert "{" + key + "}" not in filled
            assert fillers[key] in filled
        assert poison.verify.get("kind") is not None


def test_slot_fillers_deterministic():
    poison = load_poison(2)
    assert choose_slot_fillers(poison, 7) == choose_slot_fillers(poison, 7)
    # Different seeds can differ; at least API is stable
    assert "risky_norm" in choose_slot_fillers(poison, 0)


def test_pilot_matrix_size():
    pilot = load_pilot_experiment("pilot")
    assert pilot.num_episode_pairs() == 180
    cells = pilot.cells()
    assert len(cells) == 3 * 2 * 2  # levels × topologies × triggers
    assert all(len(c.seeds) == 15 for c in cells)


def test_build_episode_pair_counterfactual():
    pilot = load_pilot_experiment("pilot")
    adv, clean = build_episode_pair(
        pilot,
        poison_level=2,
        topology="star",
        seed=3,
        trigger="no_trigger",
    )
    assert adv.pair_id == clean.pair_id
    assert adv.seed == clean.seed == 3
    assert adv.run_type == RunType.ADVERSARIAL
    assert clean.run_type == RunType.CLEAN_COUNTERFACTUAL
    assert adv.poison.level == 2
    assert clean.poison.level == 0
    assert adv.injection.budget_k == 3
    assert adv.injection.target_agent_id == clean.injection.target_agent_id
    assert adv.trigger == TriggerTiming.NONE
    for key, val in adv.slot_fillers.items():
        assert "{" + key + "}" not in adv.instantiated_content()
        assert val in adv.instantiated_content()
    for key in clean.slot_fillers:
        assert "{" + key + "}" not in clean.instantiated_content()
    assert len(adv.instantiated_content()) == len(clean.instantiated_content())
    # Adversary injects into a builder
    org = adv.org
    assert default_injection_target(org) in org.builder_ids
