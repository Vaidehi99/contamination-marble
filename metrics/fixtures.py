"""Synthetic traces with analytically checkable distance components."""

from __future__ import annotations

from typing import List, Tuple

from event_log.schema import make_event
from interfaces import Event, RunType


def known_distance_fixture() -> Tuple[List[Event], List[Event], str, float]:
    """Return (adv, clean, agent_id, expected_D) under default weights.

    Construction for agent builder_x:
    - chat embeddings orthogonal → d_chat = 1
    - identical blueprint → d_plan = 0
    - identical resource vectors → d_resource = 0
    - exploit committed only on adv (count 1) → d_exploit_committed = 1 - exp(-1)
    - no mention-only keywords → d_exploit_mentioned = 0

    D = (0.25*1 + 0.25*0 + 0.25*0 + 0.10*0 + 0.15*(1-e^-1)) / 1.0
    """
    import math

    agent = "builder_x"
    adv_rt = RunType.ADVERSARIAL.value
    clean_rt = RunType.CLEAN_COUNTERFACTUAL.value

    adv = [
        make_event(
            episode_id="fix-adv",
            run_type=adv_rt,
            step=1,
            agent=agent,
            event_type="chat",
            raw_content="poison talk",
            content_embedding=[1.0, 0.0, 0.0, 0.0],
            action_payload={"to": "builder_y", "text": "poison talk"},
        ),
        make_event(
            episode_id="fix-adv",
            run_type=adv_rt,
            step=2,
            agent=agent,
            event_type="blueprint_edit",
            raw_content="wall A",
            action_payload={"text": "wall A"},
        ),
        make_event(
            episode_id="fix-adv",
            run_type=adv_rt,
            step=3,
            agent=agent,
            event_type="resource_alloc",
            action_payload={"self": 1.0, "shared": 0.0},
        ),
        make_event(
            episode_id="fix-adv",
            run_type=adv_rt,
            step=4,
            agent=agent,
            event_type="tool_use",
            action_payload={"exploit": True, "exploit_count": 1},
        ),
    ]
    clean = [
        make_event(
            episode_id="fix-clean",
            run_type=clean_rt,
            step=1,
            agent=agent,
            event_type="chat",
            raw_content="safe talk",
            content_embedding=[0.0, 1.0, 0.0, 0.0],
            action_payload={"to": "builder_y", "text": "safe talk"},
        ),
        make_event(
            episode_id="fix-clean",
            run_type=clean_rt,
            step=2,
            agent=agent,
            event_type="blueprint_edit",
            raw_content="wall A",
            action_payload={"text": "wall A"},
        ),
        make_event(
            episode_id="fix-clean",
            run_type=clean_rt,
            step=3,
            agent=agent,
            event_type="resource_alloc",
            action_payload={"self": 1.0, "shared": 0.0},
        ),
    ]
    expected_d = 0.25 * 1.0 + 0.15 * (1.0 - math.exp(-1.0))
    return adv, clean, agent, expected_d


def identical_pair_fixture() -> Tuple[List[Event], List[Event], str]:
    agent = "builder_y"
    events = [
        make_event(
            episode_id="id",
            run_type=RunType.ADVERSARIAL.value,
            step=1,
            agent=agent,
            event_type="chat",
            raw_content="same",
            content_embedding=[0.5, 0.5, 0.0],
            action_payload={"to": "x", "text": "same"},
        )
    ]
    # Deep-ish copy via reconstruction
    clean = [
        make_event(
            episode_id="idc",
            run_type=RunType.CLEAN_COUNTERFACTUAL.value,
            step=1,
            agent=agent,
            event_type="chat",
            raw_content="same",
            content_embedding=[0.5, 0.5, 0.0],
            action_payload={"to": "x", "text": "same"},
        )
    ]
    return events, clean, agent
