"""Assemble MetricBundle for a paired episode (§4)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set

from interfaces import Event, MetricBundle
from metrics.attribution import AttributionResult, attribute_from_events
from metrics.behavior_distance import pairwise_agent_distances
from metrics.control import compute_control
from metrics.spread import compute_spread
from metrics.zombieness import zombieness_proxy_from_pair


def evaluate_population(
    agents: Sequence[str],
    *,
    exclude_roles: Optional[Set[str]] = None,
) -> List[str]:
    """Default evaluation population excludes adversary (already left) and defender."""
    skip = exclude_roles or {"adversary", "defender"}
    # agents here are ids; caller passes filtered ids. Keep helper for clarity.
    return [a for a in agents if a not in skip]


def compute_metric_bundle(
    *,
    pair_id: str,
    seed: int,
    poison_level: int,
    topology: str,
    budget_k: int,
    trigger: str,
    adv_events: Sequence[Event],
    clean_events: Sequence[Event],
    agent_ids: Sequence[str],
    detected: str,
    true_path: Sequence[str],
    poison_hash: str,
    tau_C: float = 0.15,
    tau_Z: float = 0.15,
    exclude_from_control: Optional[Iterable[str]] = None,
    clean_agents_for_attr: Optional[Set[str]] = None,
    injection_step: int = 0,
) -> MetricBundle:
    """Assemble pairwise metrics. Requires both adv and clean logs (§2.4 / §4)."""
    if not adv_events or not clean_events:
        raise ValueError(
            "compute_metric_bundle requires both adv_events and clean_events; "
            "never compute MetricBundle from a single unpaired log"
        )
    exclude = set(exclude_from_control or {"adversary", "defender"})
    eval_agents = [a for a in agent_ids if a not in exclude]

    distances = pairwise_agent_distances(adv_events, clean_events, eval_agents)
    control = compute_control(
        list(adv_events),
        list(clean_events),
        eval_agents,
        tau_C=tau_C,
        exclude=exclude,
    )
    spread = compute_spread(control, eval_agents)
    zombie = zombieness_proxy_from_pair(
        adv_events,
        clean_events,
        eval_agents,
        tau_Z=tau_Z,
        after_step=injection_step,
        trigger_label=trigger,
    )
    attr: AttributionResult = attribute_from_events(
        adv_events,
        detected=detected,
        true_path=true_path,
        poison_hash=poison_hash,
        clean_agents=clean_agents_for_attr
        or ({a for a in eval_agents if a not in set(true_path)}),
        source_absent=True,
    )

    return MetricBundle(
        pair_id=pair_id,
        seed=seed,
        poison_level=poison_level,
        topology=topology,
        budget_k=budget_k,
        trigger=trigger,
        c_i=control.c_i,
        C=control.C,
        C_bar=control.C_bar,
        S=spread.S,
        Z_i=zombie.Z_i,
        R_Z=zombie.R_Z,
        attribution=attr.to_dict(),
        extras={
            "distances": {a: d.to_dict() for a, d in distances.items()},
            "frontier_point": spread.frontier_point,
            "zombieness_by_trigger": zombie.by_trigger,
            "tau_C": tau_C,
            "tau_Z": tau_Z,
        },
    )
