"""Control metrics c_i, C, C_bar (§4.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

from metrics.behavior_distance import DistanceBreakdown, pairwise_agent_distances
from interfaces import Event


@dataclass
class ControlResult:
    c_i: Dict[str, float]
    C: List[str]
    C_bar: float
    tau_C: float

    def to_dict(self) -> Dict:
        return {
            "c_i": self.c_i,
            "C": self.C,
            "C_bar": self.C_bar,
            "tau_C": self.tau_C,
        }


def control_from_distances(
    distances: Mapping[str, DistanceBreakdown],
    *,
    tau_C: float = 0.15,
    exclude: Optional[Iterable[str]] = None,
) -> ControlResult:
    skip = set(exclude or ())
    c_i = {
        agent: breakdown.D
        for agent, breakdown in distances.items()
        if agent not in skip
    }
    controlled = sorted([a for a, c in c_i.items() if c > tau_C])
    c_bar = (
        sum(c_i[a] for a in controlled) / len(controlled) if controlled else 0.0
    )
    return ControlResult(c_i=c_i, C=controlled, C_bar=c_bar, tau_C=tau_C)


def compute_control(
    adv_events: List[Event],
    clean_events: List[Event],
    agent_ids: Iterable[str],
    *,
    tau_C: float = 0.15,
    exclude: Optional[Iterable[str]] = None,
) -> ControlResult:
    distances = pairwise_agent_distances(adv_events, clean_events, agent_ids)
    return control_from_distances(distances, tau_C=tau_C, exclude=exclude)
