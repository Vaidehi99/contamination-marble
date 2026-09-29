"""Zombieness Z_i and R_Z (§4.3).

Phase 3 supports two modes:
1. Continuation logs per trigger (preferred): mean D over future windows.
2. Fallback: use late-episode pairwise D from the main pair as a proxy Z_i.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from interfaces import Event
from metrics.behavior_distance import behavior_distance


@dataclass
class ZombienessResult:
    Z_i: Dict[str, float]
    R_Z: float
    tau_Z: float
    by_trigger: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "Z_i": self.Z_i,
            "R_Z": self.R_Z,
            "tau_Z": self.tau_Z,
            "by_trigger": self.by_trigger,
        }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def zombieness_from_continuations(
    continuations: Mapping[str, Sequence[tuple[Sequence[Event], Sequence[Event]]]],
    agent_ids: Iterable[str],
    *,
    tau_Z: float = 0.15,
) -> ZombienessResult:
    """continuations[trigger] = list of (adv_future_events, clean_future_events) windows."""
    by_trigger: Dict[str, Dict[str, float]] = {}
    aggregated: Dict[str, List[float]] = {a: [] for a in agent_ids}

    for trigger, windows in continuations.items():
        z_trigger: Dict[str, float] = {}
        for agent in agent_ids:
            dists = [
                behavior_distance(adv_f, clean_f, agent).D
                for adv_f, clean_f in windows
            ]
            z_trigger[agent] = _mean(dists)
            aggregated[agent].append(z_trigger[agent])
        by_trigger[trigger] = z_trigger

    z_i = {agent: _mean(vals) for agent, vals in aggregated.items()}
    n = len(z_i) or 1
    r_z = sum(1 for v in z_i.values() if v > tau_Z) / n
    return ZombienessResult(Z_i=z_i, R_Z=r_z, tau_Z=tau_Z, by_trigger=by_trigger)


def zombieness_proxy_from_pair(
    adv_events: Sequence[Event],
    clean_events: Sequence[Event],
    agent_ids: Iterable[str],
    *,
    tau_Z: float = 0.15,
    after_step: int = 0,
    trigger_label: str = "no_trigger",
) -> ZombienessResult:
    """Approximate Z_i using post-``after_step`` main-episode pairwise D."""
    adv_f = [e for e in adv_events if e.step >= after_step]
    clean_f = [e for e in clean_events if e.step >= after_step]
    z_i = {
        agent: behavior_distance(adv_f, clean_f, agent).D for agent in agent_ids
    }
    n = len(z_i) or 1
    r_z = sum(1 for v in z_i.values() if v > tau_Z) / n
    return ZombienessResult(
        Z_i=z_i,
        R_Z=r_z,
        tau_Z=tau_Z,
        by_trigger={trigger_label: dict(z_i)},
    )
