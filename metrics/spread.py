"""Spread S and control–spread frontier points (§4.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from metrics.control import ControlResult


@dataclass
class SpreadResult:
    S: float
    N: int
    C_size: int
    frontier_point: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "S": self.S,
            "N": self.N,
            "C_size": self.C_size,
            "frontier_point": self.frontier_point,
        }


def compute_spread(
    control: ControlResult,
    population: Iterable[str],
) -> SpreadResult:
    agents = list(population)
    n = len(agents)
    c_size = len(control.C)
    s = (c_size / n) if n else 0.0
    return SpreadResult(
        S=s,
        N=n,
        C_size=c_size,
        frontier_point={"S": s, "C_bar": control.C_bar},
    )


def frontier_points(results: List[SpreadResult]) -> List[Dict[str, float]]:
    return [r.frontier_point for r in results]
