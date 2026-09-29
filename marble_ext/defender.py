"""Read-only defender stub (RQ3/RQ4). Never issues build actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from interfaces import Event


@dataclass
class Defender:
    agent_id: str = "defender"
    observed: List[Event] = field(default_factory=list)

    def observe(self, event: Event) -> None:
        self.observed.append(event)

    def observe_many(self, events: List[Event]) -> None:
        self.observed.extend(events)
