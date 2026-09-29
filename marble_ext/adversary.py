"""Adversary with bounded interaction budget k (§2.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set

from interfaces import InjectionSpec
from marble_ext.registry import AgentRegistry


@dataclass
class AdversaryController:
    """Tracks adversary turns and forces removal after budget ``k``.

    Turn indexing is 1..k relative to the adversary's participation budget
    (not global episode step).
    """

    agent_id: str
    budget_k: int
    inject_turns: List[int] = field(default_factory=list)
    turns_used: int = 0
    removed: bool = False

    @classmethod
    def from_injection_spec(cls, agent_id: str, spec: InjectionSpec) -> "AdversaryController":
        turns = list(spec.inject_turns) if spec.inject_turns else [spec.budget_k]
        return cls(
            agent_id=agent_id,
            budget_k=int(spec.budget_k),
            inject_turns=turns,
        )

    @property
    def remaining(self) -> int:
        if self.removed:
            return 0
        return max(0, self.budget_k - self.turns_used)

    def is_active(self) -> bool:
        return not self.removed and self.turns_used < self.budget_k

    def should_inject_on_next_turn(self) -> bool:
        """True if the upcoming turn index (turns_used+1) is an inject turn."""
        if not self.is_active():
            return False
        next_turn = self.turns_used + 1
        return next_turn in self.inject_turns

    def consume_turn(self, registry: AgentRegistry) -> int:
        """Consume one adversary turn; remove from registry when budget exhausted.

        Returns the 1-indexed turn number just consumed.
        """
        if self.removed:
            raise RuntimeError(f"Adversary {self.agent_id} already removed")
        if self.turns_used >= self.budget_k:
            self._remove(registry)
            raise RuntimeError(f"Adversary {self.agent_id} has no remaining budget")
        self.turns_used += 1
        turn = self.turns_used
        if self.turns_used >= self.budget_k:
            self._remove(registry)
        return turn

    def _remove(self, registry: AgentRegistry) -> None:
        registry.remove(self.agent_id)
        self.removed = True

    def force_remove(self, registry: AgentRegistry) -> None:
        if not self.removed:
            self._remove(registry)
