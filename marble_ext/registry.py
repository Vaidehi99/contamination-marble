"""Mutable agent registry for an episode (adversary can be removed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from interfaces import AgentRoleConfig, OrgConfig


@dataclass
class AgentRegistry:
    """Active agents for the current episode step."""

    _agents: Dict[str, AgentRoleConfig] = field(default_factory=dict)
    _removed: Set[str] = field(default_factory=set)

    @classmethod
    def from_org(cls, org: OrgConfig, *, include_defender: bool = False) -> "AgentRegistry":
        agents: Dict[str, AgentRoleConfig] = {}
        for agent in org.agents:
            if agent.role == "defender" and not include_defender:
                continue
            agents[agent.agent_id] = agent
        return cls(_agents=agents)

    def active_ids(self) -> List[str]:
        return sorted(self._agents.keys())

    def is_active(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def get(self, agent_id: str) -> Optional[AgentRoleConfig]:
        return self._agents.get(agent_id)

    def role(self, agent_id: str) -> Optional[str]:
        agent = self._agents.get(agent_id)
        return agent.role if agent else None

    def add(self, agent: AgentRoleConfig) -> None:
        """Add or restore an agent (used by horizon adversary_join)."""
        self._agents[agent.agent_id] = agent
        self._removed.discard(agent.agent_id)

    def remove(self, agent_id: str) -> bool:
        """Remove agent from the registry (no longer addressable)."""
        if agent_id not in self._agents:
            return False
        del self._agents[agent_id]
        self._removed.add(agent_id)
        return True

    @property
    def removed(self) -> Set[str]:
        return set(self._removed)

    def neighbors(self, org: OrgConfig, agent_id: str) -> List[str]:
        """Active neighbors per org relationship edges."""
        out: List[str] = []
        for rel in org.relationships:
            if len(rel) < 2:
                continue
            a, b = rel[0], rel[1]
            if a == agent_id and self.is_active(b):
                out.append(b)
            elif b == agent_id and self.is_active(a):
                out.append(a)
        # unique preserve order
        seen: Set[str] = set()
        uniq: List[str] = []
        for n in out:
            if n not in seen:
                seen.add(n)
                uniq.append(n)
        return uniq
