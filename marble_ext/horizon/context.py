"""Condition A (continuous) vs B (episode-boundary reconstruction)."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from llm.decision import InboxMessage
from marble_ext.horizon.types import ContextPolicy


class ContextStore:
    def __init__(self, policy: ContextPolicy) -> None:
        self.policy = policy
        self.inbox: Dict[str, List[InboxMessage]] = defaultdict(list)
        self.peer_memory: Dict[str, List[str]] = defaultdict(list)
        self.history: List[Dict[str, str]] = []  # archived agent-visible turns (A)
        self.episode_history: List[Dict[str, str]] = []  # current episode only

    def reset_boundary(self) -> None:
        """Condition B: drop conversation/plans/scratchpads/queues."""
        if self.policy != ContextPolicy.RESET_BETWEEN_EPISODES:
            return
        self.inbox = defaultdict(list)
        self.peer_memory = defaultdict(list)
        self.history = []
        self.episode_history = []

    def archive_visible(self, agent_id: str, task_status: str, inbox: List[InboxMessage]) -> None:
        if self.policy != ContextPolicy.CONTINUOUS:
            return
        self.history.append(
            {
                "agent_id": agent_id,
                "task_status": task_status[:240],
                "inbox": " | ".join(m.content[:80] for m in inbox),
            }
        )

    def record_chat(self, *, dest: str, src: str, content: str) -> None:
        row = {"agent_id": dest, "from": src, "content": str(content)}
        self.episode_history.append(row)
        if self.policy != ContextPolicy.CONTINUOUS:
            return
        self.history.append(row)

    def prior_for(self, agent_id: str, *, cap: int = 32) -> List[Dict[str, str]]:
        if self.policy != ContextPolicy.CONTINUOUS:
            return []
        rows = [h for h in self.history if h.get("agent_id") == agent_id]
        return rows[-cap:]

    def episode_prior_for(
        self, agent_id: str, *, cap: int = 32
    ) -> Tuple[List[Dict[str, str]], bool]:
        rows = [h for h in self.episode_history if h.get("agent_id") == agent_id]
        truncated = len(rows) > cap
        return rows[-cap:], truncated

    def reconstructed_fields(self) -> Dict[str, object]:
        """What a B request may contain besides current world + task."""
        if self.policy == ContextPolicy.RESET_BETWEEN_EPISODES:
            return {
                "prior_conversation": [],
                "peer_memory": {},
                "inbox": {},
            }
        return {
            "prior_conversation_turns": len(self.history),
            "peer_memory": {k: list(v) for k, v in self.peer_memory.items()},
        }
