"""Explicit memory (plan §7). Condition C is private-only; D/E are not implemented.

C policy (frozen before each C run):
- Context: B-style reset at every episode boundary (inbox, peer notes, prior turns).
- World: persist (same manifest as B).
- Memory: private per agent; injected and L0 agents have identical access.
- Do not auto-save the injection, inbox, or chat notes.
- Capacity: 4 items / agent. Overwrite: FIFO drop oldest. No within-run expiry.
- Retrieval: most-recent ``retrieval_k`` (default 4) items, injected into the
  fresh observation as ``own_state.private_memory.retrieved`` — not as fake chat.
- Logging: save → retrieve → action (``memory_trace``); FIFO evictions logged.

Write modes (archived separately):
- v1 (archived): optional appendix field; Luna uptake was zero.
- v2: ``memory.write`` required every turn; saving remains optional
  (``enabled: false`` is a valid no-save). Applied after that turn's action.
- v3: one mandatory agent-chosen handoff per reasoning agent at each non-final
  episode boundary. Ordinary turns use the A/B action+note schema. Writing is
  required; saving the poison is not. Matched control (C3ctl) generates the
  same handoffs with ``retrieval_enabled=false``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from event_log.schema import hash_content
from marble_ext.horizon.types import MemoryPolicy

OFF_STRINGS = frozenset({"off", "false", "none", "null", "disabled", "0"})
POLICY_VALUES = frozenset(p.value for p in MemoryPolicy)

DEFAULT_CAPACITY = 4
DEFAULT_RETRIEVAL_K = 4

WRITE_MODE_TURN_EXPLICIT = "agent_turn_explicit"
WRITE_MODE_BOUNDARY = "episode_boundary_mandatory"
WRITE_MODE_VALUES = frozenset({WRITE_MODE_TURN_EXPLICIT, WRITE_MODE_BOUNDARY})

PRIVATE_MEMORY_SPEC: Dict[str, Any] = {
    "policy": MemoryPolicy.PRIVATE.value,
    "scope": "private_per_agent",
    "read_permission": "author_only",
    "write_permission": "author_only",
    "capacity": DEFAULT_CAPACITY,
    "retrieval_k": DEFAULT_RETRIEVAL_K,
    "retrieval": "most_recent",
    "retrieval_enabled": True,
    "write_mode": WRITE_MODE_TURN_EXPLICIT,
    "write_timing": (
        "agent_chosen on a normal turn; applied after the action; "
        "available on the next decision, including after a B reset"
    ),
    "overwrite": "fifo_drop_oldest",
    "expiry": "none_within_trajectory",
    "auto_save_injection": False,
    "auto_save_notes": False,
    "auto_save_inbox": False,
    "injected_l0_identical": True,
    "presentation": "own_state.private_memory.retrieved (not reconstructed as chat)",
    "consolidation_llm_calls": 0,
    "response_schema": "action_note_memory_explicit",
    "write_field_required": True,
    "handoff_write_required": False,
    "save_optional": True,
    "protocol_id": "long_horizon_memory_private_v2",
    "prior_protocol": (
        "long_horizon_memory_private_v1 optional-field policy: zero memory uptake"
    ),
}


def parse_memory_policy(raw: Any) -> MemoryPolicy:
    """Canonicalize YAML/CLI memory_policy.

    Unquoted YAML 1.1 ``off``/``false``/``no`` becomes boolean ``False``. That is
    a configuration bug to tolerate (map to ``off``), not a distinct policy.
    Unquoted ``on``/``true``/``yes`` becomes ``True`` and is rejected.
    """
    if isinstance(raw, MemoryPolicy):
        return raw
    if raw is None:
        return MemoryPolicy.OFF
    if raw is False:
        return MemoryPolicy.OFF
    if raw is True:
        raise ValueError(
            'memory_policy true/on/yes is not a valid policy. '
            'Quote "off" or use an explicit enum: off | private | shared | both.'
        )
    text = str(raw).strip().lower()
    if text in OFF_STRINGS:
        return MemoryPolicy.OFF
    for policy in MemoryPolicy:
        if text == policy.value:
            return policy
    raise ValueError(
        f"Unknown memory_policy {raw!r}. Expected off | private | shared | both "
        '(quote "off" in YAML so PyYAML does not coerce it to false).'
    )


def parse_write_mode(raw: Any) -> str:
    if raw is None:
        return WRITE_MODE_TURN_EXPLICIT
    text = str(raw).strip().lower()
    if text in {WRITE_MODE_TURN_EXPLICIT, "turn_explicit", "v2", "agent_chosen_turn"}:
        return WRITE_MODE_TURN_EXPLICIT
    if text in {
        WRITE_MODE_BOUNDARY,
        "boundary",
        "handoff",
        "v3",
        "mandatory_handoff",
        "episode_boundary",
    }:
        return WRITE_MODE_BOUNDARY
    raise ValueError(
        f"Unknown write_mode {raw!r}. Expected {WRITE_MODE_TURN_EXPLICIT} "
        f"or {WRITE_MODE_BOUNDARY}."
    )


def parse_bool_flag(raw: Any, *, default: bool = True) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"true", "on", "yes", "1"}:
        return True
    if text in {"false", "off", "no", "0"}:
        return False
    raise ValueError(f"Expected boolean flag, got {raw!r}")


def yaml_off_coerced_to_bool(raw: Any) -> bool:
    return raw is False


def _apply_write_mode_defaults(spec: Dict[str, Any]) -> None:
    mode = parse_write_mode(spec.get("write_mode"))
    spec["write_mode"] = mode
    if mode != WRITE_MODE_BOUNDARY:
        return
    spec["protocol_id"] = "long_horizon_memory_private_v3"
    spec["prior_protocol"] = (
        "long_horizon_memory_private_v2 every-turn explicit field; save optional. "
        "v1 optional-field policy: zero memory uptake"
    )
    spec["write_timing"] = (
        "one mandatory agent-chosen handoff per reasoning agent at each "
        "non-final episode boundary; ordinary turns do not write; "
        "content is selected by the agent (writing required, poison not)"
    )
    spec["response_schema"] = (
        "ordinary turns: action+note (A/B schema); "
        "episode-end: idle + memory.write required"
    )
    spec["write_field_required"] = False
    spec["handoff_write_required"] = True
    spec["save_optional"] = True
    spec["consolidation_llm_calls"] = "one_per_reasoning_agent_per_non_final_episode"
    spec["matched_control"] = (
        "same mandatory handoff generation with retrieval_enabled=false (C3ctl)"
    )


def memory_spec_for(
    policy: MemoryPolicy, overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if policy == MemoryPolicy.OFF:
        spec = {
            "policy": MemoryPolicy.OFF.value,
            "auto_save_injection": False,
            "injected_l0_identical": True,
            "capacity": 0,
            "retrieval_k": 0,
            "retrieval_enabled": False,
            "write_mode": WRITE_MODE_TURN_EXPLICIT,
            "consolidation_llm_calls": 0,
        }
        return spec
    if policy in {MemoryPolicy.SHARED, MemoryPolicy.BOTH}:
        raise NotImplementedError(
            f"memory_policy {policy.value!r} (D/E) is not implemented; run C (private) first"
        )
    spec = dict(PRIVATE_MEMORY_SPEC)
    for key, value in dict(overrides or {}).items():
        if key in spec:
            spec[key] = value
    if spec.get("auto_save_injection"):
        raise ValueError("auto_save_injection must stay false; do not save the poison for agents")
    if spec.get("auto_save_notes") or spec.get("auto_save_inbox"):
        raise ValueError("C writes are agent-chosen only; auto-save of notes/inbox is off")
    spec["capacity"] = int(spec.get("capacity") or DEFAULT_CAPACITY)
    spec["retrieval_k"] = int(spec.get("retrieval_k") or DEFAULT_RETRIEVAL_K)
    if spec["capacity"] < 1 or spec["retrieval_k"] < 1:
        raise ValueError("private memory capacity and retrieval_k must be >= 1")
    _apply_write_mode_defaults(spec)
    retrieval = spec.get("retrieval_enabled", True)
    if not isinstance(retrieval, list):
        spec["retrieval_enabled"] = parse_bool_flag(retrieval, default=True)
    return spec


def condition_letter(
    context_policy: Any,
    memory_policy: MemoryPolicy,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    from marble_ext.horizon.types import ContextPolicy

    if context_policy == ContextPolicy.CONTINUOUS:
        if memory_policy != MemoryPolicy.OFF:
            raise ValueError(
                "Condition A (continuous context) is memory-off; "
                "C/D/E require episode-boundary resets"
            )
        return "A"
    if memory_policy == MemoryPolicy.OFF:
        return "B"
    if memory_policy == MemoryPolicy.PRIVATE:
        mem = dict((extra or {}).get("memory") or {})
        if parse_write_mode(mem.get("write_mode")) == WRITE_MODE_BOUNDARY:
            retrieval = mem.get("retrieval_enabled", True)
            if isinstance(retrieval, list):
                retrieval = retrieval[0] if retrieval else True
            if not parse_bool_flag(retrieval, default=True):
                return "C3ctl"
            return "C3"
        return "C"
    if memory_policy == MemoryPolicy.SHARED:
        return "D"
    if memory_policy == MemoryPolicy.BOTH:
        return "E"
    raise ValueError(f"Unknown memory_policy {memory_policy!r}")


@dataclass
class MemoryItem:
    item_id: str
    author: str
    content: str
    content_hash: str
    episode_id: Optional[str]
    episode_step: int
    global_step: int
    source_event_ids: List[str] = field(default_factory=list)
    revision: int = 1
    deleted: bool = False
    retrievals: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def prompt_view(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "content": self.content,
            "written_episode": self.episode_id,
            "written_step": self.global_step,
        }


class MemoryStore:
    """Private (C) store. Shared/both scopes raise until D/E are specified."""

    def __init__(
        self,
        policy: MemoryPolicy = MemoryPolicy.OFF,
        *,
        spec: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.policy = parse_memory_policy(policy)
        self.spec = memory_spec_for(self.policy, spec if self.policy != MemoryPolicy.OFF else None)
        self.capacity = int(self.spec.get("capacity") or 0)
        self.retrieval_k = int(self.spec.get("retrieval_k") or 0)
        retrieval = self.spec.get("retrieval_enabled", True)
        if isinstance(retrieval, list):
            raise ValueError("retrieval_enabled must be a single bool on a trajectory")
        self.retrieval_enabled = bool(retrieval)
        self.write_mode = parse_write_mode(self.spec.get("write_mode"))
        self._private: Dict[str, List[MemoryItem]] = {}
        self._seq = 0
        self.trace: List[Dict[str, Any]] = []
        self.rejected_writes: List[Dict[str, Any]] = []
        self.evictions: List[Dict[str, Any]] = []

    def n_private(self, agent_id: str) -> int:
        return len(self._private.get(agent_id) or [])

    def total_private_items(self) -> int:
        return sum(len(v) for v in self._private.values())

    def items_snapshot(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            agent: [item.to_dict() for item in items]
            for agent, items in sorted(self._private.items())
        }

    def retrieve_for_prompt(
        self,
        agent_id: str,
        *,
        clocks: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if self.policy != MemoryPolicy.PRIVATE or not self.retrieval_enabled:
            return []
        items = [i for i in self._private.get(agent_id, []) if not i.deleted]
        chosen = items[-self.retrieval_k :] if self.retrieval_k else []
        for item in chosen:
            item.retrievals.append(
                {
                    "global_step": clocks.get("global_step"),
                    "horizon_episode": clocks.get("horizon_episode"),
                    "episode_step": clocks.get("episode_step"),
                    "agent_id": agent_id,
                }
            )
        return [item.prompt_view() for item in chosen]

    def maybe_write(
        self,
        *,
        agent_id: str,
        enabled: bool,
        content: Optional[str],
        clocks: Dict[str, Any],
        source_event_ids: Optional[List[str]] = None,
        phase: str = "turn",
    ) -> Optional[MemoryItem]:
        text = (content or "").strip() if content else ""
        if not enabled or not text:
            return None
        record = {
            "agent_id": agent_id,
            "clocks": dict(clocks),
            "content_len": len(text),
            "phase": phase,
        }
        if self.policy != MemoryPolicy.PRIVATE:
            self.rejected_writes.append({**record, "reason": "memory_policy_off_or_unsupported"})
            return None
        if self.write_mode == WRITE_MODE_BOUNDARY and phase != "handoff":
            self.rejected_writes.append({**record, "reason": "write_mode_episode_boundary_only"})
            return None
        self._seq += 1
        episode_id = clocks.get("horizon_episode")
        global_step = int(clocks.get("global_step") or 0)
        item = MemoryItem(
            item_id=f"mem|{agent_id}|{global_step}|{self._seq}",
            author=agent_id,
            content=text,
            content_hash=hash_content(text),
            episode_id=str(episode_id) if episode_id is not None else None,
            episode_step=int(clocks.get("episode_step") or 0),
            global_step=global_step,
            source_event_ids=list(source_event_ids or []),
        )
        bucket = self._private.setdefault(agent_id, [])
        bucket.append(item)
        if len(bucket) > self.capacity:
            n_drop = len(bucket) - self.capacity
            for dropped in bucket[:n_drop]:
                self.evictions.append(
                    {
                        "item_id": dropped.item_id,
                        "author": dropped.author,
                        "content": dropped.content,
                        "content_hash": dropped.content_hash,
                        "written_episode": dropped.episode_id,
                        "written_global_step": dropped.global_step,
                        "evicted_at": dict(clocks),
                        "reason": "fifo_drop_oldest",
                    }
                )
            del bucket[:n_drop]
        return item

    def record_decision(
        self,
        *,
        agent_id: str,
        clocks: Dict[str, Any],
        retrieved: List[Dict[str, Any]],
        action_type: str,
        action_payload: Optional[Dict[str, Any]],
        wrote: Optional[MemoryItem],
        rejected: bool = False,
        memory_field_present: bool = False,
        write_decision: str = "omitted",
        phase: str = "turn",
    ) -> Dict[str, Any]:
        payload = dict(action_payload or {})
        row = {
            "agent_id": agent_id,
            "horizon_episode": clocks.get("horizon_episode"),
            "episode_step": clocks.get("episode_step"),
            "global_step": clocks.get("global_step"),
            "phase": phase,
            "retrieved": [
                {"item_id": r.get("item_id"), "content": r.get("content")}
                for r in retrieved
            ],
            "retrieved_item_ids": [r.get("item_id") for r in retrieved],
            "action_type": action_type,
            "action": {
                "exploit": payload.get("exploit"),
                "exploit_name": payload.get("exploit_name"),
                "ledger_violation": payload.get("ledger_violation"),
                "resource": payload.get("resource"),
            },
            "memory_field_present": memory_field_present,
            "write_decision": write_decision,
            "wrote": (
                {
                    "item_id": wrote.item_id,
                    "content": wrote.content,
                    "content_hash": wrote.content_hash,
                }
                if wrote
                else None
            ),
            "rejected_write": rejected,
        }
        self.trace.append(row)
        return row
