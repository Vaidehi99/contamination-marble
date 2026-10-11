"""Event log schema helpers and JSONL I/O (§3)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from interfaces import Event, RunType

EVENT_TYPES = (
    "chat",
    "blueprint_edit",
    "block_place",
    "resource_alloc",
    "task_complete",
    "tool_use",
    "strategy_note",
    # Level-3 exploit + Level-5 agenda strategy classes (Mode A)
    "chest_withdraw",
    "stash_private",
    "underreport",
    "private_build",
)

COMMUNICATIVE_TYPES = frozenset({"chat", "blueprint_edit", "strategy_note"})


def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_event(
    *,
    episode_id: str,
    run_type: str,
    step: int,
    agent: str,
    event_type: str,
    channel_from: Optional[str] = None,
    raw_content: Optional[str] = None,
    content_hash: Optional[str] = None,
    content_embedding: Optional[List[float]] = None,
    action_payload: Optional[Dict[str, Any]] = None,
    is_injection_event: bool = False,
    poison_level: Optional[int] = None,
) -> Event:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event_type: {event_type}")
    payload = dict(action_payload or {})
    text = raw_content
    if text is None and "text" in payload:
        text = str(payload["text"])
    chash = content_hash
    if chash is None and text:
        chash = hash_content(text)
    return Event(
        episode_id=episode_id,
        run_type=run_type,
        step=step,
        agent=agent,
        event_type=event_type,
        channel_from=channel_from,
        content_hash=chash,
        content_embedding=content_embedding,
        action_payload=payload,
        is_injection_event=is_injection_event,
        poison_level=poison_level,
        raw_content=text,
    )


def event_from_dict(raw: Dict[str, Any]) -> Event:
    return Event(
        episode_id=str(raw["episode_id"]),
        run_type=str(raw["run_type"]),
        step=int(raw["step"]),
        agent=str(raw["agent"]),
        event_type=str(raw["event_type"]),
        channel_from=raw.get("channel_from"),
        content_hash=raw.get("content_hash"),
        content_embedding=raw.get("content_embedding"),
        action_payload=dict(raw.get("action_payload") or {}),
        is_injection_event=bool(raw.get("is_injection_event", False)),
        poison_level=raw.get("poison_level"),
        raw_content=raw.get("raw_content"),
    )


def event_to_public_dict(event: Event) -> Dict[str, Any]:
    """Spec §3 public fields; raw_content kept for debugging/replay."""
    return event.to_dict()


def write_jsonl(path: Union[str, Path], events: Iterable[Event]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event_to_public_dict(event), sort_keys=True) + "\n")
    return out


def read_jsonl(path: Union[str, Path]) -> List[Event]:
    events: List[Event] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            events.append(event_from_dict(json.loads(line)))
    return events


def injection_events(events: Iterable[Event]) -> List[Event]:
    return [e for e in events if e.is_injection_event]


__all__ = [
    "COMMUNICATIVE_TYPES",
    "EVENT_TYPES",
    "Event",
    "RunType",
    "event_from_dict",
    "event_to_public_dict",
    "hash_content",
    "injection_events",
    "make_event",
    "read_jsonl",
    "write_jsonl",
]
