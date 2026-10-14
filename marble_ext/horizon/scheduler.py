"""Frozen event queue: episode-step times, deterministic seq, apply-once."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

from marble_ext.horizon.types import ScheduledEvent


class EventScheduler:
    def __init__(self, events: Iterable[ScheduledEvent]) -> None:
        self._by_key: Dict[Tuple[str, int], List[ScheduledEvent]] = defaultdict(list)
        self._applied: Set[int] = set()
        self._cancelled: Set[int] = set()
        ordered = sorted(events, key=lambda e: (e.episode_id, e.episode_step, e.seq))
        for ev in ordered:
            self._by_key[(ev.episode_id, ev.episode_step)].append(ev)
        self.events: List[ScheduledEvent] = ordered

    def due(self, episode_id: str, episode_step: int) -> List[ScheduledEvent]:
        pending: List[ScheduledEvent] = []
        for ev in self._by_key.get((episode_id, episode_step), []):
            token = id(ev)
            if token in self._cancelled:
                continue
            if token in self._applied:
                raise RuntimeError(
                    f"Event applied twice: {ev.type} {ev.episode_id}@{ev.episode_step}"
                )
            self._applied.add(token)
            pending.append(ev)
        return pending

    def remaining(self) -> List[ScheduledEvent]:
        return [
            ev
            for ev in self.events
            if id(ev) not in self._applied and id(ev) not in self._cancelled
        ]

    def enqueue(self, event: ScheduledEvent) -> None:
        """Runtime insertion onto the same validated queue."""
        token = id(event)
        if token in self._applied or token in self._cancelled:
            raise RuntimeError(f"Cannot enqueue already-consumed event {event.type}")
        self.events.append(event)
        self.events.sort(key=lambda e: (e.episode_id, e.episode_step, e.seq))
        bucket = self._by_key[(event.episode_id, event.episode_step)]
        bucket.append(event)
        bucket.sort(key=lambda e: e.seq)

    def cancel_pending(self, *, types: Optional[Set[str]] = None) -> List[ScheduledEvent]:
        types = types or set()
        dropped: List[ScheduledEvent] = []
        for ev in self.events:
            token = id(ev)
            if token in self._applied or token in self._cancelled:
                continue
            if ev.type in types:
                self._cancelled.add(token)
                dropped.append(ev)
        return dropped
