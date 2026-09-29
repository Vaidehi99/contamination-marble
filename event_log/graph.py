"""Directed interaction graph from episode event logs (§3).

edge(i -> j, t, content_hash) when j's communicative act is attributable to
something i said/wrote earlier (direct chat recipient, or content reuse within
a following step window).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from event_log.schema import COMMUNICATIVE_TYPES, Event


@dataclass(frozen=True)
class InteractionEdge:
    src: str
    dst: str
    step: int
    content_hash: Optional[str]
    reason: str  # direct_chat | content_reuse | blueprint_reuse | injection


@dataclass
class InteractionGraph:
    edges: List[InteractionEdge] = field(default_factory=list)
    injection_roots: List[InteractionEdge] = field(default_factory=list)

    def outbound(self) -> Dict[str, List[InteractionEdge]]:
        out: Dict[str, List[InteractionEdge]] = defaultdict(list)
        for edge in self.edges:
            out[edge.src].append(edge)
        return out

    def inbound(self) -> Dict[str, List[InteractionEdge]]:
        inp: Dict[str, List[InteractionEdge]] = defaultdict(list)
        for edge in self.edges:
            inp[edge.dst].append(edge)
        return inp

    def edge_triples(self) -> List[Tuple[str, str, int]]:
        return [(e.src, e.dst, e.step) for e in self.edges]


def build_interaction_graph(
    events: Sequence[Event],
    *,
    window: int = 30,
) -> InteractionGraph:
    """Build attribution edges from a chronological event log.

    Rules:
    1. Direct chat: agent A sends chat with action_payload.to = B → edge A→B.
    2. Injection: is_injection_event chat/strategy/blueprint also recorded as root.
    3. Content reuse: if B later emits communicative content whose content_hash
       matches a prior communicative hash from A within ``window`` steps, edge A→B
       (skip if A==B). Prefer earliest prior A for that hash.
    """
    ordered = sorted(events, key=lambda e: (e.step, e.agent, e.event_type))
    edges: List[InteractionEdge] = []
    roots: List[InteractionEdge] = []

    # hash -> list of (step, agent)
    producers: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    for event in ordered:
        recipient = None
        if event.event_type == "chat":
            recipient = event.action_payload.get("to")
            if recipient:
                edge = InteractionEdge(
                    src=event.agent,
                    dst=str(recipient),
                    step=event.step,
                    content_hash=event.content_hash,
                    reason="injection" if event.is_injection_event else "direct_chat",
                )
                edges.append(edge)
                if event.is_injection_event:
                    roots.append(edge)

        if (
            event.event_type in COMMUNICATIVE_TYPES
            and event.content_hash
            and not (event.event_type == "chat" and recipient)
        ):
            # Non-chat communicative or chat without explicit to: still track hash.
            pass

        if event.event_type in COMMUNICATIVE_TYPES and event.content_hash:
            # Content-reuse edges: does this event reuse someone else's prior hash?
            priors = producers.get(event.content_hash, [])
            prior_candidates = [
                (t, src)
                for t, src in priors
                if src != event.agent and 0 < event.step - t <= window
            ]
            if prior_candidates:
                t0, src0 = min(prior_candidates, key=lambda item: (item[0], item[1]))
                reason = (
                    "blueprint_reuse"
                    if event.event_type == "blueprint_edit"
                    else "content_reuse"
                )
                # Avoid duplicating the same direct_chat edge at same step/src/dst/hash
                dup = any(
                    e.src == src0
                    and e.dst == event.agent
                    and e.step == event.step
                    and e.content_hash == event.content_hash
                    for e in edges
                )
                if not dup:
                    edges.append(
                        InteractionEdge(
                            src=src0,
                            dst=event.agent,
                            step=event.step,
                            content_hash=event.content_hash,
                            reason=reason,
                        )
                    )

            producers[event.content_hash].append((event.step, event.agent))

        elif event.is_injection_event and event.event_type != "chat":
            # Injection on blueprint/strategy without chat recipient still marks root
            # via channel_from target if provided in payload.
            target = event.action_payload.get("to") or event.channel_from
            if target:
                edge = InteractionEdge(
                    src=event.agent,
                    dst=str(target),
                    step=event.step,
                    content_hash=event.content_hash,
                    reason="injection",
                )
                edges.append(edge)
                roots.append(edge)

    # Stable unique-ish order
    edges.sort(key=lambda e: (e.step, e.src, e.dst, e.reason))
    return InteractionGraph(edges=edges, injection_roots=roots)


def recover_path_from_detected(
    graph: InteractionGraph,
    detected: str,
    *,
    prefer_hashes: Optional[Set[str]] = None,
) -> List[str]:
    """Walk earliest inbound edges backward from detected to an injection root.

    If ``prefer_hashes`` is set, only edges carrying those content hashes are used
    (poison fingerprint tracing).
    """
    inbound = graph.inbound()
    roots = {e.dst for e in graph.injection_roots} | {e.src for e in graph.injection_roots}

    def edge_ok(edge: InteractionEdge) -> bool:
        if prefer_hashes is None:
            return True
        return edge.content_hash in prefer_hashes if edge.content_hash else False

    chain_rev = [detected]
    seen = {detected}
    current = detected
    while True:
        candidates = [e for e in inbound.get(current, []) if edge_ok(e)]
        if not candidates:
            break
        candidates.sort(key=lambda e: (e.step, e.src))
        pred_edge = candidates[0]
        pred = pred_edge.src
        if pred in seen:
            break
        chain_rev.append(pred)
        seen.add(pred)
        current = pred
        if pred in {e.src for e in graph.injection_roots}:
            break

    return list(reversed(chain_rev))


def ground_truth_path_from_roots(
    graph: InteractionGraph,
    detected: str,
    poison_hash: str,
) -> List[str]:
    return recover_path_from_detected(
        graph, detected, prefer_hashes={poison_hash}
    )
