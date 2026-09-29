"""Attach real (or hash-fallback) embeddings onto communicative events for d_chat."""

from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence

from interfaces import Event
from llm.cache import JsonCache
from llm.client import embed_texts, load_api_key
from paths import RESULTS_ROOT

_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
COMMUNICATIVE = frozenset({"chat", "strategy_note", "blueprint_edit"})


def hash_embed(text: str, dim: int = 16) -> List[float]:
    vec = [0.0] * dim
    for tok in _TOKEN_RE.findall(text.lower()):
        vec[hash(tok) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def enrich_event_embeddings(
    events: Sequence[Event],
    *,
    backend: str = "auto",
    model: str = "text-embedding-3-small",
    cache: Optional[JsonCache] = None,
) -> int:
    """Fill ``content_embedding`` on communicative events missing one.

    backend:
      - ``auto``: OpenAI if API key present, else hash-bag
      - ``openai``: require API
      - ``hash``: always hash-bag (tests / offline)
    Returns number of events newly embedded.
    """
    mode = backend
    if mode == "auto":
        mode = "openai" if load_api_key() else "hash"

    pending_idx: List[int] = []
    pending_text: List[str] = []
    for i, event in enumerate(events):
        if event.event_type not in COMMUNICATIVE:
            continue
        if event.content_embedding:
            continue
        text = event.raw_content or str((event.action_payload or {}).get("text") or "")
        if not text:
            continue
        pending_idx.append(i)
        pending_text.append(text)

    if not pending_text:
        return 0

    if mode == "hash":
        for i, text in zip(pending_idx, pending_text):
            events[i].content_embedding = hash_embed(text)
        return len(pending_idx)

    if cache is None:
        cache = JsonCache(RESULTS_ROOT / "cache" / "embeddings.json")
    vectors = embed_texts(pending_text, model=model, cache=cache)
    for i, vec in zip(pending_idx, vectors):
        events[i].content_embedding = vec
    return len(pending_idx)
