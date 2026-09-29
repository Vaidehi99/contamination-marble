"""OpenAI-compatible client helpers (chat + embeddings)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llm.cache import JsonCache, prompt_hash


def load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    for path in (
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
        Path(__file__).resolve().parents[2] / "MARBLE" / ".env",
    ):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == "OPENAI_API_KEY":
                val = v.strip().strip('"').strip("'")
                if val:
                    return val
    return ""


def make_openai_client():
    from openai import OpenAI

    key = load_api_key()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running LLM Mode A."
        )
    return OpenAI(api_key=key)


def chat_json(
    *,
    messages: Sequence[Dict[str, str]],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 400,
    cache: Optional[JsonCache] = None,
    client: Any = None,
) -> str:
    """Chat completion returning raw text (expected JSON). Cached by prompt hash."""
    key_payload = {
        "messages": list(messages),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "kind": "chat_json",
    }
    key = prompt_hash(key_payload)
    if cache is not None:
        hit = cache.get(key)
        if isinstance(hit, str):
            return hit

    if client is None:
        client = make_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=list(messages),
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Empty model response")
    text = content.strip()
    if cache is not None:
        cache.set(key, text)
    return text


def embed_texts(
    texts: Sequence[str],
    *,
    model: str = "text-embedding-3-small",
    cache: Optional[JsonCache] = None,
    client: Any = None,
) -> List[List[float]]:
    """Embed a batch of texts; per-text cache by (model, text)."""
    out: List[Optional[List[float]]] = [None] * len(texts)
    missing_idx: List[int] = []
    missing_texts: List[str] = []

    for i, text in enumerate(texts):
        key = prompt_hash({"kind": "embed", "model": model, "text": text})
        if cache is not None:
            hit = cache.get(key)
            if isinstance(hit, list):
                out[i] = [float(x) for x in hit]
                continue
        missing_idx.append(i)
        missing_texts.append(text)

    if missing_texts:
        if client is None:
            client = make_openai_client()
        resp = client.embeddings.create(model=model, input=list(missing_texts))
        for j, item in enumerate(resp.data):
            vec = [float(x) for x in item.embedding]
            i = missing_idx[j]
            out[i] = vec
            if cache is not None:
                key = prompt_hash(
                    {"kind": "embed", "model": model, "text": missing_texts[j]}
                )
                cache.set(key, vec)

    return [v if v is not None else [] for v in out]
