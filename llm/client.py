"""OpenAI-compatible client helpers (chat + embeddings)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llm.cache import JsonCache, prompt_hash

# Default agent model for LLM Mode A (OpenAI GPT-5.6 Luna).
DEFAULT_LLM_MODEL = "gpt-5.6-luna"


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


def _is_reasoning_model(model: str) -> bool:
    m = (model or "").lower()
    return (
        m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
        or m.startswith("gpt-5")
        or "luna" in m
        or "terra" in m
        or "sol" in m
    )


def _attach_reasoning_effort(create_kwargs: Dict[str, Any], effort: str) -> None:
    """Pass reasoning_effort even on older openai SDKs that lack the kwarg."""
    import inspect

    try:
        from openai.resources.chat.completions import Completions

        params = inspect.signature(Completions.create).parameters
    except Exception:
        params = {}
    if "reasoning_effort" in params:
        create_kwargs["reasoning_effort"] = effort
        return
    extra = dict(create_kwargs.get("extra_body") or {})
    extra["reasoning_effort"] = effort
    create_kwargs["extra_body"] = extra


def make_openai_client():
    from openai import OpenAI

    key = load_api_key()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running LLM Mode A."
        )
    kwargs: Dict[str, Any] = {"api_key": key}
    base = os.environ.get("OPENAI_BASE_URL", "").strip()
    if base:
        kwargs["base_url"] = base
    return OpenAI(**kwargs)


def _usage_debug(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        "reasoning_tokens": (
            getattr(details, "reasoning_tokens", None) if details is not None else None
        ),
    }


def _choice_debug(response: Any) -> Dict[str, Any]:
    choice = response.choices[0]
    message = choice.message
    content = getattr(message, "content", None)
    return {
        "finish_reason": getattr(choice, "finish_reason", None),
        "content_len": len(content) if isinstance(content, str) else 0,
        "refusal": getattr(message, "refusal", None),
        "usage": _usage_debug(response),
    }


def chat_json(
    *,
    messages: Sequence[Dict[str, str]],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    reasoning_effort: Optional[str] = None,
    cache: Optional[JsonCache] = None,
    client: Any = None,
) -> str:
    """Chat completion returning raw text (expected JSON). Cached by prompt hash."""
    effort = reasoning_effort
    if effort is None:
        effort = os.environ.get("OPENAI_REASONING_EFFORT", "").strip() or None
    if effort is None and _is_reasoning_model(model):
        effort = "low"

    key_payload = {
        "messages": list(messages),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": effort,
        "kind": "chat_json",
    }
    key = prompt_hash(key_payload)
    if cache is not None:
        hit = cache.get(key)
        if isinstance(hit, str):
            return hit

    if client is None:
        client = make_openai_client()

    reasoning = _is_reasoning_model(model)

    def _create(limit: int) -> Any:
        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "response_format": {"type": "json_object"},
        }
        if reasoning:
            # GPT-5 / o-series: use max_completion_tokens; custom temperature often rejected.
            create_kwargs["max_completion_tokens"] = limit
            if effort:
                _attach_reasoning_effort(create_kwargs, effort)
        else:
            create_kwargs["temperature"] = temperature
            create_kwargs["max_tokens"] = limit
        return client.chat.completions.create(**create_kwargs)

    response = _create(max_tokens)
    content = response.choices[0].message.content
    debug = _choice_debug(response)
    if not content:
        finish = debug.get("finish_reason")
        refusal = debug.get("refusal")
        retry_limit = max(int(max_tokens) * 4, 4000)
        if (
            not refusal
            and finish != "content_filter"
            and retry_limit > max_tokens
        ):
            response = _create(retry_limit)
            content = response.choices[0].message.content
            debug["retry_max_tokens"] = retry_limit
            debug["retry"] = _choice_debug(response)
        if not content:
            raise RuntimeError(f"Empty model response {debug}")
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
