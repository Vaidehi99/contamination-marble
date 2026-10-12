"""OpenAI chat_json empty-content retry / debug."""

from __future__ import annotations

from types import SimpleNamespace

from llm.client import chat_json


def _response(*, content, finish_reason="stop", reasoning_tokens=0, refusal=None):
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=550,
        total_tokens=560,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )
    message = SimpleNamespace(content=content, refusal=refusal)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


class _TruncatingThenOk:
    def __init__(self) -> None:
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        limit = kwargs.get("max_completion_tokens") or kwargs.get("max_tokens")
        if limit and int(limit) < 1000:
            return _response(content="", finish_reason="length", reasoning_tokens=550)
        return _response(content='{"action":{"action_type":"idle"}}', finish_reason="stop")


class _AlwaysEmpty:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        return _response(content="", finish_reason="length", reasoning_tokens=4000)


def test_empty_length_retries_with_higher_cap():
    client = _TruncatingThenOk()
    text = chat_json(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-5.6-luna",
        max_tokens=550,
        client=client,
    )
    assert "idle" in text
    assert len(client.calls) == 2
    first = client.calls[0].get("max_completion_tokens")
    second = client.calls[1].get("max_completion_tokens")
    assert first == 550
    assert second >= 2000


def test_empty_after_retry_includes_finish_reason():
    client = _AlwaysEmpty()
    try:
        chat_json(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-5.6-luna",
            max_tokens=100,
            client=client,
        )
    except RuntimeError as exc:
        msg = str(exc)
        assert "Empty model response" in msg
        assert "finish_reason" in msg
        assert "reasoning_tokens" in msg
    else:
        raise AssertionError("expected RuntimeError")
