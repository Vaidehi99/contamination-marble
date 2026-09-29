"""Seed-matching checks for adversarial vs clean counterfactual traces (§2.4).

The two runs share seed/task/topology/turn order. Content may diverge only at
the injection turn (and subsequent contagion forwards that reuse that content).
Pre-injection events must be bit-identical after stripping run metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from interfaces import Event


_META_FIELDS = ("episode_id", "run_type")


@dataclass
class SeedMatchReport:
    injection_step: int
    n_pre_injection: int
    n_events: int
    structural_match: bool
    content_divergence_steps: List[int]


def _payload_structural(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Drop free-text fields that legitimately differ after poison substitution."""
    out = dict(payload or {})
    out.pop("text", None)
    return out


def _pre_injection_dict(event: Event) -> Dict[str, Any]:
    data = event.to_dict()
    for key in _META_FIELDS:
        data.pop(key, None)
    return data


def _structural_tuple(event: Event) -> Tuple:
    return (
        event.step,
        event.agent,
        event.event_type,
        event.channel_from,
        event.is_injection_event,
        tuple(sorted((k, repr(v)) for k, v in _payload_structural(event.action_payload).items())),
    )


def find_injection_step(events: Sequence[Event]) -> Optional[int]:
    for event in events:
        if event.is_injection_event:
            return event.step
    return None


def assert_seed_matched_pair(
    adv_events: Sequence[Event],
    clean_events: Sequence[Event],
    *,
    require_post_structural: bool = True,
) -> SeedMatchReport:
    """Assert §2.4 seed-matching; raise AssertionError on violation.

    - Same length and step sequence (when require_post_structural)
    - Events before injection: bit-identical after stripping episode_id/run_type
    - Injection step and later (scripted mode): identical structural turn order;
      content/hash/poison_level may differ
    - LLM mode: set require_post_structural=False — post-injection action choices
      may legitimately diverge (that divergence is what c_i measures)
    """
    if not adv_events or not clean_events:
        raise AssertionError("Empty event logs")

    inj_adv = find_injection_step(adv_events)
    inj_clean = find_injection_step(clean_events)
    if inj_adv is None or inj_clean is None:
        raise AssertionError("Both runs must contain an injection event")
    if inj_adv != inj_clean:
        raise AssertionError(
            f"Injection step mismatch: adv={inj_adv} clean={inj_clean}"
        )

    adv_pre = [e for e in adv_events if e.step < inj_adv]
    clean_pre = [e for e in clean_events if e.step < inj_adv]
    if len(adv_pre) != len(clean_pre):
        raise AssertionError(
            f"Pre-injection length mismatch: adv={len(adv_pre)} clean={len(clean_pre)}"
        )
    for adv_ev, clean_ev in zip(adv_pre, clean_pre):
        if _pre_injection_dict(adv_ev) != _pre_injection_dict(clean_ev):
            raise AssertionError(
                f"Pre-injection bit mismatch at step={adv_ev.step} "
                f"agent={adv_ev.agent} type={adv_ev.event_type}"
            )

    content_divergence_steps: List[int] = []
    if require_post_structural:
        if len(adv_events) != len(clean_events):
            raise AssertionError(
                f"Seed-matched pair length mismatch: adv={len(adv_events)} clean={len(clean_events)}"
            )
        for adv_ev, clean_ev in zip(adv_events, clean_events):
            if adv_ev.step != clean_ev.step:
                raise AssertionError(
                    f"Step mismatch at aligned index: adv={adv_ev.step} clean={clean_ev.step}"
                )
            if adv_ev.step >= inj_adv:
                if _structural_tuple(adv_ev) != _structural_tuple(clean_ev):
                    raise AssertionError(
                        f"Post-injection structural mismatch at step={adv_ev.step} "
                        f"adv=({adv_ev.agent},{adv_ev.event_type}) "
                        f"clean=({clean_ev.agent},{clean_ev.event_type})"
                    )
                adv_text = adv_ev.raw_content or ""
                clean_text = clean_ev.raw_content or ""
                if adv_text != clean_text or adv_ev.content_hash != clean_ev.content_hash:
                    content_divergence_steps.append(adv_ev.step)
    else:
        # LLM mode: only require content divergence at injection itself
        adv_inj = next(e for e in adv_events if e.is_injection_event)
        clean_inj = next(e for e in clean_events if e.is_injection_event)
        if (adv_inj.raw_content or "") == (clean_inj.raw_content or ""):
            raise AssertionError(
                "Expected content divergence at the injection turn "
                "(poison vs length-matched level-0)"
            )
        content_divergence_steps.append(inj_adv)

    if require_post_structural and inj_adv not in content_divergence_steps:
        raise AssertionError(
            "Expected content divergence at the injection turn "
            "(poison vs length-matched level-0)"
        )

    return SeedMatchReport(
        injection_step=inj_adv,
        n_pre_injection=len(adv_pre),
        n_events=len(adv_events),
        structural_match=require_post_structural,
        content_divergence_steps=content_divergence_steps,
    )
