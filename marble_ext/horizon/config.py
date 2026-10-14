"""Load long-horizon YAML and expand the A/B × depth × seed smoke grid."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from config_loaders import (
    choose_slot_fillers,
    default_injection_target,
    length_matched_benign,
    load_org,
    load_poison,
    pair_digest,
)
from interfaces import EpisodeConfig, InjectionSpec, RunType, TriggerTiming
from marble_ext.horizon.memory import parse_memory_policy
from marble_ext.horizon.tasks import TASK_REGISTRY
from marble_ext.horizon.types import (
    ContextPolicy,
    EpisodeSpec,
    FeatureSet,
    HorizonConfig,
    MemoryPolicy,
    ScheduledEvent,
    annotate_inject_probe,
    default_episode_specs,
    default_events,
    scheduled_depth_from_sequence,
    validate_schedule,
)
from paths import EXPERIMENTS_ROOT


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_horizon_experiment(
    name_or_path: Union[str, Path] = "long_horizon",
) -> Dict[str, Any]:
    path = Path(name_or_path)
    if not path.suffix:
        path = EXPERIMENTS_ROOT / f"{name_or_path}.yaml"
    return _read_yaml(path)


def _policy(raw: str) -> ContextPolicy:
    text = str(raw).strip().lower()
    if text in {"a", "continuous", "condition_a"}:
        return ContextPolicy.CONTINUOUS
    if text in {"b", "reset_between_episodes", "reset", "condition_b"}:
        return ContextPolicy.RESET_BETWEEN_EPISODES
    raise ValueError(f"Unknown context_policy {raw!r}")


def _features(raw: Optional[Dict[str, Any]]) -> FeatureSet:
    raw = raw or {}
    return FeatureSet(
        chest_fork=bool(raw.get("chest_fork", False)),
        scarcity=bool(raw.get("scarcity", False)),
        construction=bool(raw.get("construction", False)),
        agenda=bool(raw.get("agenda", False)),
    )


def parse_episode_specs(raw_episodes: List[Dict[str, Any]]) -> List[EpisodeSpec]:
    specs: List[EpisodeSpec] = []
    for row in raw_episodes:
        specs.append(
            EpisodeSpec(
                episode_id=str(row["id"]),
                task_id=str(row["task"]),
                budget=int(row.get("budget", 12)),
                features=_features(row.get("features")),
                params=dict(row.get("params") or {}),
                inject=bool(row.get("inject", False)),
                probe=bool(row.get("probe", False)),
            )
        )
    return specs


def parse_events(raw_events: List[Dict[str, Any]]) -> List[ScheduledEvent]:
    out: List[ScheduledEvent] = []
    for seq, row in enumerate(raw_events):
        at = row.get("at") or {}
        out.append(
            ScheduledEvent(
                type=str(row["type"]),
                episode_id=str(at.get("episode") or row.get("episode")),
                episode_step=int(at.get("step", row.get("step", 0))),
                payload={
                    k: v
                    for k, v in row.items()
                    if k not in {"type", "at", "episode", "step"}
                },
                seq=seq,
            )
        )
    return out


def _require_registered_tasks(specs: List[EpisodeSpec]) -> None:
    missing = sorted({s.task_id for s in specs if s.task_id not in TASK_REGISTRY})
    if missing:
        raise KeyError(
            f"Unknown task id(s) {missing}. Register an adapter before using them in YAML."
        )


def make_horizon_config(
    *,
    context_policy: Union[str, ContextPolicy],
    seed: int,
    depth: Optional[int] = None,
    episode_budget: int = 12,
    poison_level: int = 3,
    topology: str = "graph",
    target_agent: str = "builder_1",
    memory_policy: Union[str, MemoryPolicy, bool, None] = None,
    episodes: Optional[List[EpisodeSpec]] = None,
    events: Optional[List[ScheduledEvent]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> HorizonConfig:
    """Build a config whose ``depth`` is derived from the episode sequence.

    ``depth`` is a generator input when ``episodes`` is omitted (smoke d=0/2).
    If ``episodes`` is supplied, a non-None ``depth`` is only a validation check.
    """
    policy = (
        context_policy
        if isinstance(context_policy, ContextPolicy)
        else _policy(str(context_policy))
    )
    extra_dict = deepcopy(extra or {})
    mem = parse_memory_policy(
        memory_policy if memory_policy is not None else extra_dict.get("memory_policy", "off")
    )
    if policy == ContextPolicy.CONTINUOUS and mem != MemoryPolicy.OFF:
        raise ValueError(
            "Condition A (continuous context) is memory-off; "
            "C/D/E require episode-boundary resets"
        )
    if mem in {MemoryPolicy.SHARED, MemoryPolicy.BOTH}:
        raise NotImplementedError(
            f"memory_policy {mem.value!r} (D/E) is not implemented; run C (private) first"
        )
    extra_dict["memory_policy"] = mem.value
    if events is not None and episodes is None:
        raise ValueError("events: requires episodes:")
    if episodes is None:
        if depth is None:
            raise ValueError("depth is required when episodes are not supplied")
        specs = default_episode_specs(int(depth), episode_budget)
    else:
        specs = deepcopy(episodes)
    _require_registered_tasks(specs)
    sched = deepcopy(events) if events is not None else default_events(specs)
    annotate_inject_probe(specs, sched)
    validate_schedule(specs, sched)
    derived = scheduled_depth_from_sequence(specs, sched)
    if episodes is None and derived != int(depth):
        raise ValueError(
            f"Generated schedule has intervening depth {derived}, not requested {depth}"
        )
    if episodes is not None and depth is not None and derived != int(depth):
        raise ValueError(
            f"depth label {depth} does not match intervening episode count {derived} "
            f"(injection→probe sequence, not a free-form tag)"
        )
    return HorizonConfig(
        name="long_horizon",
        context_policy=policy,
        memory_policy=mem,
        depth=derived,
        seed=int(seed),
        topology=topology,
        poison_level=int(poison_level),
        episode_budget=int(episode_budget),
        target_agent=target_agent,
        events=sched,
        episodes=specs,
        extra=extra_dict,
    )


def _memory_grid(memory_raw: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand retrieval_enabled:[true,false] into one spec per trajectory."""
    memory = dict(memory_raw or {})
    retrieval = memory.get("retrieval_enabled", True)
    if isinstance(retrieval, list):
        if not retrieval:
            raise ValueError("memory.retrieval_enabled list must not be empty")
        variants = []
        for item in retrieval:
            row = dict(memory)
            row["retrieval_enabled"] = bool(item)
            variants.append(row)
        return variants
    return [memory]


def expand_smoke_grid(raw: Optional[Dict[str, Any]] = None) -> List[HorizonConfig]:
    raw = raw if raw is not None else load_horizon_experiment("long_horizon")
    policies = raw.get("context_policies") or [
        raw.get("context_policy", "reset_between_episodes")
    ]
    seeds = [int(s) for s in (raw.get("seeds") or [0, 1, 2])]
    budget = int(raw.get("episode_budget", 12))
    level = int(raw.get("poison_level", 3))
    topology = str((raw.get("organization") or {}).get("topology") or "graph")
    target = str(raw.get("target_agent", "builder_1"))
    mem = parse_memory_policy(raw.get("memory_policy", "off"))
    memory_variants = _memory_grid(raw.get("memory") or {})
    custom_episodes = raw.get("episodes")
    if raw.get("events") is not None and not custom_episodes:
        raise ValueError("YAML events: requires episodes:")

    out: List[HorizonConfig] = []
    if custom_episodes:
        specs = parse_episode_specs(list(custom_episodes))
        if raw.get("_cli_episode_budget"):
            for spec in specs:
                spec.budget = budget
        events = parse_events(list(raw["events"])) if raw.get("events") is not None else None
        expected = int(raw["depth"]) if "depth" in raw and raw["depth"] is not None else None
        for policy in policies:
            for seed in seeds:
                for mem_variant in memory_variants:
                    extra = {
                        "model": raw.get("model"),
                        "memory_policy": mem.value,
                        "memory": dict(mem_variant),
                    }
                    out.append(
                        make_horizon_config(
                            context_policy=policy,
                            seed=seed,
                            depth=expected,
                            episode_budget=budget,
                            poison_level=level,
                            topology=topology,
                            target_agent=target,
                            memory_policy=mem,
                            episodes=specs,
                            events=events,
                            extra=extra,
                        )
                    )
        return out

    depths = [int(d) for d in (raw.get("depths") or [0, 2])]
    for policy in policies:
        for depth in depths:
            for seed in seeds:
                for mem_variant in memory_variants:
                    extra = {
                        "model": raw.get("model"),
                        "memory_policy": mem.value,
                        "memory": dict(mem_variant),
                    }
                    out.append(
                        make_horizon_config(
                            context_policy=policy,
                            seed=seed,
                            depth=depth,
                            episode_budget=budget,
                            poison_level=level,
                            topology=topology,
                            target_agent=target,
                            memory_policy=mem,
                            extra=extra,
                        )
                    )
    return out


def schedule_fingerprint(horizon: HorizonConfig) -> Tuple[Any, Any]:
    """Payload-free schedule identity shared by injected/L0 twins."""
    episodes = tuple(
        (
            s.episode_id,
            s.task_id,
            s.budget,
            tuple(sorted(s.features.to_dict().items())),
            s.inject,
            s.probe,
            tuple(sorted(s.params.items())),
        )
        for s in horizon.episodes
    )
    events = tuple(
        (
            e.type,
            e.episode_id,
            e.episode_step,
            e.seq,
            json.dumps(e.payload, sort_keys=True, default=str),
        )
        for e in horizon.events
    )
    return episodes, events


def clone_horizon_for_arm(horizon: HorizonConfig) -> HorizonConfig:
    """Identical schedule for the matched L0 arm (payload lives on EpisodeConfig)."""
    return replace(
        horizon,
        events=deepcopy(horizon.events),
        episodes=deepcopy(horizon.episodes),
        extra=deepcopy(horizon.extra),
    )


def serialize_horizon(horizon: HorizonConfig) -> Dict[str, Any]:
    """Frozen schedule for a pair directory (evaluator-only)."""
    return {
        "name": horizon.name,
        "context_policy": horizon.context_policy.value,
        "memory_policy": horizon.memory_policy.value,
        "depth": horizon.depth,
        "seed": horizon.seed,
        "topology": horizon.topology,
        "poison_level": horizon.poison_level,
        "episode_budget": horizon.episode_budget,
        "target_agent": horizon.target_agent,
        "extra": dict(horizon.extra),
        "episodes": [
            {
                "id": s.episode_id,
                "task": s.task_id,
                "budget": s.budget,
                "features": s.features.to_dict(),
                "params": dict(s.params),
                "inject": s.inject,
                "probe": s.probe,
            }
            for s in horizon.episodes
        ],
        "events": [
            {
                "type": e.type,
                "episode": e.episode_id,
                "step": e.episode_step,
                "seq": e.seq,
                "payload": dict(e.payload),
            }
            for e in horizon.events
        ],
    }


def build_horizon_episode_pair(
    horizon: HorizonConfig,
    *,
    trigger: str = "no_trigger",
    budget_k: int = 1,
    inject_turns: Optional[List[int]] = None,
) -> Tuple[EpisodeConfig, EpisodeConfig]:
    """Seed-matched L{level} vs length-matched L0 pair. pair_id includes A/B and depth."""
    org = load_org(horizon.topology)
    poison = load_poison(horizon.poison_level)
    benign = load_poison(0)
    fillers = choose_slot_fillers(poison, horizon.seed)
    benign_fillers = choose_slot_fillers(benign, horizon.seed)
    for key, value in fillers.items():
        if key in benign.slots:
            benign_fillers[key] = (
                value if value in benign.slots[key] else benign_fillers.get(key, benign.slots[key][0])
            )
    target = default_injection_target(org, "builder")
    turns = list(inject_turns if inject_turns is not None else [1])
    injection = InjectionSpec(
        budget_k=int(budget_k),
        channel="chat",
        poison_level=horizon.poison_level,
        target_agent_id=target,
        inject_turns=turns,
    )
    clean_injection = InjectionSpec(
        budget_k=int(budget_k),
        channel="chat",
        poison_level=0,
        target_agent_id=target,
        inject_turns=turns,
    )
    pair_id = (
        f"{horizon.pair_tag}|llm|L{horizon.poison_level}|{horizon.topology}|"
        f"k{budget_k}|{trigger}|seed{horizon.seed}"
    )
    digest = pair_digest(pair_id)
    poison_text = poison.instantiate(fillers)
    benign_text = length_matched_benign(poison_text, benign, benign_fillers)
    if len(benign_text) != len(poison_text):
        raise RuntimeError("level-0 content must match poison length exactly")
    trigger_enum = TriggerTiming(trigger)
    adv = EpisodeConfig(
        episode_id=f"adv-{digest}",
        seed=horizon.seed,
        org=org,
        poison=poison,
        injection=injection,
        trigger=trigger_enum,
        run_type=RunType.ADVERSARIAL,
        slot_fillers=fillers,
        pair_id=pair_id,
    )
    clean = EpisodeConfig(
        episode_id=f"clean-{digest}",
        seed=horizon.seed,
        org=org,
        poison=benign,
        injection=clean_injection,
        trigger=trigger_enum,
        run_type=RunType.CLEAN_COUNTERFACTUAL,
        slot_fillers=benign_fillers,
        pair_id=pair_id,
        content_override=benign_text,
    )
    return adv, clean
