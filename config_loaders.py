"""Load org / poison / experiment YAML configs (Phase 1)."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from interfaces import (
    AgentRoleConfig,
    Channel,
    EpisodeConfig,
    InjectionSpec,
    OrgConfig,
    PilotExperimentConfig,
    PoisonTemplate,
    RunType,
    TriggerTiming,
)
from paths import EXPERIMENTS_ROOT, ORGS_ROOT, POISONS_ROOT


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_org(name_or_path: Union[str, Path]) -> OrgConfig:
    path = Path(name_or_path)
    if not path.suffix:
        path = ORGS_ROOT / f"{name_or_path}.yaml"
    raw = _read_yaml(path)
    agents = [
        AgentRoleConfig(
            agent_id=str(a["agent_id"]),
            role=str(a["role"]),
            profile=str(a.get("profile", "")).strip(),
        )
        for a in raw.get("agents", [])
    ]
    relationships = [[str(x) for x in rel] for rel in raw.get("relationships", [])]
    return OrgConfig(
        name=str(raw.get("name", path.stem)),
        topology=str(raw["topology"]),
        coordinate_mode=str(raw.get("coordinate_mode", raw["topology"])),
        agents=agents,
        relationships=relationships,
        task=str(raw.get("task", "")).strip(),
        max_steps=int(raw.get("max_steps", 100)),
        include_defender=bool(raw.get("include_defender", False)),
        raw=raw,
    )


def load_poison(level_or_path: Union[int, str, Path]) -> PoisonTemplate:
    if isinstance(level_or_path, int):
        path = POISONS_ROOT / f"level_{level_or_path}.yaml"
    else:
        path = Path(level_or_path)
        if not path.suffix:
            path = POISONS_ROOT / f"{level_or_path}.yaml"
            if not path.exists():
                path = POISONS_ROOT / f"level_{level_or_path}.yaml"
    raw = _read_yaml(path)
    slots = {
        str(k): [str(v) for v in vals]
        for k, vals in (raw.get("slots") or {}).items()
    }
    return PoisonTemplate(
        level=int(raw["level"]),
        name=str(raw["name"]),
        template=str(raw["template"]).strip(),
        slots=slots,
        channel_default=str(raw.get("channel_default", Channel.CHAT.value)),
        notes=str(raw.get("notes", "")).strip(),
        verify=dict(raw.get("verify") or {}),
    )


def load_pilot_experiment(
    name_or_path: Union[str, Path] = "pilot",
) -> PilotExperimentConfig:
    path = Path(name_or_path)
    if not path.suffix:
        path = EXPERIMENTS_ROOT / f"{name_or_path}.yaml"
    raw = _read_yaml(path)
    org_by_topology = {
        str(k): str(v) for k, v in (raw.get("org_by_topology") or {}).items()
    }
    return PilotExperimentConfig(
        name=str(raw.get("name", path.stem)),
        org_by_topology=org_by_topology,
        poison_levels=[int(x) for x in raw["poison_levels"]],
        topologies=[str(x) for x in raw["topologies"]],
        budget_k=int(raw["budget_k"]),
        triggers=[str(x) for x in raw["triggers"]],
        seeds=[int(x) for x in raw["seeds"]],
        channel=str(raw.get("channel", Channel.CHAT.value)),
        target_role=str(raw.get("target_role", "builder")),
        results_dir=str(raw.get("results_dir", "results/pilot")),
        raw=raw,
    )


def choose_slot_fillers(poison: PoisonTemplate, seed: int) -> Dict[str, str]:
    """Pick slot fillers. Parallel slot lists (same length) share one index.

    Used so ``norm_id`` and ``risky_norm`` stay aligned for Level 2/4.
    """
    rng = random.Random(seed)
    fillers: Dict[str, str] = {}
    slots = poison.slots
    if not slots:
        return fillers

    # Group slots that should share an index (all lists of equal length > 1)
    lengths = {name: len(opts) for name, opts in slots.items() if opts}
    if not lengths:
        return fillers

    # If multiple slots share the same length, draw one index for that length group
    by_len: Dict[int, List[str]] = {}
    for name, n in lengths.items():
        by_len.setdefault(n, []).append(name)

    for n, names in by_len.items():
        idx = rng.randrange(n)
        for name in names:
            fillers[name] = slots[name][idx]
    return fillers


_BENIGN_PAD = (
    " Keep shared storage labeled, check the blueprint before placing, and "
    "flag layout conflicts to the coordinator so the settlement stays consistent."
)


def length_matched_benign(
    poison_text: str,
    benign: PoisonTemplate,
    fillers: Dict[str, str],
) -> str:
    """Instantiate level-0 content padded/trimmed to ``len(poison_text)`` (§2.4)."""
    base = benign.instantiate(fillers).rstrip()
    target_len = len(poison_text)
    if len(base) == target_len:
        return base
    if len(base) > target_len:
        return base[:target_len]
    # Pad with neutral build guidance until lengths match exactly.
    out = base
    while len(out) < target_len:
        out += _BENIGN_PAD
    return out[:target_len]


def pair_digest(pair_id: str) -> str:
    return hashlib.sha1(pair_id.encode("utf-8")).hexdigest()[:10]


def default_injection_target(org: OrgConfig, target_role: str = "builder") -> str:
    """Pick injection target: prefer adversary's relationship neighbor with role."""
    adv = org.adversary_id
    neighbors: List[str] = []
    if adv:
        for rel in org.relationships:
            if len(rel) < 2:
                continue
            a, b = rel[0], rel[1]
            if a == adv:
                neighbors.append(b)
            elif b == adv:
                neighbors.append(a)
    role_ids = [a.agent_id for a in org.agents if a.role == target_role]
    for nid in neighbors:
        if nid in role_ids:
            return nid
    if role_ids:
        return role_ids[0]
    raise ValueError(f"No agent with role={target_role} in org {org.name}")


def build_episode_pair(
    pilot: PilotExperimentConfig,
    *,
    poison_level: int,
    topology: str,
    seed: int,
    trigger: str,
    budget_k: Optional[int] = None,
    mode: str = "scripted",
) -> tuple[EpisodeConfig, EpisodeConfig]:
    """Build seed-matched adversarial + level-0 clean EpisodeConfig pair."""
    k = int(budget_k if budget_k is not None else pilot.budget_k)
    org_stem = pilot.org_by_topology.get(topology, topology)
    org = load_org(org_stem)
    poison = load_poison(poison_level)
    benign = load_poison(0)
    fillers = choose_slot_fillers(poison, seed)
    # Keep benign filler keys overlapping when possible for length parity scaffolding.
    benign_fillers = choose_slot_fillers(benign, seed)
    for key, value in fillers.items():
        if key in benign.slots:
            benign_fillers[key] = value if value in benign.slots[key] else benign_fillers.get(
                key, benign.slots[key][0]
            )

    target = default_injection_target(org, pilot.target_role)
    inject_turns = [int(x) for x in (pilot.raw.get("inject_turns") or [k])]
    injection = InjectionSpec(
        budget_k=k,
        channel=pilot.channel,
        poison_level=poison_level,
        target_agent_id=target,
        inject_turns=inject_turns,
    )
    clean_injection = InjectionSpec(
        budget_k=k,
        channel=pilot.channel,
        poison_level=0,
        target_agent_id=target,
        inject_turns=inject_turns,
    )

    mode_tag = "llm" if mode == "llm" else "scripted"
    pair_id = (
        f"{pilot.name}|{mode_tag}|L{poison_level}|{topology}|k{k}|{trigger}|seed{seed}"
    )
    digest = pair_digest(pair_id)
    trigger_enum = TriggerTiming(trigger)

    poison_text = poison.instantiate(fillers)
    benign_text = length_matched_benign(poison_text, benign, benign_fillers)
    if len(benign_text) != len(poison_text):
        raise RuntimeError("level-0 content must match poison length exactly")

    adv = EpisodeConfig(
        episode_id=f"adv-{digest}",
        seed=seed,
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
        seed=seed,
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
