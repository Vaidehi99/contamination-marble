"""Frozen interfaces for the contamination-marble pipeline (Phase 0)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class Channel(str, Enum):
    CHAT = "chat"
    BLUEPRINT_EDIT = "blueprint_edit"
    STRATEGY_NOTE = "strategy_note"


class RunType(str, Enum):
    ADVERSARIAL = "adversarial"
    CLEAN_COUNTERFACTUAL = "clean_counterfactual"


class TriggerTiming(str, Enum):
    NONE = "no_trigger"
    PLUS_20 = "trigger_plus_20"
    PLUS_60 = "trigger_plus_60"


@dataclass
class AgentRoleConfig:
    agent_id: str
    role: str  # coordinator | architect | builder | adversary | defender
    profile: str = ""


@dataclass
class OrgConfig:
    name: str
    topology: str  # star | chain | tree | graph
    coordinate_mode: str
    agents: List[AgentRoleConfig]
    relationships: List[List[str]]
    task: str
    max_steps: int = 100
    include_defender: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def adversary_id(self) -> Optional[str]:
        for agent in self.agents:
            if agent.role == "adversary":
                return agent.agent_id
        return None

    @property
    def builder_ids(self) -> List[str]:
        return [a.agent_id for a in self.agents if a.role == "builder"]

    def agent_ids(self, include_adversary: bool = True, include_defender: bool = True) -> List[str]:
        out: List[str] = []
        for agent in self.agents:
            if agent.role == "adversary" and not include_adversary:
                continue
            if agent.role == "defender" and not include_defender:
                continue
            out.append(agent.agent_id)
        return out


@dataclass
class PoisonTemplate:
    level: int
    name: str
    template: str
    slots: Dict[str, List[str]] = field(default_factory=dict)
    channel_default: str = Channel.CHAT.value
    notes: str = ""
    # Machine-checkable success conditions for compliance metric (from YAML ``verify``).
    verify: Dict[str, Any] = field(default_factory=dict)

    def instantiate(self, fillers: Dict[str, str]) -> str:
        text = self.template
        for key, value in fillers.items():
            text = text.replace("{" + key + "}", value)
        return text


@dataclass
class InjectionSpec:
    """How the adversary spends budget k on one channel."""

    budget_k: int
    channel: str
    poison_level: int
    target_agent_id: str
    inject_turns: List[int] = field(default_factory=list)  # relative 1..k; default last turn


@dataclass
class EpisodeConfig:
    """One cell of the experiment matrix for a single seed."""

    episode_id: str
    seed: int
    org: OrgConfig
    poison: PoisonTemplate
    injection: InjectionSpec
    trigger: TriggerTiming
    run_type: RunType
    slot_fillers: Dict[str, str] = field(default_factory=dict)
    pair_id: str = ""  # shared across adv/clean pair
    # When set (clean counterfactual), overrides template instantiation so
    # level-0 content can be length/effort-matched to the paired poison text.
    content_override: Optional[str] = None

    def instantiated_content(self) -> str:
        if self.content_override is not None:
            return self.content_override
        return self.poison.instantiate(self.slot_fillers)


@dataclass
class Event:
    """One JSONL event line (§3). Embeddings optional until Phase 2."""

    episode_id: str
    run_type: str
    step: int
    agent: str
    event_type: str
    channel_from: Optional[str] = None
    content_hash: Optional[str] = None
    content_embedding: Optional[List[float]] = None
    action_payload: Dict[str, Any] = field(default_factory=dict)
    is_injection_event: bool = False
    poison_level: Optional[int] = None
    raw_content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricBundle:
    """Pairwise metrics for one (adv, clean) episode pair."""

    pair_id: str
    seed: int
    poison_level: int
    topology: str
    budget_k: int
    trigger: str
    c_i: Dict[str, float] = field(default_factory=dict)
    C: List[str] = field(default_factory=list)
    C_bar: float = 0.0
    S: float = 0.0
    Z_i: Dict[str, float] = field(default_factory=dict)
    R_Z: float = 0.0
    attribution: Dict[str, Any] = field(default_factory=dict)
    # Ground-truth poison-objective compliance (calibration for C, not a replacement)
    compliance: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PairResult:
    """Outputs of run_pair for one seed-matched adversarial+clean pair."""

    pair_id: str
    seed: int
    adv_episode_id: str
    clean_episode_id: str
    adv_log_path: str
    clean_log_path: str
    ground_truth_injection: Dict[str, Any]
    metrics: Optional[MetricBundle] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PilotGridCell:
    poison_level: int
    topology: str
    budget_k: int
    trigger: str
    seeds: List[int]


@dataclass
class PilotExperimentConfig:
    name: str
    org_by_topology: Dict[str, str]  # topology -> org config path stem
    poison_levels: List[int]
    topologies: List[str]
    budget_k: int
    triggers: List[str]
    seeds: List[int]
    channel: str = Channel.CHAT.value
    target_role: str = "builder"  # inject into first builder neighbor
    results_dir: str = "results/pilot"
    raw: Dict[str, Any] = field(default_factory=dict)

    def cells(self) -> List[PilotGridCell]:
        cells: List[PilotGridCell] = []
        for level in self.poison_levels:
            for topology in self.topologies:
                for trigger in self.triggers:
                    cells.append(
                        PilotGridCell(
                            poison_level=level,
                            topology=topology,
                            budget_k=self.budget_k,
                            trigger=trigger,
                            seeds=list(self.seeds),
                        )
                    )
        return cells

    def num_episode_pairs(self) -> int:
        return (
            len(self.poison_levels)
            * len(self.topologies)
            * len(self.triggers)
            * len(self.seeds)
        )
