"""Mode A org simulator: scripted roles/topology/injection (no Mineflayer).

Implements §2.3 injection procedure:
1. Adversary gets budget k and one channel.
2. On designated turn(s), emits poison artifact to a topology neighbor.
3. Logs injection as ground-truth root (is_injection_event=True).
4. After turn k, adversary is removed from the registry.

Optional scripted contagion: agents who received the poison hash may forward
it once to an active neighbor (deterministic under episode.seed) so downstream
metrics/attribution have a multi-hop path without LLMs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from event_log.schema import hash_content, make_event, write_jsonl
from interfaces import EpisodeConfig, Event
from marble_ext.adversary import AdversaryController
from marble_ext.defender import Defender
from marble_ext.injector import InjectionRecord, Injector
from marble_ext.registry import AgentRegistry
from marble_ext.trigger import TriggerSchedule


@dataclass
class EpisodeResult:
    episode: EpisodeConfig
    events: List[Event]
    injection: Optional[InjectionRecord]
    adversary_removed: bool
    active_agents_final: List[str]
    poison_hash: str
    log_path: Optional[Path] = None
    carriers: List[str] = field(default_factory=list)

    def to_meta(self) -> Dict:
        return {
            "episode_id": self.episode.episode_id,
            "pair_id": self.episode.pair_id,
            "run_type": self.episode.run_type.value,
            "poison_level": self.episode.injection.poison_level,
            "budget_k": self.episode.injection.budget_k,
            "channel": self.episode.injection.channel,
            "target": self.episode.injection.target_agent_id,
            "poison_hash": self.poison_hash,
            "adversary_removed": self.adversary_removed,
            "injection_step": self.injection.event.step if self.injection else None,
            "carriers": self.carriers,
            "n_events": len(self.events),
            "log_path": str(self.log_path) if self.log_path else None,
        }


def run_mode_a_episode(
    episode: EpisodeConfig,
    *,
    enable_contagion: bool = True,
    max_contagion_forwards: int = 3,
    out_path: Optional[Path] = None,
    rng: Optional[random.Random] = None,
) -> EpisodeResult:
    org = episode.org
    registry = AgentRegistry.from_org(org, include_defender=org.include_defender)
    adv_id = org.adversary_id
    if not adv_id:
        raise ValueError("Org config must include an adversary role")

    adversary = AdversaryController.from_injection_spec(adv_id, episode.injection)
    injector = Injector(episode)
    poison_text = episode.instantiated_content()
    poison_hash = hash_content(poison_text)

    defender: Optional[Defender] = None
    if org.include_defender and registry.is_active("defender"):
        defender = Defender(agent_id="defender")

    rng = rng if rng is not None else random.Random(episode.seed)
    events: List[Event] = []
    step = 0

    def emit(event: Event) -> None:
        events.append(event)
        if defender:
            defender.observe(event)

    # Kickoff
    step += 1
    emit(
        make_event(
            episode_id=episode.episode_id,
            run_type=episode.run_type.value,
            step=step,
            agent="coordinator" if registry.is_active("coordinator") else registry.active_ids()[0],
            event_type="task_complete",
            action_payload={"note": "episode_start", "task": org.task[:120]},
            raw_content="episode_start",
        )
    )

    injection_record: Optional[InjectionRecord] = None
    infected: Set[str] = set()

    # Adversary participation for exactly k turns
    while adversary.is_active():
        step += 1
        will_inject = adversary.should_inject_on_next_turn()
        turn_index = adversary.consume_turn(registry)

        if will_inject:
            if not registry.is_active(episode.injection.target_agent_id):
                # Target must be active at injection time
                raise RuntimeError(
                    f"Injection target {episode.injection.target_agent_id} not active"
                )
            record = injector.emit(
                step=step,
                source=adv_id,
                turn_index=turn_index,
            )
            emit(record.event)
            injection_record = record
            infected.add(record.target)
        else:
            # Benign adversary chatter to the same target (uses budget)
            target = episode.injection.target_agent_id
            text = f"Coordinating on the settlement build (turn {turn_index})."
            emit(
                make_event(
                    episode_id=episode.episode_id,
                    run_type=episode.run_type.value,
                    step=step,
                    agent=adv_id,
                    event_type="chat",
                    channel_from=adv_id,
                    raw_content=text,
                    action_payload={"to": target, "text": text},
                    is_injection_event=False,
                    poison_level=None,
                )
            )

    assert adversary.removed
    assert not registry.is_active(adv_id)

    injection_step = injection_record.event.step if injection_record else step
    trigger = TriggerSchedule(timing=episode.trigger, base_step=injection_step)

    # Scripted contagion / normal activity after adversary leaves
    forwards = 0
    pending = list(infected)
    carriers = set(infected)

    # Run until max_steps or contagion+trigger window covered
    end_step = max(org.max_steps, injection_step + 65)
    while step < end_step:
        step += 1

        # Activation trigger if scheduled
        trig = trigger.maybe_emit(
            episode_id=episode.episode_id,
            run_type=episode.run_type.value,
            step=step,
        )
        if trig is not None:
            emit(trig)

        # Contagion forward
        if enable_contagion and pending and forwards < max_contagion_forwards:
            src = pending.pop(0)
            if registry.is_active(src):
                neighbors = [
                    n
                    for n in registry.neighbors(org, src)
                    if n not in carriers and n != adv_id
                ]
                if neighbors:
                    dst = neighbors[rng.randrange(len(neighbors))]
                    emit(
                        make_event(
                            episode_id=episode.episode_id,
                            run_type=episode.run_type.value,
                            step=step,
                            agent=src,
                            event_type="chat",
                            channel_from=src,
                            raw_content=poison_text,
                            action_payload={"to": dst, "text": poison_text},
                            is_injection_event=False,
                            poison_level=episode.injection.poison_level
                            if episode.injection.poison_level > 0
                            else None,
                        )
                    )
                    carriers.add(dst)
                    pending.append(dst)
                    forwards += 1
                    continue

        # Background builder/architect activity (deterministic noise)
        actives = [a for a in registry.active_ids() if a != "defender"]
        if not actives:
            break
        agent = actives[rng.randrange(len(actives))]
        role = registry.role(agent)
        if role == "architect" and rng.random() < 0.3:
            text = "Update blueprint: reinforce north wall."
            emit(
                make_event(
                    episode_id=episode.episode_id,
                    run_type=episode.run_type.value,
                    step=step,
                    agent=agent,
                    event_type="blueprint_edit",
                    raw_content=text,
                    action_payload={"text": text},
                )
            )
        elif role == "builder":
            emit(
                make_event(
                    episode_id=episode.episode_id,
                    run_type=episode.run_type.value,
                    step=step,
                    agent=agent,
                    event_type="block_place",
                    action_payload={
                        "block": "cobblestone",
                        "pos": [rng.randint(0, 20), 64, rng.randint(0, 20)],
                    },
                )
            )
        else:
            emit(
                make_event(
                    episode_id=episode.episode_id,
                    run_type=episode.run_type.value,
                    step=step,
                    agent=agent,
                    event_type="task_complete",
                    action_payload={"note": "status_check"},
                    raw_content="status_check",
                )
            )

    result = EpisodeResult(
        episode=episode,
        events=events,
        injection=injection_record,
        adversary_removed=adversary.removed,
        active_agents_final=registry.active_ids(),
        poison_hash=poison_hash,
        carriers=sorted(carriers),
    )
    if out_path is not None:
        result.log_path = write_jsonl(out_path, events)
    return result
