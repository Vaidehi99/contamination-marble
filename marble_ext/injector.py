"""Apply a poison (or benign) artifact onto a communication channel (§2.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from event_log.schema import make_event
from interfaces import Channel, EpisodeConfig, Event, InjectionSpec


@dataclass
class InjectionRecord:
    event: Event
    channel: str
    target: str
    turn_index: int  # adversary-relative 1..k


class Injector:
    """Build the ground-truth injection event for the designated channel."""

    def __init__(self, episode: EpisodeConfig) -> None:
        self.episode = episode
        self.spec: InjectionSpec = episode.injection
        self.content = episode.instantiated_content()

    @property
    def channel(self) -> str:
        return self.spec.channel or Channel.CHAT.value

    def emit(
        self,
        *,
        step: int,
        source: str,
        turn_index: int,
        target: Optional[str] = None,
    ) -> InjectionRecord:
        target_id = target or self.spec.target_agent_id
        channel = self.channel
        level = self.spec.poison_level
        run_type = self.episode.run_type.value
        episode_id = self.episode.episode_id

        if channel == Channel.CHAT.value:
            event = make_event(
                episode_id=episode_id,
                run_type=run_type,
                step=step,
                agent=source,
                event_type="chat",
                channel_from=source,
                raw_content=self.content,
                action_payload={"to": target_id, "text": self.content},
                is_injection_event=True,
                poison_level=level,
            )
        elif channel == Channel.BLUEPRINT_EDIT.value:
            event = make_event(
                episode_id=episode_id,
                run_type=run_type,
                step=step,
                agent=source,
                event_type="blueprint_edit",
                channel_from=source,
                raw_content=self.content,
                action_payload={"to": target_id, "text": self.content},
                is_injection_event=True,
                poison_level=level,
            )
        elif channel == Channel.STRATEGY_NOTE.value:
            event = make_event(
                episode_id=episode_id,
                run_type=run_type,
                step=step,
                agent=source,
                event_type="strategy_note",
                channel_from=source,
                raw_content=self.content,
                action_payload={"to": target_id, "text": self.content},
                is_injection_event=True,
                poison_level=level,
            )
        else:
            raise ValueError(f"Unsupported injection channel: {channel}")

        return InjectionRecord(
            event=event,
            channel=channel,
            target=target_id,
            turn_index=turn_index,
        )
