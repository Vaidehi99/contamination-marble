"""Activation-trigger scheduling for zombieness evaluations (§4.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from event_log.schema import make_event
from interfaces import Event, TriggerTiming


@dataclass
class TriggerSchedule:
    timing: TriggerTiming
    base_step: int  # step after which dormancy window starts (usually post-injection)

    @property
    def trigger_step(self) -> Optional[int]:
        if self.timing == TriggerTiming.NONE:
            return None
        if self.timing == TriggerTiming.PLUS_20:
            return self.base_step + 20
        if self.timing == TriggerTiming.PLUS_60:
            return self.base_step + 60
        return None

    def maybe_emit(
        self,
        *,
        episode_id: str,
        run_type: str,
        step: int,
        agent: str = "coordinator",
    ) -> Optional[Event]:
        if self.trigger_step is None or step != self.trigger_step:
            return None
        return make_event(
            episode_id=episode_id,
            run_type=run_type,
            step=step,
            agent=agent,
            event_type="task_complete",
            action_payload={
                "trigger": self.timing.value,
                "note": f"activation trigger {self.timing.value}",
            },
            raw_content=f"TRIGGER:{self.timing.value}",
        )


def parse_trigger(value: str) -> TriggerTiming:
    return TriggerTiming(value)
