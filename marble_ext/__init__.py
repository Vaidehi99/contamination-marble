"""MARBLE extension package (Phase 4 Mode A)."""

from marble_ext.adversary import AdversaryController
from marble_ext.injector import Injector
from marble_ext.mode_a_sim import EpisodeResult, run_mode_a_episode
from marble_ext.registry import AgentRegistry

__all__ = [
    "AdversaryController",
    "AgentRegistry",
    "Injector",
    "EpisodeResult",
    "run_mode_a_episode",
]
