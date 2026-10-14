"""Long-horizon episode engine (thin adapters over Mode A)."""

from marble_ext.horizon.engine import HorizonResult, run_horizon_trajectory
from marble_ext.horizon.scoring import probe_episode_Y, paired_delta
from marble_ext.horizon.types import ContextPolicy, FeatureSet, HorizonConfig, MemoryPolicy

__all__ = [
    "ContextPolicy",
    "MemoryPolicy",
    "FeatureSet",
    "HorizonConfig",
    "HorizonResult",
    "run_horizon_trajectory",
    "probe_episode_Y",
    "paired_delta",
]
