"""Metrics package (§4)."""

from metrics.behavior_distance import behavior_distance, pairwise_agent_distances
from metrics.compute import compute_metric_bundle
from metrics.control import compute_control
from metrics.spread import compute_spread

__all__ = [
    "behavior_distance",
    "pairwise_agent_distances",
    "compute_control",
    "compute_spread",
    "compute_metric_bundle",
]
