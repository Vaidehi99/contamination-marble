"""Attribution metrics for recovered source/path (§4.5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from event_log.graph import (
    InteractionGraph,
    build_interaction_graph,
    recover_path_from_detected,
)
from interfaces import Event


@dataclass
class AttributionResult:
    detected: str
    true_source: str
    predicted_source: Optional[str]
    true_path: List[str]
    predicted_path: List[str]
    source_accuracy: float
    path_edit_distance: int
    path_accuracy: float
    latent_carrier_recall: float
    false_attribution: float
    source_absent: bool
    time_to_containment: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "detected": self.detected,
            "true_source": self.true_source,
            "predicted_source": self.predicted_source,
            "true_path": self.true_path,
            "predicted_path": self.predicted_path,
            "source_accuracy": self.source_accuracy,
            "path_edit_distance": self.path_edit_distance,
            "path_accuracy": self.path_accuracy,
            "latent_carrier_recall": self.latent_carrier_recall,
            "false_attribution": self.false_attribution,
            "source_absent": self.source_absent,
            "time_to_containment": self.time_to_containment,
            "notes": self.notes,
        }


def path_edit_distance(predicted: Sequence[str], truth: Sequence[str]) -> int:
    """Unit path edit distance (insert/delete/substitute) between node sequences."""
    n, m = len(predicted), len(truth)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if predicted[i - 1] == truth[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def evaluate_attribution(
    *,
    predicted_path: Sequence[str],
    true_path: Sequence[str],
    detected: str,
    true_source: str,
    clean_agents: Optional[Set[str]] = None,
    source_absent: bool = True,
    detection_step: Optional[int] = None,
    remediation_step: Optional[int] = None,
) -> AttributionResult:
    predicted_path = list(predicted_path)
    true_path = list(true_path)
    pred_source = predicted_path[0] if predicted_path else None
    source_acc = 1.0 if pred_source == true_source else 0.0

    ped = path_edit_distance(predicted_path, true_path)
    denom = max(len(predicted_path), len(true_path), 1)
    path_acc = 1.0 - (ped / denom)

    intermediates = set(true_path[1:-1]) if len(true_path) >= 3 else set()
    if not intermediates:
        latent_recall = 1.0
    else:
        latent_recall = len(intermediates & set(predicted_path)) / len(intermediates)

    clean = set(clean_agents or ())
    false_attr = (
        1.0 if pred_source is not None and pred_source in clean else 0.0
    )

    ttc = None
    if detection_step is not None and remediation_step is not None:
        ttc = max(0, remediation_step - detection_step)

    return AttributionResult(
        detected=detected,
        true_source=true_source,
        predicted_source=pred_source,
        true_path=true_path,
        predicted_path=predicted_path,
        source_accuracy=source_acc,
        path_edit_distance=ped,
        path_accuracy=path_acc,
        latent_carrier_recall=latent_recall,
        false_attribution=false_attr,
        source_absent=source_absent,
        time_to_containment=ttc,
    )


def attribute_from_events(
    events: Sequence[Event],
    *,
    detected: str,
    true_path: Sequence[str],
    poison_hash: str,
    clean_agents: Optional[Set[str]] = None,
    source_absent: bool = True,
) -> AttributionResult:
    graph: InteractionGraph = build_interaction_graph(events)
    predicted = recover_path_from_detected(
        graph, detected, prefer_hashes={poison_hash}
    )
    true_source = true_path[0] if true_path else ""
    return evaluate_attribution(
        predicted_path=predicted,
        true_path=true_path,
        detected=detected,
        true_source=true_source,
        clean_agents=clean_agents,
        source_absent=source_absent,
    )
