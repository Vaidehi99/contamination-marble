"""CLI: write a synthetic adv/clean pair for manual inspection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from event_log.graph import build_interaction_graph, ground_truth_path_from_roots
from event_log.synthetic import write_synthetic_pair
from paths import RESULTS_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit synthetic contamination pair logs")
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_ROOT / "synthetic" / "demo_pair",
    )
    args = parser.parse_args()
    bundle = write_synthetic_pair(args.out)
    graph = build_interaction_graph(bundle.adv_events)
    path = ground_truth_path_from_roots(
        graph, bundle.detected, bundle.poison_hash
    )
    print(json.dumps({**bundle.to_meta(), "recovered_path": path}, indent=2))
    print(f"match: {path == bundle.ground_truth_path}")
    return 0 if path == bundle.ground_truth_path else 1


if __name__ == "__main__":
    raise SystemExit(main())
