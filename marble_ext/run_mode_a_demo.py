"""CLI: run one Mode A episode from the pilot config loaders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loaders import build_episode_pair, load_pilot_experiment
from event_log.graph import build_interaction_graph, recover_path_from_detected
from marble_ext.mode_a_sim import run_mode_a_episode
from paths import RESULTS_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Mode A contamination episode")
    parser.add_argument("--poison-level", type=int, default=2)
    parser.add_argument("--topology", type=str, default="graph")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trigger", type=str, default="no_trigger")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RESULTS_ROOT / "mode_a" / "demo",
    )
    args = parser.parse_args()

    pilot = load_pilot_experiment("pilot")
    adv, clean = build_episode_pair(
        pilot,
        poison_level=args.poison_level,
        topology=args.topology,
        seed=args.seed,
        trigger=args.trigger,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    adv_result = run_mode_a_episode(
        adv, out_path=args.out_dir / "adv.jsonl", enable_contagion=True
    )
    clean_result = run_mode_a_episode(
        clean, out_path=args.out_dir / "clean.jsonl", enable_contagion=True
    )

    graph = build_interaction_graph(adv_result.events)
    detected = adv_result.carriers[-1] if adv_result.carriers else adv.injection.target_agent_id
    path = recover_path_from_detected(
        graph, detected, prefer_hashes={adv_result.poison_hash}
    )

    summary = {
        "adv": adv_result.to_meta(),
        "clean": clean_result.to_meta(),
        "adversary_in_final_active": adv.org.adversary_id in adv_result.active_agents_final,
        "recovered_path_from": detected,
        "recovered_path": path,
    }
    print(json.dumps(summary, indent=2))
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    ok = (
        adv_result.injection is not None
        and adv_result.adversary_removed
        and adv.org.adversary_id not in adv_result.active_agents_final
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
