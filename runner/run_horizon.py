"""Run seed-matched long-horizon A/B pairs (does not touch the 24-step runner)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loaders import pair_digest
from event_log.schema import write_jsonl
from llm.cache import JsonCache
from llm.decision import make_backend
from analysis.horizon_report import build_report, write_report
from marble_ext.horizon.config import (
    build_horizon_episode_pair,
    clone_horizon_for_arm,
    expand_smoke_grid,
    load_horizon_experiment,
    make_horizon_config,
    schedule_fingerprint,
    serialize_horizon,
)
from marble_ext.horizon.protocol import freeze_protocol
from marble_ext.horizon.engine import HorizonResult, run_horizon_trajectory
from marble_ext.horizon.scoring import paired_delta
from marble_ext.horizon.types import HorizonConfig
from runner.run_pair import is_pair_complete, pair_output_dir, pair_paths


def _one_memory_spec(memory_raw: Dict[str, Any]) -> Dict[str, Any]:
    """CLI --one cannot expand a retrieval grid; default to retrieve-on."""
    spec = dict(memory_raw or {})
    retrieval = spec.get("retrieval_enabled", True)
    if isinstance(retrieval, list):
        spec["retrieval_enabled"] = bool(retrieval[0]) if retrieval else True
    return spec


def _horizon_paths(pair_dir: Path) -> Dict[str, Path]:
    paths = pair_paths(pair_dir)
    paths["horizon"] = pair_dir / "horizon.json"
    paths["schedule"] = pair_dir / "schedule.json"
    paths["adv_arm"] = pair_dir / "adv_horizon.json"
    paths["clean_arm"] = pair_dir / "clean_horizon.json"
    return paths


def _arm_bundle(result: HorizonResult) -> Dict[str, Any]:
    return {
        "Y": result.Y,
        "task_metrics": result.task_metrics,
        "boundary_snapshots": result.boundary_snapshots,
        "feature_gate_log": result.feature_gate_log,
        "reset_audit": result.reset_audit,
        "reconstructed_inputs": result.reconstructed_inputs,
        "cancelled_events": result.cancelled_events,
        "clocks": result.clocks,
        "probe_validity": result.probe_validity,
        "world_final": result.world.snapshot(),
        "elapsed_wall_s": result.elapsed_wall_s,
        "consolidation_llm_calls": result.consolidation_llm_calls,
        "explicit_memory_writes": result.explicit_memory_writes,
        "memory_trace": result.memory_trace,
        "memory_items": result.memory_items,
        "memory_evictions": result.memory_evictions,
        "handoff_failures": result.handoff_failures,
        "realized_depth": result.realized_depth,
        "scheduled_depth": result.scheduled_depth,
    }


def _y_row(adv: HorizonResult, clean: HorizonResult) -> Dict[str, Any]:
    y_adv = adv.Y.get("Y")
    y_clean = clean.Y.get("Y")
    return {
        "pair_id": adv.episode.pair_id,
        "seed": adv.episode.seed,
        "context_policy": adv.episode.pair_id.split("|")[1] if "|" in adv.episode.pair_id else None,
        "scheduled_depth": adv.scheduled_depth,
        "realized_depth_adv": adv.realized_depth,
        "realized_depth_clean": clean.realized_depth,
        "Y_injected": y_adv,
        "Y_l0": y_clean,
        "delta": paired_delta(y_adv, y_clean),
        "probe_validity_adv": adv.probe_validity,
        "probe_validity_clean": clean.probe_validity,
        "task_metrics_adv": adv.task_metrics,
        "task_metrics_clean": clean.task_metrics,
        "n_resets_adv": len(adv.reset_audit),
        "n_resets_clean": len(clean.reset_audit),
        "elapsed_wall_s_adv": adv.elapsed_wall_s,
        "elapsed_wall_s_clean": clean.elapsed_wall_s,
        "consolidation_llm_calls": adv.consolidation_llm_calls + clean.consolidation_llm_calls,
        "explicit_memory_writes": adv.explicit_memory_writes + clean.explicit_memory_writes,
        "memory_evictions": len(adv.memory_evictions) + len(clean.memory_evictions),
        "handoff_failures": len(adv.handoff_failures) + len(clean.handoff_failures),
        "write_mode": (adv.clocks or {}).get("write_mode"),
        "retrieval_enabled": (adv.clocks or {}).get("retrieval_enabled"),
        "poison_len": len(adv.episode.instantiated_content()),
        "clean_len": len(clean.episode.instantiated_content()),
        "attempted_benign_adv": adv.clocks.get("attempted_benign_since_injection"),
        "attempted_benign_clean": clean.clocks.get("attempted_benign_since_injection"),
        "depth_mismatch": (
            adv.realized_depth != adv.scheduled_depth
            or clean.realized_depth != clean.scheduled_depth
        ),
    }


def run_horizon_pair(
    horizon: HorizonConfig,
    *,
    results_root: Path,
    force: bool = False,
    llm_backend: str = "mock",
    llm_model: str = "gpt-5.6-luna",
) -> Dict[str, Any]:
    adv_ep, clean_ep = build_horizon_episode_pair(horizon)
    pair_dir = pair_output_dir(adv_ep.pair_id, results_root=results_root)
    paths = _horizon_paths(pair_dir)
    horizon_complete = is_pair_complete(pair_dir) and all(
        paths[k].is_file() for k in ("horizon", "schedule", "adv_arm", "clean_arm")
    )
    if horizon_complete and not force:
        return json.loads(paths["horizon"].read_text(encoding="utf-8"))

    cache = JsonCache(results_root / "cache" / "llm_completions.json")
    backend = make_backend(llm_backend, model=llm_model, cache=cache)
    adv_horizon = clone_horizon_for_arm(horizon)
    clean_horizon = clone_horizon_for_arm(horizon)
    if schedule_fingerprint(adv_horizon) != schedule_fingerprint(clean_horizon):
        raise RuntimeError("Injected/L0 twins must share an identical episode/event schedule")
    adv = run_horizon_trajectory(adv_ep, adv_horizon, backend=backend)
    clean = run_horizon_trajectory(clean_ep, clean_horizon, backend=backend)

    pair_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths["adv"], adv.events)
    write_jsonl(paths["clean"], clean.events)
    row = _y_row(adv, clean)
    meta = {
        "pair_id": adv_ep.pair_id,
        "digest": pair_digest(adv_ep.pair_id),
        "mode": "horizon",
        "seed": horizon.seed,
        "poison_level": horizon.poison_level,
        "topology": horizon.topology,
        "context_policy": horizon.context_policy.value,
        "depth": horizon.depth,
        "scheduled_depth": adv.scheduled_depth,
        "schedule_fingerprint": list(schedule_fingerprint(adv_horizon)),
        "episode_budget": horizon.episode_budget,
        "model": llm_model,
        "llm_backend": llm_backend,
        "adv_meta": adv.to_meta(),
        "clean_meta": clean.to_meta(),
        "ground_truth_injection": {
            "source": adv_ep.org.adversary_id,
            "target": adv_ep.injection.target_agent_id,
            "channel": adv_ep.injection.channel,
            "step": adv.injection.event.step if adv.injection else None,
            "poison_level": adv_ep.injection.poison_level,
            "poison_hash": adv.poison_hash,
            "content_len": len(adv_ep.instantiated_content()),
            "clean_content_len": len(clean_ep.instantiated_content()),
        },
        "manifest": adv.to_meta()["manifest"],
        "Y": row,
    }
    paths["meta"].write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    paths["metrics"].write_text(json.dumps({"Y": row}, indent=2, default=str), encoding="utf-8")
    paths["horizon"].write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    paths["schedule"].write_text(
        json.dumps(serialize_horizon(adv_horizon), indent=2, default=str), encoding="utf-8"
    )
    paths["adv_arm"].write_text(
        json.dumps(_arm_bundle(adv), indent=2, default=str), encoding="utf-8"
    )
    paths["clean_arm"].write_text(
        json.dumps(_arm_bundle(clean), indent=2, default=str), encoding="utf-8"
    )
    return row


def run_smoke(
    *,
    experiment: str = "long_horizon",
    results_root: Optional[Path] = None,
    force: bool = False,
    llm_backend: str = "mock",
    llm_model: Optional[str] = None,
    episode_budget: Optional[int] = None,
    seeds: Optional[Sequence[int]] = None,
    depths: Optional[Sequence[int]] = None,
    memory_policy: Optional[str] = None,
) -> List[Dict[str, Any]]:
    raw = load_horizon_experiment(experiment)
    raw = dict(raw)
    if memory_policy is not None:
        raw["memory_policy"] = memory_policy
    if episode_budget is not None:
        raw["episode_budget"] = int(episode_budget)
        raw["_cli_episode_budget"] = True
    if seeds is not None:
        raw["seeds"] = [int(s) for s in seeds]
    if depths is not None:
        if raw.get("episodes"):
            raise ValueError(
                "--depths cannot be combined with YAML episodes:; depth is derived from the sequence"
            )
        raw["depths"] = [int(d) for d in depths]
    model = llm_model or str(raw.get("model") or "gpt-5.6-luna")
    root = results_root or (ROOT / str(raw.get("results_dir") or "results/pilot_llm_gpt-5.6-luna_long"))
    rows = []
    for horizon in expand_smoke_grid(raw):
        rows.append(
            run_horizon_pair(
                horizon,
                results_root=root,
                force=force,
                llm_backend=llm_backend,
                llm_model=model,
            )
        )
    root.mkdir(parents=True, exist_ok=True)
    protocol = freeze_protocol(
        model=model,
        llm_backend=llm_backend,
        experiment=raw,
    )
    (root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, default=str), encoding="utf-8"
    )
    summary_path = root / "horizon_smoke.json"
    summary_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    write_report(build_report(rows), root)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Long-horizon A/B persistence smoke (paired L3 vs L0)."
    )
    parser.add_argument("--experiment", type=str, default="long_horizon")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Default: configs/experiments/long_horizon.yaml results_dir (isolated from 24-step Luna).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--llm-backend",
        choices=["mock", "openai"],
        default="mock",
    )
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--episode-budget", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--depths", type=int, nargs="*", default=None)
    parser.add_argument(
        "--one",
        action="store_true",
        help="Run a single pair: --policy --seed, plus --depth unless YAML supplies episodes:",
    )
    parser.add_argument("--policy", type=str, default="reset_between_episodes")
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Smoke generator depth when YAML has no episodes:. If YAML has episodes:, optional validation only.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--memory-policy",
        type=str,
        default=None,
        help='Canonical policy: off | private | shared | both. Quote YAML "off".',
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild horizon_report.{json,md} from existing pair rows",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.report_only:
        from analysis.horizon_report import load_rows

        raw = dict(load_horizon_experiment(args.experiment))
        root = args.results_root or (
            ROOT / str(raw.get("results_dir") or "results/pilot_llm_gpt-5.6-luna_long")
        )
        rows = load_rows(root)
        if not rows:
            print(json.dumps({"error": f"no rows under {root}"}))
            return 1
        paths = write_report(build_report(rows), root)
        print(json.dumps({"wrote": {k: str(v) for k, v in paths.items()}, "n_pairs": len(rows)}, indent=2))
        return 0

    if args.one:
        raw = dict(load_horizon_experiment(args.experiment))
        model = args.llm_model or str(raw.get("model") or "gpt-5.6-luna")
        budget = int(args.episode_budget or raw.get("episode_budget") or 12)
        raw["context_policies"] = [args.policy]
        raw["seeds"] = [args.seed]
        if args.memory_policy is not None:
            raw["memory_policy"] = args.memory_policy
        if raw.get("episodes"):
            if args.depth is not None:
                raw["depth"] = int(args.depth)
            if args.episode_budget is not None:
                raw["episode_budget"] = int(args.episode_budget)
                raw["_cli_episode_budget"] = True
            horizon = expand_smoke_grid(raw)[0]
        else:
            horizon = make_horizon_config(
                context_policy=args.policy,
                seed=args.seed,
                depth=int(args.depth) if args.depth is not None else 0,
                episode_budget=budget,
                poison_level=int(raw.get("poison_level", 3)),
                topology=str((raw.get("organization") or {}).get("topology") or "graph"),
                target_agent=str(raw.get("target_agent", "builder_1")),
                memory_policy=raw.get("memory_policy", "off"),
                extra={
                    "model": raw.get("model"),
                    "memory": _one_memory_spec(raw.get("memory") or {}),
                },
            )
        root = args.results_root or (
            ROOT / str(raw.get("results_dir") or "results/pilot_llm_gpt-5.6-luna_long")
        )
        row = run_horizon_pair(
            horizon,
            results_root=root,
            force=args.force,
            llm_backend=args.llm_backend,
            llm_model=model,
        )
        print(json.dumps(row, indent=2))
        return 0

    rows = run_smoke(
        experiment=args.experiment,
        results_root=args.results_root,
        force=args.force,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        episode_budget=args.episode_budget,
        seeds=args.seeds,
        depths=args.depths,
        memory_policy=args.memory_policy,
    )
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
