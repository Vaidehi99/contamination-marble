"""Run one seed-matched adversarial + clean counterfactual pair (§2.4 / Phase 5).

Invariants:
1. Identical seed, task, topology, and turn order (asserted via seed_match).
2. Clean injection is length-matched level-0 for that poison instantiation.
3. Output layout ``results/pairs/{digest}/{adv,clean}.jsonl`` is resume-safe.
4. Metrics are computed pairwise only — never from a single unpaired log.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loaders import build_episode_pair, load_pilot_experiment, pair_digest
from event_log.graph import build_interaction_graph, recover_path_from_detected
from event_log.schema import injection_events, read_jsonl, write_jsonl
from interfaces import EpisodeConfig, Event, MetricBundle, PairResult, PilotExperimentConfig
from marble_ext.mode_a_sim import EpisodeResult, run_mode_a_episode
from metrics.compute import compute_metric_bundle
from paths import RESULTS_ROOT
from runner.seed_match import SeedMatchReport, assert_seed_matched_pair


def pair_output_dir(
    pair_id: str,
    *,
    results_root: Path = RESULTS_ROOT,
) -> Path:
    """``results/pairs/{digest}/`` — digest shared by adv-/clean- episode ids."""
    return results_root / "pairs" / pair_digest(pair_id)


def pair_paths(pair_dir: Path) -> Dict[str, Path]:
    return {
        "adv": pair_dir / "adv.jsonl",
        "clean": pair_dir / "clean.jsonl",
        "metrics": pair_dir / "metrics.json",
        "meta": pair_dir / "meta.json",
    }


def is_pair_complete(pair_dir: Path) -> bool:
    paths = pair_paths(pair_dir)
    return all(paths[k].is_file() for k in ("adv", "clean", "metrics", "meta"))


def _true_path_from_adv(
    adv_result: EpisodeResult,
) -> Tuple[str, List[str]]:
    """Recover ground-truth injection→…→detected path for attribution."""
    assert adv_result.injection is not None
    target = adv_result.injection.target
    graph = build_interaction_graph(adv_result.events)
    detected = adv_result.carriers[-1] if adv_result.carriers else target
    path = recover_path_from_detected(
        graph, detected, prefer_hashes={adv_result.poison_hash}
    )
    if not path:
        path = ["adversary", target]
        detected = target
    return detected, path


def metrics_from_pair_results(
    adv_episode: EpisodeConfig,
    adv_result: EpisodeResult,
    clean_result: EpisodeResult,
    *,
    tau_C: float = 0.15,
    tau_Z: float = 0.15,
    embed_backend: str = "auto",
) -> MetricBundle:
    """Sole entry point from the runner into metric computation (pairwise only)."""
    if adv_result.injection is None or clean_result.injection is None:
        raise ValueError("Both adv and clean results must include an injection record")

    # Prefer real embeddings for d_chat when available; hash fallback offline.
    from llm.embeddings import enrich_event_embeddings

    enrich_event_embeddings(adv_result.events, backend=embed_backend)
    enrich_event_embeddings(clean_result.events, backend=embed_backend)

    detected, true_path = _true_path_from_adv(adv_result)
    agent_ids = adv_episode.org.agent_ids(
        include_adversary=True, include_defender=True
    )
    return compute_metric_bundle(
        pair_id=adv_episode.pair_id,
        seed=adv_episode.seed,
        poison_level=adv_episode.injection.poison_level,
        topology=adv_episode.org.topology,
        budget_k=adv_episode.injection.budget_k,
        trigger=adv_episode.trigger.value,
        adv_events=adv_result.events,
        clean_events=clean_result.events,
        agent_ids=agent_ids,
        detected=detected,
        true_path=true_path,
        poison_hash=adv_result.poison_hash,
        injection_step=adv_result.injection.event.step,
        tau_C=tau_C,
        tau_Z=tau_Z,
    )


def _run_matched_episodes(
    adv: EpisodeConfig,
    clean: EpisodeConfig,
    *,
    enable_contagion: bool = True,
    mode: str = "scripted",
    llm_backend: str = "mock",
    llm_model: str = "gpt-4o-mini",
    max_steps: Optional[int] = None,
) -> Tuple[EpisodeResult, EpisodeResult, SeedMatchReport]:
    """Run both episodes seed-matched (scripted RNG or shared LLM cache)."""
    if adv.seed != clean.seed:
        raise ValueError(f"Seed mismatch: adv={adv.seed} clean={clean.seed}")
    if adv.org.task != clean.org.task or adv.org.topology != clean.org.topology:
        raise ValueError("Adv/clean must share task and topology")
    if len(adv.instantiated_content()) != len(clean.instantiated_content()):
        raise ValueError("Clean level-0 content must equal poison length")

    if mode == "llm":
        from llm.cache import JsonCache
        from llm.decision import make_backend
        from marble_ext.mode_a_llm import run_mode_a_llm_episode

        cache = JsonCache(RESULTS_ROOT / "cache" / "llm_completions.json")
        backend = make_backend(llm_backend, model=llm_model, cache=cache)
        # Shared backend+cache → identical completions for identical prompts
        # (guarantees bit-identical pre-injection trajectories).
        steps = max_steps if max_steps is not None else 24
        adv_result = run_mode_a_llm_episode(
            adv, backend=backend, max_steps=steps
        )
        clean_result = run_mode_a_llm_episode(
            clean, backend=backend, max_steps=steps
        )
        report = assert_seed_matched_pair(
            adv_result.events,
            clean_result.events,
            require_post_structural=False,
        )
        return adv_result, clean_result, report

    adv_result = run_mode_a_episode(
        adv,
        enable_contagion=enable_contagion,
        rng=random.Random(adv.seed),
    )
    clean_result = run_mode_a_episode(
        clean,
        enable_contagion=enable_contagion,
        rng=random.Random(clean.seed),
    )
    report = assert_seed_matched_pair(adv_result.events, clean_result.events)
    return adv_result, clean_result, report


def run_pair(
    *,
    poison_level: int,
    topology: str,
    seed: int,
    trigger: str = "no_trigger",
    pilot: Optional[PilotExperimentConfig] = None,
    results_root: Path = RESULTS_ROOT,
    force: bool = False,
    enable_contagion: bool = True,
    tau_C: float = 0.15,
    tau_Z: float = 0.15,
    mode: str = "scripted",
    llm_backend: str = "mock",
    llm_model: str = "gpt-4o-mini",
    max_steps: Optional[int] = None,
    embed_backend: str = "auto",
) -> PairResult:
    """Execute one seed-matched pair, write artifacts, return PairResult + metrics."""
    pilot = pilot or load_pilot_experiment("pilot")
    adv, clean = build_episode_pair(
        pilot,
        poison_level=poison_level,
        topology=topology,
        seed=seed,
        trigger=trigger,
        mode=mode,
    )
    pair_dir = pair_output_dir(adv.pair_id, results_root=results_root)
    paths = pair_paths(pair_dir)

    if is_pair_complete(pair_dir) and not force:
        return _load_pair_result(pair_dir)

    pair_dir.mkdir(parents=True, exist_ok=True)
    adv_result, clean_result, match_report = _run_matched_episodes(
        adv,
        clean,
        enable_contagion=enable_contagion,
        mode=mode,
        llm_backend=llm_backend,
        llm_model=llm_model,
        max_steps=max_steps,
    )

    write_jsonl(paths["adv"], adv_result.events)
    write_jsonl(paths["clean"], clean_result.events)
    adv_result.log_path = paths["adv"]
    clean_result.log_path = paths["clean"]

    metrics = metrics_from_pair_results(
        adv,
        adv_result,
        clean_result,
        tau_C=tau_C,
        tau_Z=tau_Z,
        embed_backend=embed_backend,
    )
    paths["metrics"].write_text(
        json.dumps(metrics.to_dict(), indent=2), encoding="utf-8"
    )

    inj = adv_result.injection
    assert inj is not None
    ground_truth = {
        "source": adv.org.adversary_id,
        "target": inj.target,
        "channel": inj.channel,
        "step": inj.event.step,
        "turn_index": inj.turn_index,
        "poison_level": adv.injection.poison_level,
        "poison_hash": adv_result.poison_hash,
        "content_len": len(adv.instantiated_content()),
        "clean_content_len": len(clean.instantiated_content()),
    }
    meta: Dict[str, Any] = {
        "pair_id": adv.pair_id,
        "digest": pair_digest(adv.pair_id),
        "mode": mode,
        "seed": seed,
        "poison_level": poison_level,
        "topology": topology,
        "trigger": trigger,
        "budget_k": adv.injection.budget_k,
        "adv_episode_id": adv.episode_id,
        "clean_episode_id": clean.episode_id,
        "ground_truth_injection": ground_truth,
        "seed_match": asdict(match_report),
        "adv_meta": adv_result.to_meta(),
        "clean_meta": clean_result.to_meta(),
        "adversary_in_final_active": {
            "adv": adv.org.adversary_id in adv_result.active_agents_final,
            "clean": clean.org.adversary_id in clean_result.active_agents_final,
        },
    }
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return PairResult(
        pair_id=adv.pair_id,
        seed=seed,
        adv_episode_id=adv.episode_id,
        clean_episode_id=clean.episode_id,
        adv_log_path=str(paths["adv"]),
        clean_log_path=str(paths["clean"]),
        ground_truth_injection=ground_truth,
        metrics=metrics,
    )


def _load_pair_result(pair_dir: Path) -> PairResult:
    paths = pair_paths(pair_dir)
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    metrics_raw = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    metrics = MetricBundle(**{k: metrics_raw[k] for k in MetricBundle.__dataclass_fields__})
    return PairResult(
        pair_id=meta["pair_id"],
        seed=int(meta["seed"]),
        adv_episode_id=meta["adv_episode_id"],
        clean_episode_id=meta["clean_episode_id"],
        adv_log_path=str(paths["adv"]),
        clean_log_path=str(paths["clean"]),
        ground_truth_injection=meta["ground_truth_injection"],
        metrics=metrics,
    )


def load_pair_logs(pair_dir: Path) -> Tuple[List[Event], List[Event], Dict[str, Any]]:
    """Load both logs from a pair directory. Refuses partial pairs."""
    paths = pair_paths(pair_dir)
    if not paths["adv"].is_file() or not paths["clean"].is_file():
        raise FileNotFoundError(
            f"Incomplete pair directory {pair_dir}: need both adv.jsonl and clean.jsonl"
        )
    meta: Dict[str, Any] = {}
    if paths["meta"].is_file():
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    return read_jsonl(paths["adv"]), read_jsonl(paths["clean"]), meta


def recompute_metrics_from_pair_dir(
    pair_dir: Path,
    *,
    tau_C: float = 0.15,
    tau_Z: float = 0.15,
) -> MetricBundle:
    """Recompute metrics from an on-disk pair. Always requires both logs."""
    adv_events, clean_events, meta = load_pair_logs(pair_dir)
    if not meta:
        raise ValueError(f"Missing meta.json in {pair_dir}; cannot recompute safely")
    gt = meta["ground_truth_injection"]
    pilot = load_pilot_experiment("pilot")
    adv, _ = build_episode_pair(
        pilot,
        poison_level=int(meta["poison_level"]),
        topology=str(meta["topology"]),
        seed=int(meta["seed"]),
        trigger=str(meta["trigger"]),
    )
    # Rebuild minimal EpisodeResults for the pairwise metrics helper
    from marble_ext.injector import InjectionRecord

    inj_ev = injection_events(adv_events)[0]
    inj = InjectionRecord(
        event=inj_ev,
        channel=str(gt["channel"]),
        target=str(gt["target"]),
        turn_index=int(gt.get("turn_index", adv.injection.budget_k)),
    )
    carriers = meta.get("adv_meta", {}).get("carriers") or [gt["target"]]
    adv_result = EpisodeResult(
        episode=adv,
        events=adv_events,
        injection=inj,
        adversary_removed=True,
        active_agents_final=[],
        poison_hash=str(gt["poison_hash"]),
        carriers=list(carriers),
    )
    clean_result = EpisodeResult(
        episode=adv,  # placeholder; metrics only need clean events
        events=clean_events,
        injection=inj,
        adversary_removed=True,
        active_agents_final=[],
        poison_hash=str(gt["poison_hash"]),
    )
    return metrics_from_pair_results(
        adv, adv_result, clean_result, tau_C=tau_C, tau_Z=tau_Z
    )


def _summary_dict(result: PairResult) -> Dict[str, Any]:
    m = result.metrics
    assert m is not None
    return {
        "pair_id": result.pair_id,
        "seed": result.seed,
        "adv_log": result.adv_log_path,
        "clean_log": result.clean_log_path,
        "ground_truth_injection": result.ground_truth_injection,
        "c_i": m.c_i,
        "C": m.C,
        "C_bar": m.C_bar,
        "S": m.S,
        "R_Z": m.R_Z,
        "attribution": m.attribution,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one seed-matched adv+clean pair and compute pairwise MetricBundle. "
            "Does not accept a single unpaired log path."
        )
    )
    parser.add_argument("--poison-level", type=int, default=2)
    parser.add_argument("--topology", type=str, default="graph")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trigger", type=str, default="no_trigger")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Root for results/pairs/{digest}/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if pair artifacts already exist",
    )
    parser.add_argument("--tau-C", type=float, default=0.15)
    parser.add_argument("--tau-Z", type=float, default=0.15)
    parser.add_argument(
        "--mode",
        choices=["scripted", "llm"],
        default="scripted",
        help="scripted Mode A (RNG) or llm Mode A (reasoning agents)",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["mock", "openai"],
        default="mock",
        help="Decision backend when --mode llm (mock needs no API key)",
    )
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap episode length (LLM Mode A default 24)",
    )
    parser.add_argument(
        "--embed-backend",
        choices=["auto", "openai", "hash"],
        default="auto",
        help="d_chat embedding source",
    )
    parser.add_argument(
        "--pair-dir",
        type=Path,
        default=None,
        help="Load both adv.jsonl+clean.jsonl from an existing pair dir and recompute metrics",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.pair_dir is not None:
        metrics = recompute_metrics_from_pair_dir(
            args.pair_dir, tau_C=args.tau_C, tau_Z=args.tau_Z
        )
        print(json.dumps(metrics.to_dict(), indent=2))
        return 0

    result = run_pair(
        poison_level=args.poison_level,
        topology=args.topology,
        seed=args.seed,
        trigger=args.trigger,
        results_root=args.results_root,
        force=args.force,
        tau_C=args.tau_C,
        tau_Z=args.tau_Z,
        mode=args.mode,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        max_steps=args.max_steps,
        embed_backend=args.embed_backend,
    )
    print(json.dumps(_summary_dict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
