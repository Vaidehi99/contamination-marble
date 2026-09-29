"""Sweep the pilot experiment matrix (§6 / Phase 6).

Calls ``run_pair`` for every cell × seed. Resume-safe via the Phase 5 pair
layout under ``results/pairs/``; also writes a pilot index under
``results/pilot/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loaders import load_pilot_experiment
from interfaces import PairResult, PilotExperimentConfig
from paths import RESULTS_ROOT
from runner.run_pair import run_pair


@dataclass(frozen=True)
class GridJob:
    poison_level: int
    topology: str
    seed: int
    trigger: str
    budget_k: int


@dataclass
class GridSummary:
    experiment: str
    n_jobs: int
    n_ok: int
    n_skipped: int
    n_failed: int
    elapsed_sec: float
    results_dir: str
    failures: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def iter_grid_jobs(
    pilot: PilotExperimentConfig,
    *,
    poison_levels: Optional[Sequence[int]] = None,
    topologies: Optional[Sequence[str]] = None,
    triggers: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    max_pairs: Optional[int] = None,
) -> List[GridJob]:
    levels = list(poison_levels) if poison_levels is not None else list(pilot.poison_levels)
    tops = list(topologies) if topologies is not None else list(pilot.topologies)
    trigs = list(triggers) if triggers is not None else list(pilot.triggers)
    seed_list = list(seeds) if seeds is not None else list(pilot.seeds)

    jobs: List[GridJob] = []
    for level in levels:
        for topology in tops:
            for trigger in trigs:
                for seed in seed_list:
                    jobs.append(
                        GridJob(
                            poison_level=int(level),
                            topology=str(topology),
                            seed=int(seed),
                            trigger=str(trigger),
                            budget_k=int(pilot.budget_k),
                        )
                    )
                    if max_pairs is not None and len(jobs) >= max_pairs:
                        return jobs
    return jobs


def _job_to_metrics_row(job: GridJob, result: PairResult) -> Dict[str, Any]:
    m = result.metrics
    assert m is not None
    return {
        "pair_id": result.pair_id,
        "poison_level": job.poison_level,
        "topology": job.topology,
        "budget_k": job.budget_k,
        "trigger": job.trigger,
        "seed": job.seed,
        "C_bar": m.C_bar,
        "S": m.S,
        "R_Z": m.R_Z,
        "n_controlled": len(m.C),
        "C": m.C,
        "c_i": m.c_i,
        "Z_i": m.Z_i,
        "source_accuracy": (m.attribution or {}).get("source_accuracy"),
        "path_accuracy": (m.attribution or {}).get("path_accuracy"),
        "adv_log": result.adv_log_path,
        "clean_log": result.clean_log_path,
    }


def _run_one_job(
    job_dict: Dict[str, Any],
    *,
    results_root: str,
    force: bool,
    tau_C: float,
    tau_Z: float,
    mode: str = "scripted",
    llm_backend: str = "mock",
    llm_model: str = "gpt-4o-mini",
    max_steps: Optional[int] = None,
    embed_backend: str = "auto",
) -> Dict[str, Any]:
    """Process-pool entrypoint (picklable)."""
    job = GridJob(**job_dict)
    try:
        result = run_pair(
            poison_level=job.poison_level,
            topology=job.topology,
            seed=job.seed,
            trigger=job.trigger,
            results_root=Path(results_root),
            force=force,
            tau_C=tau_C,
            tau_Z=tau_Z,
            mode=mode,
            llm_backend=llm_backend,
            llm_model=llm_model,
            max_steps=max_steps,
            embed_backend=embed_backend,
        )
        return {
            "ok": True,
            "job": job_dict,
            "row": _job_to_metrics_row(job, result),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — collect per-cell failures for the grid
        return {
            "ok": False,
            "job": job_dict,
            "row": None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def run_grid(
    *,
    experiment: str = "pilot",
    results_root: Path = RESULTS_ROOT,
    force: bool = False,
    workers: int = 1,
    poison_levels: Optional[Sequence[int]] = None,
    topologies: Optional[Sequence[str]] = None,
    triggers: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    max_pairs: Optional[int] = None,
    tau_C: float = 0.15,
    tau_Z: float = 0.15,
    mode: str = "scripted",
    llm_backend: str = "mock",
    llm_model: str = "gpt-4o-mini",
    max_steps: Optional[int] = None,
    embed_backend: str = "auto",
) -> GridSummary:
    pilot = load_pilot_experiment(experiment)
    jobs = iter_grid_jobs(
        pilot,
        poison_levels=poison_levels,
        topologies=topologies,
        triggers=triggers,
        seeds=seeds,
        max_pairs=max_pairs,
    )

    pilot_dir = results_root / "pilot"
    rel = Path(pilot.results_dir)
    if rel.parts and rel.parts[0] == "results":
        pilot_dir = results_root.joinpath(*rel.parts[1:])
    else:
        pilot_dir = results_root / rel
    if mode == "llm":
        pilot_dir = pilot_dir.parent / f"{pilot_dir.name}_llm"
    pilot_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    n_skipped = 0
    t0 = time.time()

    # Count resumes before run (run_pair itself skips complete pairs)
    from runner.run_pair import is_pair_complete, pair_output_dir
    from config_loaders import build_episode_pair

    if not force:
        for job in jobs:
            adv, _ = build_episode_pair(
                pilot,
                poison_level=job.poison_level,
                topology=job.topology,
                seed=job.seed,
                trigger=job.trigger,
                mode=mode,
            )
            if is_pair_complete(pair_output_dir(adv.pair_id, results_root=results_root)):
                n_skipped += 1

    job_dicts = [asdict(j) for j in jobs]
    results_root_s = str(results_root)
    run_kwargs = dict(
        results_root=results_root_s,
        force=force,
        tau_C=tau_C,
        tau_Z=tau_Z,
        mode=mode,
        llm_backend=llm_backend,
        llm_model=llm_model,
        max_steps=max_steps,
        embed_backend=embed_backend,
    )

    if workers <= 1:
        outcomes = [_run_one_job(jd, **run_kwargs) for jd in job_dicts]
    else:
        outcomes = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(_run_one_job, jd, **run_kwargs): jd for jd in job_dicts
            }
            for fut in as_completed(futs):
                outcomes.append(fut.result())

    for out in outcomes:
        if out["ok"]:
            rows.append(out["row"])
        else:
            failures.append(
                {"job": out["job"], "error": out["error"], "traceback": out.get("traceback")}
            )

    # Stable sort for reproducible index
    rows.sort(
        key=lambda r: (
            r["poison_level"],
            r["topology"],
            r["trigger"],
            r["seed"],
        )
    )

    metrics_path = pilot_dir / "metrics.jsonl"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = GridSummary(
        experiment=pilot.name,
        n_jobs=len(jobs),
        n_ok=len(rows),
        n_skipped=n_skipped,
        n_failed=len(failures),
        elapsed_sec=round(time.time() - t0, 3),
        results_dir=str(pilot_dir),
        failures=failures,
    )
    (pilot_dir / "grid_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )
    (pilot_dir / "jobs.json").write_text(
        json.dumps(job_dicts, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run pilot contamination grid")
    parser.add_argument("--experiment", type=str, default="pilot")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--poison-levels", type=int, nargs="*", default=None)
    parser.add_argument("--topologies", type=str, nargs="*", default=None)
    parser.add_argument("--triggers", type=str, nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--tau-C", type=float, default=0.15)
    parser.add_argument("--tau-Z", type=float, default=0.15)
    parser.add_argument("--mode", choices=["scripted", "llm"], default="scripted")
    parser.add_argument("--llm-backend", choices=["mock", "openai"], default="mock")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--embed-backend", choices=["auto", "openai", "hash"], default="auto"
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="After the grid, run analysis/aggregate + frontier_plot",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = run_grid(
        experiment=args.experiment,
        results_root=args.results_root,
        force=args.force,
        workers=args.workers,
        poison_levels=args.poison_levels,
        topologies=args.topologies,
        triggers=args.triggers,
        seeds=args.seeds,
        max_pairs=args.max_pairs,
        tau_C=args.tau_C,
        tau_Z=args.tau_Z,
        mode=args.mode,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        max_steps=args.max_steps,
        embed_backend=args.embed_backend,
    )
    print(json.dumps(summary.to_dict(), indent=2))

    if args.aggregate and summary.n_ok > 0:
        from analysis.aggregate import aggregate_pilot
        from analysis.frontier_plot import write_frontier

        pilot_dir = Path(summary.results_dir)
        agg = aggregate_pilot(pilot_dir)
        frontier = write_frontier(pilot_dir)
        print(
            json.dumps(
                {"aggregate": agg, "frontier_path": str(frontier)},
                indent=2,
            )
        )

    return 0 if summary.n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
