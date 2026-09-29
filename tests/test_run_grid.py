"""Phase 6: pilot grid + aggregate + frontier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.aggregate import aggregate_pilot, aggregate_rows, format_table_markdown  # noqa: E402
from analysis.frontier_plot import write_frontier  # noqa: E402
from config_loaders import load_pilot_experiment  # noqa: E402
from runner.run_grid import iter_grid_jobs, run_grid  # noqa: E402


def test_iter_grid_jobs_full_pilot_size():
    pilot = load_pilot_experiment("pilot")
    jobs = iter_grid_jobs(pilot)
    assert len(jobs) == 180


def test_iter_grid_jobs_filters_and_max_pairs():
    pilot = load_pilot_experiment("pilot")
    jobs = iter_grid_jobs(
        pilot,
        poison_levels=[2],
        topologies=["graph"],
        triggers=["no_trigger"],
        seeds=[0, 1, 2],
    )
    assert len(jobs) == 3
    limited = iter_grid_jobs(pilot, max_pairs=5)
    assert len(limited) == 5


def test_run_grid_subset_aggregate_frontier(tmp_path: Path):
    summary = run_grid(
        experiment="pilot",
        results_root=tmp_path,
        force=True,
        workers=1,
        poison_levels=[2, 4],
        topologies=["graph"],
        triggers=["no_trigger"],
        seeds=[0, 1],
        tau_C=0.05,
        tau_Z=0.05,
    )
    assert summary.n_jobs == 4
    assert summary.n_ok == 4
    assert summary.n_failed == 0

    pilot_dir = Path(summary.results_dir)
    assert (pilot_dir / "metrics.jsonl").is_file()
    agg = aggregate_pilot(pilot_dir)
    assert agg["n_pairs"] == 4
    assert len(agg["by_poison_family"]) == 2
    md = format_table_markdown(agg)
    assert "prescription" in md
    assert "recruitment" in md

    frontier_path = write_frontier(pilot_dir)
    payload = json.loads(frontier_path.read_text(encoding="utf-8"))
    assert payload["n_pairs"] == 4
    assert (pilot_dir / "frontier.svg").is_file()

    # Resume: second run should succeed and report skips
    summary2 = run_grid(
        experiment="pilot",
        results_root=tmp_path,
        force=False,
        workers=1,
        poison_levels=[2, 4],
        topologies=["graph"],
        triggers=["no_trigger"],
        seeds=[0, 1],
        tau_C=0.05,
        tau_Z=0.05,
    )
    assert summary2.n_ok == 4
    assert summary2.n_skipped == 4


def test_aggregate_rows_empty_safe():
    assert aggregate_rows([])["n_pairs"] == 0
