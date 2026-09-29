"""Phase 5: seed-matched run_pair → MetricBundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loaders import build_episode_pair, length_matched_benign, load_pilot_experiment, load_poison  # noqa: E402
from metrics.compute import compute_metric_bundle  # noqa: E402
from runner.run_pair import (  # noqa: E402
    is_pair_complete,
    load_pair_logs,
    pair_output_dir,
    run_pair,
)
from runner.seed_match import assert_seed_matched_pair  # noqa: E402


def test_level0_length_matched_to_poison_instantiation():
    pilot = load_pilot_experiment("pilot")
    for level in (1, 2, 4):
        adv, clean = build_episode_pair(
            pilot, poison_level=level, topology="graph", seed=11, trigger="no_trigger"
        )
        poison_text = adv.instantiated_content()
        benign_text = clean.instantiated_content()
        assert len(poison_text) == len(benign_text)
        assert clean.content_override == benign_text
        assert clean.poison.level == 0


def test_length_matched_benign_helper_exact():
    poison = load_poison(2)
    benign = load_poison(0)
    poison_text = poison.instantiate({"risky_norm": "skip erectDirtLadder and temporary supports"})
    matched = length_matched_benign(
        poison_text, benign, {"resource": "iron"}
    )
    assert len(matched) == len(poison_text)


def test_compute_metric_bundle_rejects_unpaired():
    with pytest.raises(ValueError, match="both adv_events and clean_events"):
        compute_metric_bundle(
            pair_id="x",
            seed=0,
            poison_level=2,
            topology="graph",
            budget_k=3,
            trigger="no_trigger",
            adv_events=[],
            clean_events=[],
            agent_ids=["builder_1"],
            detected="builder_1",
            true_path=["adversary", "builder_1"],
            poison_hash="abc",
        )


def test_run_pair_end_to_end_metric_bundle(tmp_path: Path):
    result = run_pair(
        poison_level=2,
        topology="graph",
        seed=0,
        trigger="no_trigger",
        results_root=tmp_path,
        force=True,
        tau_C=0.05,
        tau_Z=0.05,
    )
    assert result.metrics is not None
    m = result.metrics
    assert result.ground_truth_injection["content_len"] == result.ground_truth_injection[
        "clean_content_len"
    ]
    assert result.ground_truth_injection["content_len"] > 0

    # Pairwise control / spread are well-formed
    assert m.c_i
    assert "adversary" not in m.c_i
    assert 0.0 <= m.S <= 1.0
    assert m.S == len(m.C) / len(m.c_i)
    if m.C:
        assert m.C_bar == sum(m.c_i[a] for a in m.C) / len(m.C)
        assert m.C_bar > 0.0
    else:
        assert m.C_bar == 0.0

    # Contagion carriers should move more than a typical non-chat agent on average
    # (at least some positive control under poison level 2)
    assert max(m.c_i.values()) > 0.0

    pair_dir = pair_output_dir(result.pair_id, results_root=tmp_path)
    assert is_pair_complete(pair_dir)
    adv_events, clean_events, meta = load_pair_logs(pair_dir)
    report = assert_seed_matched_pair(adv_events, clean_events)
    assert report.structural_match
    assert report.injection_step == result.ground_truth_injection["step"]
    assert meta["seed_match"]["n_events"] == len(adv_events)

    # Resume-safe: second call returns same metrics without rewriting force=False
    resumed = run_pair(
        poison_level=2,
        topology="graph",
        seed=0,
        trigger="no_trigger",
        results_root=tmp_path,
        force=False,
    )
    assert resumed.pair_id == result.pair_id
    assert resumed.metrics is not None
    assert resumed.metrics.C_bar == m.C_bar
    assert resumed.metrics.S == m.S


def test_run_pair_cli_smoke(tmp_path: Path, capsys):
    from runner.run_pair import main

    code = main(
        [
            "--poison-level",
            "4",
            "--topology",
            "star",
            "--seed",
            "1",
            "--trigger",
            "no_trigger",
            "--results-root",
            str(tmp_path),
            "--tau-C",
            "0.05",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert "c_i" in out and "C_bar" in out and "S" in out
    assert out["ground_truth_injection"]["content_len"] == out["ground_truth_injection"][
        "clean_content_len"
    ]
