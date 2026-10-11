"""Aggregate pilot metrics into a Table-1 style summary (§6 / Phase 6)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import RESULTS_ROOT

# Spec §2.2 family names for pilot levels
POISON_FAMILY = {
    0: "benign",
    1: "goal",
    2: "prescription",
    3: "exploitation",
    4: "recruitment",
    5: "agenda",
}


def read_metrics_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Group by poison family (and optionally topology/trigger)."""
    by_level: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_level[int(row["poison_level"])].append(row)

    table: List[Dict[str, Any]] = []
    for level in sorted(by_level):
        group = by_level[level]
        c_bars = [float(r["C_bar"]) for r in group]
        ss = [float(r["S"]) for r in group]
        rzs = [float(r["R_Z"]) for r in group]
        src = [
            float(r["source_accuracy"])
            for r in group
            if r.get("source_accuracy") is not None
        ]
        comps = [
            float(r["compliance_rate"])
            for r in group
            if r.get("compliance_rate") is not None
        ]
        ctrl_comps = [
            float(r["controlled_compliance_rate"])
            for r in group
            if r.get("controlled_compliance_rate") is not None
        ]
        kinds = {r.get("compliance_kind") for r in group if r.get("compliance_kind")}
        table.append(
            {
                "poison_level": level,
                "poison_family": POISON_FAMILY.get(level, f"level_{level}"),
                "n": len(group),
                "Control_C_bar_mean": round(_mean(c_bars), 4),
                "Control_C_bar_std": round(_std(c_bars), 4),
                "Spread_S_mean": round(_mean(ss), 4),
                "Spread_S_std": round(_std(ss), 4),
                # Verifiable-poison spread (= compliance_rate): |complied| / N_eval
                "Spread_S_compliance_mean": round(_mean(comps), 4) if comps else None,
                "Spread_S_compliance_std": round(_std(comps), 4) if comps else None,
                "Zombieness_R_Z_mean": round(_mean(rzs), 4),
                "Zombieness_R_Z_std": round(_std(rzs), 4),
                # Compliance-based zombieness (= S_compliance): post-injection verify hit rate
                "Zombieness_R_Z_compliance_mean": (
                    round(_mean(comps), 4) if comps else None
                ),
                "Zombieness_R_Z_compliance_std": (
                    round(_std(comps), 4) if comps else None
                ),
                "source_accuracy_mean": round(_mean(src), 4) if src else None,
                "compliance_rate_mean": round(_mean(comps), 4) if comps else None,
                "compliance_rate_std": round(_std(comps), 4) if comps else None,
                "controlled_compliance_rate_mean": (
                    round(_mean(ctrl_comps), 4) if ctrl_comps else None
                ),
                "controlled_compliance_rate_std": (
                    round(_std(ctrl_comps), 4) if ctrl_comps else None
                ),
                "compliance_kind": next(iter(kinds)) if len(kinds) == 1 else sorted(kinds) or None,
            }
        )

    # Cell-level means (level × topology × trigger) for diagnostics
    cells: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (int(row["poison_level"]), str(row["topology"]), str(row["trigger"]))
        cells[key].append(row)
    cell_table: List[Dict[str, Any]] = []
    for (level, topo, trig) in sorted(cells):
        group = cells[(level, topo, trig)]
        comps = [
            float(r["compliance_rate"])
            for r in group
            if r.get("compliance_rate") is not None
        ]
        ctrl_comps = [
            float(r["controlled_compliance_rate"])
            for r in group
            if r.get("controlled_compliance_rate") is not None
        ]
        cell_table.append(
            {
                "poison_level": level,
                "poison_family": POISON_FAMILY.get(level, f"level_{level}"),
                "topology": topo,
                "trigger": trig,
                "n": len(group),
                "C_bar_mean": round(_mean([float(r["C_bar"]) for r in group]), 4),
                "S_mean": round(_mean([float(r["S"]) for r in group]), 4),
                "S_compliance_mean": round(_mean(comps), 4) if comps else None,
                "R_Z_mean": round(_mean([float(r["R_Z"]) for r in group]), 4),
                "compliance_rate_mean": round(_mean(comps), 4) if comps else None,
                "controlled_compliance_rate_mean": (
                    round(_mean(ctrl_comps), 4) if ctrl_comps else None
                ),
            }
        )

    return {
        "n_pairs": len(rows),
        "by_poison_family": table,
        "by_cell": cell_table,
    }


def format_table_markdown(agg: Dict[str, Any]) -> str:
    lines = [
        "| Poison family | n | Control (C̄) | Spread (S) | Verifiable spread / zombieness (S_comp = R_Z_comp) | Dist. zombieness (R_Z) | Controlled compliance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg["by_poison_family"]:
        s_comp = row.get("Spread_S_compliance_mean")
        s_comp_s = row.get("Spread_S_compliance_std") or 0.0
        ccomp = row.get("controlled_compliance_rate_mean")
        ccomp_s = row.get("controlled_compliance_rate_std") or 0.0
        s_comp_cell = "—" if s_comp is None else f"{s_comp:.3f} ± {s_comp_s:.3f}"
        ccomp_cell = "—" if ccomp is None else f"{ccomp:.3f} ± {ccomp_s:.3f}"
        lines.append(
            "| {family} | {n} | {c:.3f} ± {cs:.3f} | {s:.3f} ± {ss:.3f} | {sc} | {z:.3f} ± {zs:.3f} | {cc} |".format(
                family=row["poison_family"],
                n=row["n"],
                c=row["Control_C_bar_mean"],
                cs=row["Control_C_bar_std"],
                s=row["Spread_S_mean"],
                ss=row["Spread_S_std"],
                sc=s_comp_cell,
                z=row["Zombieness_R_Z_mean"],
                zs=row["Zombieness_R_Z_std"],
                cc=ccomp_cell,
            )
        )
    return "\n".join(lines) + "\n"


def aggregate_pilot(pilot_dir: Path) -> Dict[str, Any]:
    metrics_path = pilot_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing {metrics_path}; run run_grid.py first")
    rows = read_metrics_jsonl(metrics_path)
    agg = aggregate_rows(rows)
    (pilot_dir / "aggregate.json").write_text(
        json.dumps(agg, indent=2), encoding="utf-8"
    )
    md = format_table_markdown(agg)
    (pilot_dir / "table1.md").write_text(md, encoding="utf-8")
    return agg


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate pilot grid metrics")
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=RESULTS_ROOT / "pilot",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    agg = aggregate_pilot(args.pilot_dir)
    print(format_table_markdown(agg))
    print(json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
