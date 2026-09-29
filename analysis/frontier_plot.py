"""Control–spread frontier plot: achievable (S, C_bar) under fixed k (§4.4 / Phase 6).

Writes JSON points plus a dependency-free SVG scatter (no matplotlib required).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.aggregate import POISON_FAMILY, read_metrics_jsonl
from paths import RESULTS_ROOT

# Distinct enough for print / SVG without a color library
_LEVEL_COLOR = {
    1: "#1f77b4",
    2: "#ff7f0e",
    4: "#2ca02c",
}


def frontier_points(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One point per pair; also mean points per (level, topology)."""
    points: List[Dict[str, Any]] = []
    for row in rows:
        points.append(
            {
                "S": float(row["S"]),
                "C_bar": float(row["C_bar"]),
                "poison_level": int(row["poison_level"]),
                "poison_family": POISON_FAMILY.get(
                    int(row["poison_level"]), f"level_{row['poison_level']}"
                ),
                "topology": row["topology"],
                "trigger": row["trigger"],
                "seed": int(row["seed"]),
                "budget_k": int(row["budget_k"]),
                "pair_id": row["pair_id"],
            }
        )
    return points


def mean_frontier_by_cell(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["poison_level"]), str(row["topology"]), str(row["trigger"]))
        buckets.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for (level, topo, trig), group in sorted(buckets.items()):
        n = len(group)
        out.append(
            {
                "S": sum(float(r["S"]) for r in group) / n,
                "C_bar": sum(float(r["C_bar"]) for r in group) / n,
                "poison_level": level,
                "poison_family": POISON_FAMILY.get(level, f"level_{level}"),
                "topology": topo,
                "trigger": trig,
                "budget_k": int(group[0]["budget_k"]),
                "n": n,
            }
        )
    return out


def _svg_scatter(
    means: Sequence[Dict[str, Any]],
    *,
    width: int = 640,
    height: int = 420,
    pad: int = 50,
) -> str:
    """Minimal SVG scatter of mean (S, C_bar) points."""
    if not means:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="80">'
            '<text x="20" y="40">No frontier points</text></svg>'
        )
    xs = [p["S"] for p in means]
    ys = [p["C_bar"] for p in means]
    xmin, xmax = 0.0, max(1.0, max(xs) * 1.05)
    ymin, ymax = 0.0, max(0.05, max(ys) * 1.15)
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad

    def sx(x: float) -> float:
        return pad + (x - xmin) / (xmax - xmin or 1.0) * plot_w

    def sy(y: float) -> float:
        return pad + plot_h - (y - ymin) / (ymax - ymin or 1.0) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fafafa"/>',
        f'<line x1="{pad}" y1="{pad + plot_h}" x2="{pad + plot_w}" y2="{pad + plot_h}" '
        f'stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad + plot_h}" '
        f'stroke="#333" stroke-width="1.5"/>',
        f'<text x="{width / 2}" y="{height - 12}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="13">Spread S</text>',
        f'<text x="16" y="{height / 2}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13" transform="rotate(-90 16 {height / 2})">Control C̄</text>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-family="sans-serif" '
        f'font-size="14" font-weight="bold">Control–spread frontier (fixed k)</text>',
    ]
    for p in means:
        color = _LEVEL_COLOR.get(int(p["poison_level"]), "#555")
        r = 6 if p["topology"] == "graph" else 5
        shape = "circle" if p["topology"] == "graph" else "rect"
        cx, cy = sx(p["S"]), sy(p["C_bar"])
        if shape == "circle":
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}" '
                f'fill-opacity="0.85" stroke="#222" stroke-width="0.8">'
                f'<title>{p["poison_family"]} {p["topology"]} {p["trigger"]} '
                f'S={p["S"]:.3f} C̄={p["C_bar"]:.3f}</title></circle>'
            )
        else:
            parts.append(
                f'<rect x="{cx - r:.1f}" y="{cy - r:.1f}" width="{2 * r}" height="{2 * r}" '
                f'fill="{color}" fill-opacity="0.85" stroke="#222" stroke-width="0.8">'
                f'<title>{p["poison_family"]} {p["topology"]} {p["trigger"]} '
                f'S={p["S"]:.3f} C̄={p["C_bar"]:.3f}</title></rect>'
            )

    # Legend
    ly = pad
    for level, label in ((1, "goal"), (2, "prescription"), (4, "recruitment")):
        color = _LEVEL_COLOR[level]
        parts.append(
            f'<circle cx="{width - 140}" cy="{ly}" r="5" fill="{color}"/>'
            f'<text x="{width - 128}" y="{ly + 4}" font-family="sans-serif" '
            f'font-size="11">{label}</text>'
        )
        ly += 18
    parts.append("</svg>")
    return "\n".join(parts)


def write_frontier(pilot_dir: Path) -> Path:
    metrics_path = pilot_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing {metrics_path}; run run_grid.py first")
    rows = read_metrics_jsonl(metrics_path)
    points = frontier_points(rows)
    means = mean_frontier_by_cell(rows)
    payload = {
        "budget_k_fixed": True,
        "n_pairs": len(points),
        "points": points,
        "means_by_cell": means,
    }
    json_path = pilot_dir / "frontier.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    svg_path = pilot_dir / "frontier.svg"
    svg_path.write_text(_svg_scatter(means), encoding="utf-8")
    return json_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Write control–spread frontier")
    parser.add_argument("--pilot-dir", type=Path, default=RESULTS_ROOT / "pilot")
    args = parser.parse_args(list(argv) if argv is not None else None)
    path = write_frontier(args.pilot_dir)
    print(json.dumps({"frontier": str(path), "svg": str(path.with_suffix(".svg"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
