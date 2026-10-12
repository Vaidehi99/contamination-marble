"""CLI demo for LLM Mode A (mock by default; --backend openai with API key)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runner.run_pair import run_pair
from paths import RESULTS_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM Mode A pair demo")
    parser.add_argument("--poison-level", type=int, default=4)
    parser.add_argument("--topology", type=str, default="graph")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=["mock", "openai"], default="mock")
    parser.add_argument("--model", type=str, default="gpt-5.6-luna")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()

    result = run_pair(
        poison_level=args.poison_level,
        topology=args.topology,
        seed=args.seed,
        trigger="no_trigger",
        results_root=args.results_root,
        force=args.force,
        mode="llm",
        llm_backend=args.backend,
        llm_model=args.model,
        max_steps=args.max_steps,
        embed_backend="hash",
        tau_C=0.05,
        tau_Z=0.05,
    )
    m = result.metrics
    assert m is not None
    print(
        json.dumps(
            {
                "pair_id": result.pair_id,
                "mode": "llm",
                "backend": args.backend,
                "c_i": m.c_i,
                "C": m.C,
                "C_bar": m.C_bar,
                "S": m.S,
                "R_Z": m.R_Z,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
