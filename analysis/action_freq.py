"""CLI wrapper for metrics.action_freq pair diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from event_log.schema import read_jsonl  # noqa: E402
from metrics.action_freq import summarize_pair_action_freq  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Action-type frequency for a pair")
    parser.add_argument("--adv-log", type=Path, required=True)
    parser.add_argument("--clean-log", type=Path, required=True)
    parser.add_argument("--injection-step", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    summary = summarize_pair_action_freq(
        read_jsonl(args.adv_log),
        read_jsonl(args.clean_log),
        injection_step=args.injection_step,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
