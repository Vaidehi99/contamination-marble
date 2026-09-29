"""Package path helpers."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
CONFIGS_ROOT = PACKAGE_ROOT / "configs"
ORGS_ROOT = CONFIGS_ROOT / "orgs"
POISONS_ROOT = CONFIGS_ROOT / "poisons"
EXPERIMENTS_ROOT = CONFIGS_ROOT / "experiments"
RESULTS_ROOT = PACKAGE_ROOT / "results"
MARBLE_ROOT = REPO_ROOT / "MARBLE"


def ensure_marble_on_path() -> Path:
    """Put MARBLE checkout on sys.path without forking it."""
    import sys

    marble_parent = str(MARBLE_ROOT)
    if MARBLE_ROOT.is_dir() and marble_parent not in sys.path:
        sys.path.insert(0, marble_parent)
    return MARBLE_ROOT
