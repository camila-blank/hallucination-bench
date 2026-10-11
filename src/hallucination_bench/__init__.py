"""HallucinationBench v2: types and severities of hallucination in activation-reading lenses."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
WSBENCH_PUB_COMMIT = "92d763e"

__all__ = ["DATA", "REPO_ROOT", "WSBENCH_PUB_COMMIT"]
