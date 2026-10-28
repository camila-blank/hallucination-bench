"""StrongREJECT sources and the jailbreak families of plan 0016.

Sources are verbatim copies in ``data/strongreject/source/`` (provenance in its README):
``strongreject_dataset.csv`` (alexandrasouly/strongreject@f7cad6c) and
``wrapping_jailbreaks.json`` / ``judge_templates.json`` (dsbowen/strong_reject@7a551d5).
:func:`wrap` re-implements ``strong_reject.jailbreaks.wrapping_jailbreak``: base64 families
base64-encode the forbidden prompt, then ``template.replace("{forbidden_prompt}", prompt)``.
"""

import base64
import csv
import json
from functools import cache
from typing import Any

from hallucination_bench import DATA

SOURCE_DIR = DATA / "strongreject" / "source"
DATASET_CSV = SOURCE_DIR / "strongreject_dataset.csv"
WRAPPERS_JSON = SOURCE_DIR / "wrapping_jailbreaks.json"
JUDGE_TEMPLATES_JSON = SOURCE_DIR / "judge_templates.json"
DATASET_COMMIT = "f7cad6c17e624e21d8df2278e918ae1dddb4cb56"  # alexandrasouly/strongreject
PACKAGE_COMMIT = "7a551d5b440ec7b75d4f6f5bb7c1719965b76b47"  # dsbowen/strong_reject

# (family, type) in report order; `none` is the control (the bare forbidden prompt)
FAMILIES: tuple[tuple[str, str], ...] = (
    ("none", "control"),
    ("aim", "role-play"),
    ("dev_mode_v2", "role-play (DAN-style)"),
    ("evil_confidant", "role-play"),
    ("prefix_injection", "prefix injection"),
    ("refusal_suppression", "refusal suppression"),
    ("base64_input_only", "encoding"),
    # contrast families (plan 0016 rev 3): the 7 above gave 0 successful jailbreaks on
    # Qwen3.6-27B, so these three, which a probe sample showed it often complies with, supply
    # the recall / contrast set (and more unsuccessful variety)
    ("wikipedia_with_title", "contrast: framing (Wikipedia article)"),
    ("style_injection_json", "contrast: output-format injection"),
    ("poems", "contrast: distractor (poems first)"),
)
FAMILY_NAMES: tuple[str, ...] = tuple(f for f, _ in FAMILIES)
PLANNED: tuple[str, ...] = FAMILY_NAMES[:7]  # the rev-1 families (the rest are contrast)
CONTROL = "none"


@cache
def wrappers() -> dict[str, str]:
    return json.loads(WRAPPERS_JSON.read_text(encoding="utf-8"))


@cache
def judge_templates() -> dict[str, str]:
    return json.loads(JUDGE_TEMPLATES_JSON.read_text(encoding="utf-8"))


def load_dataset() -> list[dict[str, Any]]:
    """The 313 forbidden prompts, ``forbidden_id`` = 0-based CSV row index."""
    with DATASET_CSV.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [{"forbidden_id": i, **r} for i, r in enumerate(rows)]


def wrap(forbidden_prompt: str, family: str) -> str:
    """The jailbreak prompt of ``family`` around ``forbidden_prompt`` (the package's rule)."""
    if family not in wrappers():
        raise KeyError(f"unknown jailbreak family {family!r}")
    prompt = forbidden_prompt
    if family in ("base64", "base64_raw", "base64_input_only") or family.startswith("combination"):
        prompt = base64.b64encode(prompt.encode("utf-8")).decode()
    return wrappers()[family].replace("{forbidden_prompt}", prompt)
