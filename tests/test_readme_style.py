"""The README describes the bench as it is: no dates, no plan numbers and no version-style labels
in its prose. A name with a version in it that a user has to type (a file, a flag value, an id)
goes in code formatting, where this check does not look."""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# the pages held to the rule: the README and the pages written beside it
PAGES = [
    "README.md",
    "docs/checks.md",
    "docs/judge_backends.md",
    "docs/claim_map.md",
    "docs/commands.md",
    "docs/evals.md",
    "docs/hallucination_index.md",
    "docs/judge_and_caveats.md",
]

MONTHS = "January|February|March|April|June|July|August|September|October|November|December"
DATED = [
    (re.compile(r"\b(?:19|20)\d\d-\d\d(?:-\d\d)?\b"), "an ISO date"),
    (re.compile(rf"\b(?:{MONTHS})\s+(?:\d{{1,2}},?\s+)?(?:19|20)\d\d\b"), "a month and year"),
    (re.compile(r"\b\d{1,2}\s+(?:" + MONTHS + r")\b"), "a day and month"),
    (re.compile(r"\bplans?\s+0\d{2,3}\b", re.I), "a plan number"),
]
VERSIONED = re.compile(r"(?<![\w.])[A-Za-z][\w-]*-v\d+(?:\.\d+)*\b|(?<![\w./-])v\d+(?:\.\d+)*\b")
# version-style words that are allowed in prose: none today. Add the exact word, with the reason.
ALLOWED: set[str] = set()


def prose(text: str) -> str:
    """The text a reader reads as prose: no fenced blocks, code spans or link targets."""
    text = re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)
    text = re.sub(r"`[^`\n]*`", "", text)
    return re.sub(r"\]\([^)\s]*\)", "]", text)


def problems(text: str) -> list[str]:
    out = []
    for line in prose(text).splitlines():
        where = f"in {line.strip()[:80]!r}"
        for rx, what in DATED:
            out += [f"{what}: {m.group(0)!r} {where}" for m in rx.finditer(line)]
        labels = [m.group(0) for m in VERSIONED.finditer(line)]
        out += [f"a version-style label: {w!r} {where}" for w in labels if w not in ALLOWED]
    return out


@pytest.mark.parametrize("page", PAGES)
def test_no_dates_and_no_version_labels_in_prose(page):
    if not (REPO / page).is_file():
        pytest.skip(f"{page} is not in this checkout")
    found = problems((REPO / page).read_text(encoding="utf-8"))
    assert not found, f"{page}:\n  " + "\n  ".join(found)


def test_the_check_sees_what_it_should_and_skips_code():
    bad = "Renamed on 2026-11-20 by plan 0101; the fast-v3 subset and bank v6, since 3 November."
    assert len(problems(bad)) == 5
    fine = "Pass `subset=fast-v3` (see [the page](docs/fast_v3.md)).\n"
    fine += "```\nwsbench v2 2026-11-20\n```\n"
    assert problems(fine) == []


def test_every_relative_link_in_the_readme_resolves():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\((?!https?://|#)([^)#\s]+)(?:#[^)]*)?\)", text)
    assert targets
    missing = sorted({t for t in targets if not (REPO / t).exists()})
    assert not missing, missing


def test_the_readme_spells_no_versioned_name_even_in_code():
    """What a newcomer types from the README carries no version: the commands accept a
    descriptive name, and a file whose name has one is linked by its role."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    shown = re.sub(r"\]\([^)\s]*\)", "]", text)  # link targets are not shown
    found = sorted({m.group(0) for m in VERSIONED.finditer(shown)} - ALLOWED)
    assert not found, found
