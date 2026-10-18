"""Plan 0012: the country ladder (plan 0006) stays on the lexicon its results were produced with.

Plan 0005's later review fix added bare-name aliases (Guinea, Georgia, ...) to
``br_lexicons.json``; the country ladder must not pick them up silently.
"""

import hashlib
import json

from hallucination_bench.evals.sibling_probe import ladder_lex as LX
from hallucination_bench.evals.sibling_probe import um_ladder as U

PINNED_SHA256 = "6d76f92e79def954ee588bf198adfe781fa2405a68bbbd75574124eae4ae009a"


def test_country_base_lexicon_is_pinned():
    data = U.UM_COUNTRY_BASE_LEXICON_PATH.read_bytes()
    assert hashlib.sha256(data).hexdigest() == PINNED_SHA256
    assert U.UM_COUNTRY_BASE_LEXICON_PATH != LX.LEXICON_PATH


def test_country_ladder_ignores_plan_0005_bare_name_aliases():
    plan5 = json.loads(LX.LEXICON_PATH.read_text())["categories"]["country"]["concepts"]
    assert "Guinea" in plan5["guinea"]["en"]  # plan 0005's matcher has the bare name
    fm = U.FastMatcher(U.load_lexicon())
    assert fm.concepts_in("Guinea is lovely") == set()
