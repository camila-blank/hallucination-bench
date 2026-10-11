"""The token-bag -> prose step for token-lens readouts (``interp-v1``, verbatim prompt).

Ported from WorkspaceBench ``src/wsbench/summarizer.py`` @ 92d763e (MIT): the prompt, schema,
``render_user``, ``render_bag`` and the result reader. The cached call itself lives in the eval
pipeline (``evals/wsbench_chat/run.py``), keyed ``summ:<cell key>`` with the fingerprint
``(SUMMARIZER_PROMPT_VERSION, model, effort, bundle text)``.
"""

from collections.abc import Sequence

from hallucination_bench.llm import schema_block

SUMMARIZER_PROMPT_VERSION = "interp-v1"

INTERP_SYSTEM = """You are shown the top-10 token readouts from an interpretability lens at one \
position inside a language model that was reading a passage you cannot see. Tokens may include \
noise, fragments, other languages (translate them), or byte artifacts. In one or two sentences, \
state what these outputs are collectively trying to say — the situation or mental content they \
point to. Commit to the most specific reading the tokens support; do not just say they are noisy."""
INTERP_SCHEMA = schema_block("interp", {"interpretation": {"type": "string"}}, ["interpretation"])
INTERP_USER = "TOKEN READOUTS:\n{txt}"

PROMPTS: dict[str, str] = {"INTERP_SYSTEM": INTERP_SYSTEM, "INTERP_USER": INTERP_USER}


def render_user(txt: str) -> str:
    return INTERP_USER.replace("{txt}", txt)


def render_bag(tokens: Sequence[str], scores: Sequence[float] | None) -> str:
    """``tok (score)`` list, best first, scores to 2 dp; without scores the tokens joined by
    ``" | "``."""
    if scores is None:
        return " | ".join(tokens)
    pairs = sorted(zip(tokens, scores, strict=True), key=lambda p: -p[1])
    return " | ".join(f"{t} ({s:.2f})" for t, s in pairs)


def interpretation_of(r: object) -> str | None:
    if not isinstance(r, dict):
        return None
    t = r.get("interpretation")
    if not isinstance(t, str) or not t.strip():
        return None
    return t
