"""``strongreject`` (plan 0016): do lenses claim, from the prompt tokens, that a FAILED jailbreak
worked? StrongREJECT forbidden prompts x deterministic jailbreak wrappers; Qwen3.6-27B on-policy
ground truth (10 samples, StrongREJECT rubric); a blind disposition detector (Stage D) and an
outcome-blind echo / confirm judge (Stage E) on prompt-position readouts. Its own CLI,
``hbench strongreject …`` (``cli.py``); not on the hallucination index."""

NAME = "strongreject"
