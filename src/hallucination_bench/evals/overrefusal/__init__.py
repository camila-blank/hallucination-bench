"""overrefusal (plan 0015): does a lens claim refusal on prompts the model never refuses?

OR-Bench (``bench-llm/or-bench``, pinned) hard-1k prompts that Qwen3.6-27B refuses 0/10 times
on-policy (the hallucination population) and toxic prompts it refuses 10/10 times (the recall
control). Readouts are every **prompt** position (up to ``<|im_start|>assistant``). Own driver
(not the chat pipeline): ``hbench overrefusal candidates|judge-gt|bank|run|report``. A blind
batched detector labels mentions (``refusal_intent`` / ``harm_salience`` / ``comply_intent`` /
``other``), a blind verifier re-judges the ``refusal_intent`` candidates, and ``score`` reports
rates per (arm, layer, population) with prompt-bootstrap CIs.
"""

NAME = "overrefusal"
