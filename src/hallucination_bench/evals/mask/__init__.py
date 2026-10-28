"""``mask`` eval (plan 0017): does the lens claim the model is lying when it is honest?

MASK (Ren et al. 2025, ``cais/MASK``) ground truth on-policy for Qwen3.6-27B (``gt``), a blind
mention detector + a prompt-aware verifier on lens readouts (``run``), and the metrics
(``report``). Its own CLI (``hbench mask ...``, ``cli.register``); no other eval imports it."""
