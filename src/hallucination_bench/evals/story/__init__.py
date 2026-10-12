"""story: the claim map (taxonomy v2) on "tell me a story" readouts (plan 0003).

The 20 frozen WorkspaceBench ``hallucination_creative`` items (stories, a poem, a letter, a
scene, answered on-policy by Qwen3.6-27B and capped at ~512 tokens), thinned to 8 read sites
each: 160 cells per layer. Only the bank is story-specific; the pipeline (``run``, ``score``,
``score_v2``) is ``wsbench_chat``'s, selected with ``Models.eval = "story"`` / ``--eval story``.
"""

NAME = "story"
