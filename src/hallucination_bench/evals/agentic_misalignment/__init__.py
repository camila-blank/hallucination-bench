"""agentic_misalignment: the claim map (v2.7) on lens mentions of misalignment (plan 0008).

The 32 frozen WorkspaceBench ``agentic_misalignment`` items (28 misaligned, 4 controls). The
read sites are every **prompt** position (50,452 over the 32 items), so the ``⟦READ⟧`` marker
sits inside the prompt, not the response. Unlike ``story``, this eval has its own driver
(:func:`run.run_am`): a blind batched misalignment-mention detector (Stage D), J-lens summaries
of detected bags (Stage S), the claim map on detected cells (Stage M) and one intent call per
item; ``score`` turns mentions into verdicts, regions and kind-match tiers.
"""

NAME = "agentic_misalignment"
