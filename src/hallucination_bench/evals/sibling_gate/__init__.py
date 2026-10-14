"""Plan 0009: the sibling gate (rev 21). ``substitution_sibling`` segments are grouped into units
(:mod:`.units`), resampled 10 times from the same lens (:mod:`.resample`), slot-filled by the
``sibfill-v1`` judge (:mod:`.fill_judge`, rung 1), and the units rung 1 leaves unsure go to
probes (:mod:`.classes`, :mod:`.texts`, :mod:`.probes` / :mod:`.um_probe`, :mod:`.ladder`,
rung 2). The labels file ``outputs/sibling_gate/labels/<arm>.jsonl`` is joined into scoring by
``score_v2.AccumulatorV2(sibling_labels=...)``.

Always import the unit module qualified (``from hallucination_bench.evals.sibling_gate import
units as sibling_units``): ``hallucination_bench/units.py`` is the readout splitter.
"""
