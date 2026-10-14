# Plan 0009: sibling substitutions become hallucinations, gated by a 10-sample resample check and probes

Issue #10. Author: Camila Blank (planned with Claude), 2026-10-14. Revision 1.
Branch `camila/sibling-resample`, stacked on `camila/taxonomy-v2` (PR #4, plan 0002 rev 20 at
f969f83). This plan changes only how `substitution_sibling` segments are scored. It re-labels
none of the claim-map judge's output.

Camila (2026-10-14): "for the substitution sibling category by default we should consider these
hallucinations but we should only have unambiguous ones. By default we should have a step where
for those sibling claims we sample 10 times and see if lots of different items come up (i.e.
sometimes canada pops up, sometimes russia, sometimes uk), then we can tell that canada was a
hallucination. And then for ones where we are still unsure let's actually go ahead and make the
probes and test this like we did for the user modeling eval."

## 0. What changes, what is retained

- **Changes.** `substitution_sibling` stops being a whole-leaf probe candidate. Each sibling
  segment gets a `sibling_status` from the ladder below (§1–§4). Two statuses count as
  hallucination: `unstable` and `probe_unconfirmed`. They form a fourth hallucination
  component, `substitution_sibling`.
- **Retained, rev 20 mapping.** Hallucination keeps its three components `false_claim`,
  `detail` and `substitution_category`, now with `substitution_sibling` (gated) as the fourth.
- **Retained, probe candidates.** `supplementary_false` and `ambiguous_topic` stay probe
  candidates. Sibling segments whose ladder status is not a hallucination (`probe_faithful`,
  `probe_nearby`, `cross_method`, `probe_na`, `resample_failed`) also stay in
  `probe_candidate`, which now reads as "ambiguous or supported co-activation".
- **Retained, everything else.** `model_state` stays thinky. The leaf definitions, the judge
  prompt (`claimmap-v2.7`) and the cached `:M` calls are untouched. Every other leaf's verdict is
  unchanged.
- **Default when the gate has not run.** For an arm whose ladder outputs are missing, every
  sibling segment keeps verdict `probe_candidate` and status `ungated`. The report prints the
  arm as "sibling gate not run" and never silently counts it as hallucination or non-
  hallucination.
- **No commitment judge.** The ladder in plans 0006/0007 starts with a commitment judge because
  lexicon mentions include options and comparisons. Here, the claim map has already decided that
  the segment is a contradicted claim, so that rung is skipped.

## 1. Units

- A **sibling unit** is a group of `substitution_sibling` segments sharing (arm, id, layer, pos,
  normalised `pair.said`, normalised `pair.actual`). Normalisation is casefold, NFKC, and
  stripped punctuation and whitespace.
- Repeats of the same pair in a cell's 3 samples are one unit. A segment inherits its unit's
  status. Character-weighted shares stay segment-level, as in plan 0002.
- Expected size, from the v2.7 cells: s3d-rl600-all11 has 7,488 segments in 1,661 cells,
  mlnla-rl600-all11 8,474 in 2,757, mlnla-rl2-800-all11 9,085 in 2,727, s3d-sft251 3,983 in 1,271,
  s3d-rl600 3,233 in 753, nla-rl-L42 1,185 in 271, nla-sft-L42 563 in 204 and patchscopes 109 in
  81. The token arms have at most 34 each.
- **Arms in scope:** the 14 arms in `docs/results/0002_report_full_v2.7.md`, plus
  mlnla-rl600-all11, which plans 0006/0007 use. The mlnla-ck* checkpoint-sweep arms are out of
  scope and keep status `ungated`.

## 2. Rung 1: the resample check

### 2a. Resampling

For each cell holding at least one sibling unit, draw **10 fresh samples** from the same lens at
the same (id, layer, pos):
- the same activation, from `acts-hallucination-chat`, or `acts-hallucination-chat-L42` for the
  NLA L42 arms;
- the same sampling parameters and prompt format as the original run (`run_config.json`):
  T = 1.0, top_p 0.95, top_k 64, max_new 256 for s3d and NLA; patchscopes uses its own config.
- **Fresh** means the cell's 3 original samples are not counted. The cell was selected because
  one of them named the sibling, so counting them would bias the estimate of how often the lens
  says it. As a sensitivity row, the 3 originals are added (13 samples).
- **Shared pools.** Arms that share a lens and checkpoint share one resample pool per cell:
  s3d-rl600 (5 layers) and s3d-rl600-all11 are the same model at the same cells.
- **Generators**, run from a snapshot copy of the `gw-mlnla` scripts (never edited in place):
  - s3d-*: `wsbench_read_modal.py`, `--k 10`;
  - mlnla-* and nla-*: `wsbench_nla_modal.py`, `--n-samples 10`;
  - patchscopes: `modal_lensagree`.
- **Cell subsets:** neither generator takes a per-cell subset, so each job gets a trimmed
  `manifest.json` in a new acts subdir that references the existing acts files. The trimmed
  manifest keeps only the needed prompts, with each prompt's `eval_positions` cut to its sibling
  cells' positions. If the existing scripts cannot point at the original tensors without copying,
  copy only the needed (id, layer) slices and record that in the results doc.
- **GPU:** every Modal GPU job runs on B300:8, sharded for speed.
- **Output:** `outputs/sibling_gate/resample/<pool>/resample.jsonl`, rows
  `{id, layer, pos, samples: [10 strings], config}` in the readout contract.
- **Sanity check before the full run:** for 20 cells per pool, the resampled block shape (number
  of "- " lines, length distribution) must match the originals. A mismatch means the generator
  was not reproduced faithfully, and the run stops.

**Deterministic arms.** jlens, logit, tuned, rlens, template-v3 and sae-l42 are token bags with
no sampling. Their units get status `deterministic` and go straight to rung 2.

### 2b. Slot-fill judge (`sibfill-v1`)

One call per sibling unit, `claude-opus-5-5`, effort low, no temperature. It is cached like
every other judge (`{arm}__{key}__{said}__{actual}:F`). The call sees:
- the original segment text;
- `pair.said`, `pair.actual` and `contradicted_by`;
- a ±400-character window of the item's text around `contradicted_by`;
- the 10 resampled blocks, numbered.

For each block, the judge answers what fills **the same slot** (the same role the said object
plays in the original claim, e.g. "the company that built me"):
- `said`: the same object (aliases count, e.g. "OpenAI's ChatGPT" for ChatGPT);
- `actual`: the right object;
- `other`: a different object of the same class; the judge names it;
- `none`: the slot is not filled.

Every non-`none` answer must carry a verbatim quote from that block. The quote is checked with
`readouts.verify_quote`. A failed quote makes that block `none`, and the count of failed quotes
is reported. The call fails closed: an unparseable answer after retries makes the unit
`resample_failed`.

**Calibration** (plan 0006's protocol):
- 40 dev units, labelled by a fresh subagent that is blind to the judge, with one allowed
  prompt revision;
- then 80 blind gold units, labelled by a different fresh subagent working only from this
  section's text;
- target: per-block fill agreement ≥ 90% and unit-level rung-1 decision agreement ≥ 85%;
- below target, the result is reported as "judge (below target)" and the run is not blocked.

The gold units are stratified over arms and over decisions (unstable, stable, sparse), using
the dev judge's decisions to stratify.

### 2c. Decision rule (pre-registered)

Per unit, from the 10 fresh blocks:
- n_said = blocks with fill `said`;
- n_fill = blocks with any fill other than `none`;
- n_other = distinct `other` objects.

| condition | status | goes to |
|---|---|---|
| n_fill ≥ 4 and n_said ≤ 2 | `unstable` | **hallucination** (final) |
| n_said ≥ 3 | `stable` | rung 2 |
| n_fill < 4 and n_said ≤ 2 | `sparse` | rung 2 |

- **Why these cut-offs.** n_said ≤ 2 of 10 means the lens names this sibling for the slot at
  most a fifth of the time. When the slot is filled at least 4 times, the rest of the fills are
  the right object or other siblings. Both are the "lots of different items come up" pattern:
  Canada once, Russia once, the UK once, or mostly the right answer.
- **Too sparse to tell.** When the slot is rarely filled at all, the rule cannot tell an
  unstable sibling from a lens that rarely addresses the slot, so the unit goes to the probes.
- **Diversity sub-flag** (reported): `diverse` = n_other ≥ 2.
- **Sensitivity rows:**
  - (i) the strict diversity rule: unstable also requires n_other ≥ 2, otherwise the unit goes
    to rung 2;
  - (ii) cut-off n_said ≤ 1;
  - (iii) cut-off n_said ≤ 3;
  - (iv) 13 samples including the originals, with the cut-offs scaled to ≤ 3 / ≥ 4 and
    n_fill ≥ 5.

## 3. Rung 2: probes, following the plan 0006/0007 ladder

Rung 2 runs on every `stable`, `sparse` and `deterministic` unit.

### 3a. Concept classes

- **Class judge** (`sibclass-v1`, Opus 5.5, cached): one call per distinct normalised (said,
  actual) pair. It returns a short class name ("AI assistants", "countries", "Chinese tech
  companies") and canonical names for said and actual. Pairs are clustered into classes by the
  class name, after a second dedupe pass in which one call maps the full list of class names to
  canonical classes.
- **Concept set C per class:** every said and actual in the class, plus generated fillers that
  bring it to at least 16 concepts.
  - The fillers are well-known same-class members, generated in one call per class
    (`sibfill-concepts-v1`).
  - A filler is never an item's gold or a said object.
- **Compatible stratum:** objects that are true together with the gold (a parent company, an
  alias, a product of the gold company) are a `compatible` stratum, as with `um_domains.COMPATIBLE`.
  They are flagged by the class judge, never used as distractors for that gold, and reported
  separately.
- Classes with fewer than 9 concepts after filling make their units `probe_na`.

### 3b. Probe texts and capture

- **Per concept:** 24 explicit and 24 implied texts. There are also 24 neutral texts per class,
  which are not a probe class, as in plan 0006 §4a. Generator `sibgen-chat-v1`, `claude-opus-5-5`.
- **Text format.** Unlike plans 0006/0007, the texts are short assistant replies in wsbench_chat
  form (system, user turn, assistant reply), so the reading positions resemble the eval sites:
  - explicit texts name the concept in the reply and no other concept of C;
  - implied texts make it unambiguous without naming it ("the company behind the Ascend chips");
  - neutral texts name nothing in C.
- **Code checks:** reuse `um_ladder.text_check` adapted to C. No 8-gram is shared with any
  wsbench_chat item text, and there are no duplicates.
- **Capture:** `wsbench_capture_modal.py`, run from `/workspace/gw-wrong-bridge/scripts/olens_suite/workspace_bench`
  and never edited. It captures the 12 layers (20…60 step 4, plus 42). The read positions are
  every punctuation token in the assistant reply after the concept's first (explicit or
  implied) mention, the same token classes as the wsbench_chat sites (`site_kind`), up to 24
  per text. The volume subdir is `acts-sibgate-<class_slug>-probe`, or one subdir with a class
  field if the script allows it.
- **Centring and null key.** The ladder keyed these by (ending, offset). Here the key is the
  punctuation token class of the read position (`.`, `,`, `!`, `?`, `:`, `;`, newline, `)`,
  CJK `。，！？：`, other). The eval sites use the same keying from their token. The key goes
  through `um_probe._key` / `eval_key` by setting `ending = 0` and `offset = token-class index`,
  so the vendored code needs no change.

### 3c. Probes and validity (unchanged recipe)

- **Recipe:** type (a) whitened DiffMean (`probes.cross_fit_whitened`), 4 folds by text,
  per-key centring, and the per-fold null z on other concepts' implied positions. These are
  vendored from `camila/um-ladder-v2` (`probes.py`, `um_probe.py`) by copying the two files.
  The branch is not merged, and the files are not edited beyond import paths.
- **Validity** per (concept, layer): OOF implied AUROC ≥ 0.90.
- **Type (b) logreg** is not run, which matches plan 0006 rev 8: type (b) was not evaluable.

### 3d. Fire rule (unchanged from plans 0006/0007)

- **Distractors.** Per unit, 8 distractors from C:
  - 4 nearest the gold by cosine of grand-mean-centred implied centroids at L44, plus 4
    seeded-random (seed = unit key);
  - never the gold, the said object, a compatible concept, or a concept that any in-scope arm
    names as a sibling for that item.
- **n/a rules** (status `probe_na`):
  - S has no probe or an invalid one at L;
  - fewer than 6 distractors are valid at L;
  - the item is **unreadable** at L: the gold beats all its distractors at fewer than 50% of
    that item's sites at L. This is the analogue of the 21-position window rule.
- **Fires** (status `probe_faithful`): at (id, L, pos), z_S exceeds every valid distractor's z.
- **Nearby** (status `probe_nearby`): it fires at the same (id, pos) at a neighbour layer within
  ±4 (the neighbours at distance exactly 4, plus L42 for L40/L44; for L42-only arms, L40 and
  L44).
- **Cross-method** (status `cross_method`), only if the probe neither fires nor fires nearby:
  another in-scope arm has a `substitution_sibling` segment for the same item, the same
  normalised said object, and a layer within ±4. Any position counts. The token arms count as
  methods here.
- **Otherwise** the status is `probe_unconfirmed`, which is a **hallucination**.
- **Matched null** (reported, not used in labels): the same S against the same distractors at
  the same item's other sites and at other items with the same gold, where no arm names S.
- **Shared-prior sensitivity row:** discount `cross_method` when the agreeing arm names S as a
  sibling for ≥ 50% of the items whose gold shares the unit's class.

### 3e. Eval-side scoring

- The eval residuals at the unit's (id, L, pos), and at all sites of the item at L (for the
  readability rule), come from `acts-hallucination-chat`. For the L42 arms they come from
  `acts-hallucination-chat-L42`.
- Scoring is a Modal CPU job adapted from `um_probes_modal.py --phase fit|score`, with one
  container per layer, on the volume `sibling-probes` under `/sibgate-v1/`.
- **Leakage guard:** distractors are frozen before scoring, and the score phase records the
  distractor file's sha256. `label` refuses a mismatch.

## 4. Scoring and report

- `taxonomy_v2.py`:
  - add `SIBLING_STATUSES` and `SIBLING_HALLUCINATION = ("unstable", "probe_unconfirmed")`;
  - `segment_verdict(seg)` returns `hallucination` for a `substitution_sibling` segment whose
    status is in `SIBLING_HALLUCINATION`. Otherwise it returns `verdict_of(leaf)`, which is
    unchanged.
  - The `VERDICTS` dict and its assert stay as they are, with `substitution_sibling` still listed
    under `V_PROBE` as the ungated default.
  - Everything that computes verdict shares (`score_v2.py`, `report_v2.py`, the plot scripts)
    switches from `verdict_of(seg["leaf"])` to `segment_verdict(seg)`.
- **Status source.** Statuses are read from `outputs/sibling_gate/labels/<arm>.jsonl`, keyed by
  (cell key, readout, start, end), and joined at scoring time. `cells.jsonl` is never rewritten.
- **Report columns:** the hallucination columns gain `↳ substitution_sibling` (gated).
- **New table per arm:** units and segments by status (`unstable` / `stable` / `sparse` /
  `deterministic` / `probe_faithful` / `probe_nearby` / `cross_method` / `probe_unconfirmed` /
  `probe_na` / `resample_failed` / `ungated`), the `diverse` share among unstable units, and the
  sensitivity rows from §2c and §3d.
- **Regenerated at $0:** `docs/results/0002_report_full_v2.7.md` and its plots under the new
  mapping, written as `0002_report_full_v2.8.md` (scoring rev 21). The v2.7 report stays as is.
- **Results doc** `docs/results/0009_sibling_gate.md`:
  - procedure as run;
  - the headline (old vs new hallucination share per arm);
  - the status mix per arm;
  - calibration;
  - 10 hand-read examples each of unstable / stable-faithful / stable-unconfirmed;
  - spend.
- `CLAUDE.md` and plan 0002 get a "rev 21: sibling gate" pointer.

## 5. Execution

1. **Code, with unit tests** for the decision rule, segment_verdict, the unit grouping, the
   label join, the token-class key and the ungated default.
2. **Pilot:** s3d-rl600-all11 L44 + nla-rl-L42, 150 cells total.
   - Run resample, slot judge, dev calibration, and rung 2 on the pilot's unsure units.
   - Pilot gate: the block-shape check passes; the slot judge's dev agreement ≥ 90% after at most
     one revision; at least one class has ≥ 9 concepts and valid probes at L44.
3. **Full run** in tmux session `sibgate`, log `logs/sibgate_<timestamp>.log`, as one chained
   driver `scripts/sibling_gate/run_all.sh`: resample pools → slot judge → class judge → probe
   texts → capture → fit → score → label → report.
   - The driver runs from a `/tmp` snapshot, because editing a running bash script breaks it.
4. **Gold calibration, results doc, report v2.8, plots.**

## 6. Files

```
plans/0009-sibling-resample-probe-gate.md
src/hallucination_bench/
  taxonomy_v2.py                       + SIBLING_STATUSES, segment_verdict (rev 21)
  evals/wsbench_chat/score_v2.py       uses segment_verdict + status join
  report_v2.py                         sibling-status table, ↳ substitution_sibling column
  evals/sibling_gate/
    __init__.py
    units.py        sibling units from cells.jsonl, normalisation, label join
    resample.py     trimmed manifests per pool, job specs, block-shape check
    fill_judge.py   sibfill-v1 prompt / schema / parse / quote check, decision rule (§2c)
    classes.py      sibclass-v1, class dedupe, concept sets, compatible stratum
    texts.py        sibgen-chat-v1 probe texts + checks, token-class key
    probes.py       vendored from camila/um-ladder-v2 (unchanged)
    um_probe.py     vendored from camila/um-ladder-v2 (unchanged)
    ladder.py       distractors, n/a, fire / nearby / cross-method, matched null, labels
    cli.py          hbench sibling-gate <stage>
scripts/sibling_gate/
  run_all.sh  probes_modal.py  plot_sibling_gate.py
data/sibling_gate/   fill_dev.jsonl, fill_gold.jsonl (labels), classes.json, concepts.json, probe_texts.jsonl
tests/test_sibling_gate.py
docs/results/0009_sibling_gate.md, docs/results/0002_report_full_v2.8.md
plots/sibling_gate/*.png
outputs/sibling_gate/   (gitignored) resample/, fill/, probes/, labels/
logs/sibgate_*.log
```

## 7. Cost estimate (to be updated from the pilot)

- **Resampling:** about 9.3k distinct cells × 10 samples on B300:8 across the 5 pools. The cost
  is mostly model load and prefill.
- **Slot judge:** about 15–25k units × roughly 6k input tokens.
- **Probe texts:** about 72 texts per concept, for at most a few hundred concepts.

No budget cap (Camila). The spend is reported in the results doc.
