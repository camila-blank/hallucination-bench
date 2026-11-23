# The claim map

The claim map is what the index scores. Code cuts every readout into units, and one judge call
puts every unit, or verbatim sub-spans of it, into exactly one **leaf** by comparing it with the
real conversation, so every character of every readout is in exactly one leaf. A segment is
incorrect only if the conversation **obviously contradicts** it, and the judge must quote the
text that rules it out; a claim the conversation merely never mentions is not a hallucination.
Each leaf belongs to one group and maps to one **verdict**: hallucination (`detail`,
`false_claim`, `substitution_category`), probe_candidate (`substitution_sibling`,
`supplementary_false`, `ambiguous_topic`), thinky (`model_state`), correct (`correct`,
`supplementary_true`), unresolved (`unverifiable`, `continuation_miss`) and junk (`junk`). The
headline is the character-weighted verdict share per readout and the readout-level verdict
rates, with `false_claim`, `detail` and `substitution_category` always reported separately next
to the hallucination verdict; groups and leaves are the breakdown. The leaves, with their
meanings and index weights, are tabulated in the [README](../README.md#the-taxonomy).

## Leaves, groups and verdicts

Everything is defined once in
[`src/hallucination_bench/claim_map_taxonomy.py`](../src/hallucination_bench/claim_map_taxonomy.py) (leaves,
groups, precedence, the Detail rule, tags, and the open decisions with their defaults in
`CHOSEN`) and rendered into the judge prompt.

- **The judge chooses among 12 leaves:** `non_claim`, `correct`, `detail`, `false_claim`,
  `substitution_sibling`, `substitution_category`, `model_state`, `unverifiable`,
  `supplementary_true`, `supplementary_false`, `ambiguous_topic` and `off_topic`.
- **Scoring uses 11 of its own leaves:** right after parsing, `non_claim` and `off_topic` become
  one leaf, `junk`. Code adds `continuation_miss` for a contradicted guess of the reply's
  literal next words that the continuation verifier excuses (see
  [hallucination_index.md](hallucination_index.md)), and two code-only rows: `unknown_incorrect`
  (under `anchor=v5c` only) and `failed`.
- **Groups (7):** hallucination (`detail`, `false_claim`), co_activation (the two
  substitutions), correct, supplementary (`supplementary_true`, `supplementary_false`), thinky
  (`model_state`, `unverifiable`, `continuation_miss`), unsettled (`ambiguous_topic`) and junk.
- **Verdicts (6):** as listed above (`claim_map_taxonomy.VERDICTS`). `substitution_category` is a
  hallucination and `substitution_sibling` a probe candidate, although both are in the
  co_activation group. Probe-candidate segments carry a `probe_target`
  (`probe_candidates.jsonl`).

## Rules the judge and the code apply

- **The hallucination gate.** A segment is incorrect only if the conversation obviously
  contradicts it. Predictions get no category of their own: a contradicted prediction is
  incorrect, any other is `unverifiable` with `prediction = true`. Content realised in any
  wording is `correct`.
- **Settlement, not subject.** A world claim is about the text only if the conversation settles
  it; where the text is silent, the claim is `supplementary_true` or `supplementary_false`.
- **Quotes.** Every incorrect segment must quote the conversation span that contradicts it
  (`contradicted_by`), and every `correct` segment, except `source = entailed`, the span that
  supports it (`supported_by`). A missing quote, or one that code cannot find in the
  conversation, fails closed to `unverifiable` (counted). The quotes are carried into
  `cells.jsonl`, `probe_candidates.jsonl`, the audit samples and the report's hallucination
  examples.
- **Tags.** `junk` carries `reason` (`off_topic`, `generic`, `filler`, `formatting`,
  `junk_text`, `lens_meta`), `correct` carries `source` (prefix / after / entailed), and
  `unverifiable` carries `same_slot_alternative`, `world` and `prediction`. Tags are recorded
  and, with one exception, do not change a verdict: a segment the judge tags `hedged` (a list of
  options, "Germany or France or UK") is scored `unverifiable` with `same_slot_alternative`,
  whatever its leaf.
- **Invalid answers.** An invalid judge answer is retried up to 2 times in the same run before
  it is cached as failed.

## Reports and judge consistency

- Reports include a junk-by-reason table (the off-topic sub-share per arm).
- Chat reports judged with the default prompt also carry a judge-noise section and a "judge
  noise ±" column (half the difference between the two runs of each headline metric in the
  consistency sample) once `hbench consistency` has been run on the same arms. Story reports
  and the index's `claimmap-v2.9` roots have no judge-noise numbers.
- Outputs are labelled **pre-gate** unless the held-out calibration gate has passed for the
  prompt in use.
- `hbench consistency arms=s3d-rl600:44,nla-rl-L42:42,jlens:44 n=50 seed=0` runs the
  claim-map call twice per sampled cell and reports κ, raw agreement, the between-run leaf
  confusion, per-leaf self-agreement and the leaf pairs swapped in more than 10% of a leaf's
  units (`outputs/consistency/<model>/`).

## The span taxonomy

The span taxonomy belongs to the second pipeline, the span classifier (the `classify` stage of
`hbench run`). It is defined once in
[`src/hallucination_bench/span_taxonomy.py`](../src/hallucination_bench/span_taxonomy.py) and rendered
into the classifier prompt.

- **grounding** (set in code): `contradicted` (a verified wrong span from the span stage),
  `disputed` (the verify stage thinks the response rules it out, the span stage did not flag
  it), `unsupported` (verify `unverifiable`: specific, but neither stated, entailed nor
  realised).
- **type** — would a reader who believed the span misunderstand what the text is about or
  claims? `concept` (yes: a different entity / idea / event as the subject of a claim, or a
  reversed claim; e.g. "Picasso" for da Vinci) vs `detail` (no: a wrong or invented specific
  value inside a correct understanding; e.g. "Steve" for an arbitrary character named Joe).
  Names follow an identity-first decision order; a value that flips the claim is `concept`;
  magnitude alone raises severity, not type.
- **target**: entity, quantity, event, attribute, stance, structure, topic.
- **severity**: 1 minor, 2 moderate, 3 major.
- **unsupported_kind** (unsupported spans only): prediction, inference, fabricated,
  world_knowledge. The **lenient view** drops inference and world_knowledge; the **strict
  view** (the default definition: only stated or realised content is grounded) keeps all four.
