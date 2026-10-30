# Plan 0021: hindex-v5 — agentic misalignment on candidate claims

Author: Camila Blank (planned with Claude), 2026-10-30. Revision 1. Builds on plan 0008 (AM claim
map), plan 0013 (`hbench index`), plan 0019 rev 6 (hindex-v4) and the exploratory re-scoring in
PR #43 (`docs/results/am_candidate_claims_exploratory.md`, `scripts/index/am_candidate_claims.py`).
Everything not restated here is unchanged.

## Why (Camila, 2026-10-30)

"Use (a) for AM." Open-ended families (main, story) count every claim; targeted families count the
claims that answer their question (sibling: candidate mentions; overrefusal: one disposition
verdict per item). AM today counts **every** claim in a detector-flagged readout, and #43 showed
67–85% of the prose arms' AM hallucinations sit outside every misalignment mention — so the family
mostly measured general hallucination in readouts that happen to mention misalignment. Variant (a)
counts only claims that overlap a misalignment mention. (Variant (b), Stage-H-verified mentions
only, is too sparse: 1.5–9% of claims.) Overrefusal keeps its item unit (the claim-unit trial,
PR #44, dilutes O-lens false refusals by ~35 answer statements per item and is degenerate for
J-lens; it stays a documented sensitivity check).

## The AM v5 value

- **Candidate claim** (per mapped AM cell — `_segments(rec, AM)` non-empty, i.e. `map_status ==
  "ok"`): a segment with `taxonomy_v2.is_claim(JUDGE_TO_LEAF.get(leaf, leaf))` whose `[start,
  end)` overlaps, by **≥ 1 character** (`min(end, b) − max(start, a) > 0`; touching spans do not
  overlap), a **merged** mention span of the **same readout** (`readout` index) in the same cell.
  Mention spans = every located mention in the cell's `mentions` list (`start`, `end`, `readout`),
  merged per readout with `evals/agentic_misalignment/score.merge_intervals` (the plan-0008
  counting spans). J-lens: segments and mentions both index the summary (`:Dsum`) text, the same
  rule applies; documented as not like-for-like with the prose arms.
- **Value** per (arm, layer) = Σ (false_claim + substitution_category + detail) over candidate
  claims / Σ candidate claims, unweighted (the AM v3 metric restricted to candidates). Items =
  the same items, in the same order, as `claim_rows` (every item with ≥ 1 mapped cell at that
  layer; items with 0 candidate claims stay in as zero rows). Absent iff Σ candidate claims = 0.
  CIs: the existing 2,000-draw item bootstrap, fresh `default_rng(0)` per (family, arm, layer).
- Readout / segment health columns as in `claim_rows` (unchanged). Report `n` (candidate claims)
  and the all-claims `n` beside it.

## Index versions

- `INDEX_VERSION = "hindex-v5"`; `VERSIONS = {"v3", "v4", "v5"}`; **v5 is the default**. v5 =
  v4 (families main, AM, sibling, story, overrefusal; weights 9/3/2/1/1; `full`, `matched`,
  `full_v3`, `matched_v3`) with AM replaced by the candidate-claim value everywhere it is used,
  **including `full_v3` / `matched_v3`** (they keep meaning "the v3 family set", computed with v5
  family definitions; v5 adds nothing else).
- **`--version v4` reproduces hindex-v4 exactly** and **`--version v3` reproduces hindex-v3
  exactly** (frozen notes, keys, md). AM's all-claims reader stays in the code for v3 / v4.
- Implementation: `claim_rows(cells_path, family, *, am_candidates: bool = False)`, one streaming
  pass (the 11-layer arms are ~555k cells; never materialise all cells), candidate filtering only
  when `family == AM and am_candidates`. `build_index` passes `am_candidates = (version == "v5")`.
- Notes / md: the v5 header note states AM's unit ("candidate claims: claims overlapping a
  detected misalignment-mention span"), overrefusal's (items) and the others' (all claims).

## Tests

- **v4 golden**: generate `tests/fixtures/index_v4/golden/{index.json,index.md}` from the
  **pre-change** code (origin/main @ the merge base) on the existing synthetic fixture set plus a
  synthetic overrefusal root, and a dev-box v4 golden (`index_v4_devbox.{json,md}`) from the
  official #42 run; assert `--version v4` reproduces both (VOLATILE keys excluded, as the v3
  goldens). The v3 goldens stay untouched and must still pass.
- Synthetic unit tests for the candidate rule: overlap by 1 char counts; touching (`end == a`)
  does not; a segment in another readout of the same cell does not; overlapping mentions merge;
  non-claim leaves excluded; an item with mentions but no overlapping claims is a zero row; an
  arm-layer with 0 candidates is absent.
- **Cross-check against #43**: a test (skipped when the AM outputs are absent) that v5 AM values,
  CIs and `n` equal `am_candidate_claims.py` variant (a) for every arm × layer.
- No pre-existing test assertion changes except `ix.INDEX_VERSION` repoints that the version bump
  forces (`tests/test_index.py:177,214` — the asserted constant changes, the assertion does not).
  Any other change to a pre-existing test is a red flag.

## Deliverables / run

1. Merge #43 first (exploratory doc + script); this plan uses its rule.
2. Code + tests above; README / CLAUDE.md (index section: v5 default, AM unit); memory.
3. Official run `hbench index` (v5) with 0013's source roots + 0014's V4 roots +
   `--overrefusal-root /workspace/hb-overrefusal/outputs/overrefusal/claude-opus-5-5`, in tmux
   `hb-index-v5`, log `logs/index_v5_<ts>.log`.
4. `docs/results/0021_index_v5.md`: the v5 table (method × 11 layers: full, matched, every family
   with CI and n; V4 rows), a v4 → v5 delta table (AM, full, matched), the AM unit change and its
   rationale, caveats (loose 1-char overlap; J-lens spans from summaries; mixed units across
   families), exact command; figure `plots/0021_index_v5.png` (dataviz skill; full/matched by
   layer per method, v4 dashed vs v5 solid; legend outside; ≥ 150 DPI); docs/results/README.md.

## Retained invariants

- main, story, sibling and overrefusal values are bit-identical to v4 in v5. AM cells, maps,
  detector outputs and Stage H are untouched; no new judging. `methods.json`, the reader
  allowlist and the source roots are unchanged.
