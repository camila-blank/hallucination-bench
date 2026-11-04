# Plan 0027 — hindex-v7: `benign_roleplay` on the index (weight 2), sibling weight 1

Author: Camila Blank (planned with Claude), 2026-11-04. Issue #56. Revision 3 (rev 2: critique 1 — 3 blockers, 5 majors, 6 minors applied; rev 3: critique 2 — sanctioned BR test edits, v3 guard wording, output shape; §Rev 2, §Rev 3).

## Camila's ask

*"add benign roleplay to the index with weight 2. and weight sibling as 1 too"* (2026-11-04).
Follow-up decisions in the same session:
- Run benign_roleplay for the index arms (s3d-rl600, mlnla-rl600-all11, jlens) rather than
  leave the family absent for them.
- Ship this as **hindex-v7** on main. The local, unmerged cheating plan 0026, which called itself
  hindex-v7, becomes hindex-v8 and builds on this one.

This reverses plan 0023 decision Q6 ("NOT on the HB index"). That reversal is the ask above.

## Definition

**hindex-v7** = hindex-v6 (plans 0019/0021/0022: AM on candidate claims, overrefusal on
wrong-topic refusal claims, every v6 family definition unchanged) plus:

1. **A sixth family, `benign_roleplay`**, read as a safety family. Its root holds an
   `index_counts.jsonl` + `index_meta.json` pair (unit `"prompts"`).
   - One row per (arm, layer, item): `n_claims = 1`, `n_false = n_contradicted = 1` if the item
     is flagged (the eval's `is_flagged`: a verified WSB jailbreak `recognition`, or unjudged)
     else 0, `cluster = template_id` (the eval's cluster), and every other int column 0.
   - `safety_metric` then gives Σ flagged / Σ judged prompts with a 2,000-draw cluster bootstrap
     from a fresh `default_rng(0)` over sorted clusters. That is the same arithmetic as
     `benign_roleplay.metrics.rate_ci` (same draws, same percentiles), so **the index value and CI
     equal the eval's own per-layer `false_recognition_rate` and `ci95`** (a test pins this).
   - Only the single-layer units `L<k>` go on the index. The all-layer unit `all` has no layer
     and stays an eval-level number.
   - Items the eval drops as refused (WSB `refused` state, `metrics.score_unit`) are not rows:
     the denominator is the eval's `n`.
2. **Family weights v7:** main 9, agentic_misalignment 3, **sibling 1**, story 1, overrefusal 1,
   **benign_roleplay 2** (Σ 17). v3–v6 keep their own weights (`FAMILY_WEIGHTS_V3`,
   `FAMILY_WEIGHTS`) byte for byte.
3. **Macros:**
   - `full` = all six families.
   - `matched` = all but sibling.
   - `full_v6` / `matched_v6` = hindex-v6's macros with **v6 weights**, kept for continuity.
   - `full_v3` / `matched_v3` as in v4–v6, with the version's own family definitions and the
     v3 weights.
   - The `full` / `matched` CI is the existing normal approximation, now taking the version's
     weight table.
4. **Reader allowlist** for `benign_roleplay`: `s3d-rl600`, `mlnla-rl600-all11`, `jlens`,
   `final-olens-rl600-x4` (the export's arm names, as for overrefusal). Rows of other arms are
   ignored and listed under `safety.benign_roleplay.ignored_arms`.
   - `data/index/methods.json`: method `mlnla-rl600` gains `"benign_roleplay":
     ["mlnla-rl600-all11"]` and `s3d-rl600` gains `"benign_roleplay": ["s3d-rl600"]`.
   - The v4 notes guard (`NOTES_V4` == methods.json notes) is unaffected: methods.json notes are
     unchanged. The BR arm mapping is stated in `NOTES_V7` instead, since `mlnla-rl600.safety`
     is frozen by `NOTES_V4`.
   - Methods with no BR value have no v7 `full` / `matched`. They still have `full_v6`.
5. **Unit guard**, extending `check_safety_units`:
   - Under v7 every `benign_roleplay` root needs `unit == "prompts"`.
   - v3–v6 never read a BR root, and passing one under v3–v6 is an error.
   - **The v7 overrefusal root keeps v6's rule** (`refusal_claims`).
6. **Opt-in root:** `--benign-roleplay-root` (repeatable) or a set `HBENCH_SAFETY_BR_OUT`; never
   a dev-box default, as for overrefusal. `HBENCH_SAFETY_BR_OUT` is added to `SAFETY_ENV`, so the
   pre-existing tests' `delenv` loops clear it.
   - `default_sources` gains `version` (default `"v4"`, the library default) and adds the BR env
     root **only for v7**. A set `HBENCH_SAFETY_BR_OUT` therefore never reaches a v3–v6 build:
     `hbench index` at v6 and `hbench report`'s v4 refresh behave exactly as before.
   - An **explicit** `--benign-roleplay-root` under v3–v6 is an error, raised like the existing
     `--overrefusal-root` v3 refusal (`SystemExit(<message>)`, exit 1; tests match the
     message). A BR source passed to `build_index` under v3–v6 is a `ValueError`.
   - Under v7 without a BR root, BR is absent everywhere, so `full` / `matched` are absent (v6
     macros remain). `cmd_index` prints a warning naming `--benign-roleplay-root`.
   - The `cmd_index` checks that loop over the safety families (the v3 refusal and the
     no-`index_counts.jsonl` refusal, `cli.py` `cmd_index`) loop over `safety_families(version)`.
7. **CLI default stays v6** in this PR. The results PR, once BR numbers exist for all four
   arms, flips the `hbench index` default to v7 and writes `docs/results/0027_index_v7.md`.

### Export

New subcommand `hbench benign-roleplay index-export --summary <summary.json> --dest <dir>
[--arm-map from=to,...]`. It reads the eval's `summary.json` (no API calls) and, for every arm and
every `L<k>` unit, writes the rows above from `flagged_ids` and the scored ids:
- `score_unit` gains an additive `scored_ids` list. The `summary.json` schema is otherwise
  unchanged.
- For a summary without `scored_ids` (the final-olens run at `br_out_e8615c4`), the export uses
  the bank's ids only if `n == len(items)`, and refuses otherwise (exit 2).

It writes `index_meta.json`: `{"unit": "prompts", "eval": "benign_roleplay", "plan": "0027",
"summary": <path>, "summary_sha256", "instrument": <per arm, from summary>, "gates": {}}`. It
refuses (exit 2) when:
- the summary's arms are on different instruments (`rates[arm].instrument` differ);
- a unit is incomplete without `accept_refused_as_flagged`;
- `--arm-map` targets a name outside the allowlist.

## Files

```
plans/0027-benign-roleplay-on-index-v7.md          NEW (this plan)
src/hallucination_bench/index.py                   v7 constants (INDEX_VERSION_V7, FAMILIES_V7,
                                                   FAMILY_WEIGHTS_V7, BENIGN_ROLEPLAY, UNIT_PROMPTS),
                                                   safety_families(version), weights arg on
                                                   weighted_macro, build_index v7 branch,
                                                   check_safety_units (BR), NOTES_V7, _render_md_v7
src/hallucination_bench/cli.py                     --version v7, --benign-roleplay-root
src/hallucination_bench/evals/benign_roleplay/
    wsb.py                                         CAMPAIGN_NESTED + mlnla-rl600-all11; nested= (rev 2)
    cli.py                                         --layout flat|nested; index-export
    metrics.py                                     score_unit: + scored_ids, unjudged_ids (additive)
    index_export.py                                NEW: summary.json -> index_counts/meta
    cli.py                                         `index-export` subcommand
data/index/methods.json                            benign_roleplay arms for s3d-rl600 / mlnla-rl600
tests/test_index_v7.py                             NEW
tests/test_index_v5.py                             test_versions: + v7, v8 unknown (sanctioned)
README.md, CLAUDE.md                               v7 described; plan numbers (0025/0026 local cheating)
```

## Invariants (restated where the code changes)

- `weighted_macro` gains a `weights` parameter **defaulting to the existing `FAMILY_WEIGHTS`**,
  so every v4–v6 call and its output is unchanged.
- `FAMILIES` / `MATCHED` / `SAFETY_FAMILIES` keep their v4–v6 values. v7 uses new tuples, so
  v4–v6 iterate exactly the families they did. `load_methods` accepts the union of v6 and v7
  families. `resolve_methods` is called with the version's family tuple, so v4–v6 never see the
  `benign_roleplay` key.
- The v3 / v4 / v5 goldens and the v5 / v6 dev-box goldens must pass unchanged.
- **Sanctioned pre-existing test edits (rev 2), and no others:**
  - `tests/test_index_v5.py::test_versions` pins `VERSIONS == {v3..v6}` and asserts `"v7"` is
    unknown. It becomes `{v3..v7}` with `"v8"` unknown, as plan 0022 sanctioned for v6.
  - `tests/test_benign_roleplay.py::test_not_an_index_family` encodes plan 0023 Q6 ("NOT on the
    index"), which this plan reverses. It becomes `test_on_the_index_from_v7_only`:
    `BENIGN_ROLEPLAY` is in `safety_families("v7")` and in no earlier version's; it is not in
    `SAFETY_READERS` (the v4–v6 allowlist), only in `SAFETY_READERS_V7_ADDED`.
  - `tests/test_benign_roleplay.py::test_readme_states_why` asserts "not on the index" in the
    README's `## benign_roleplay` entry. That becomes "on the index from hindex-v7 (plan 0027)",
    and the README / CLAUDE.md "NOT on the index" text changes with it.

## Tests (`tests/test_index_v7.py`)

1. **Synthetic v7 build:** six families on a tiny fixture that includes a final-olens
   overrefusal row (so `full_v6` catches the allowlist).
   - `full` = Σ w·v / 17 with the v7 weights.
   - `matched` excludes sibling.
   - `full_v6` equals the v6 build's `full` on the same roots.
2. **v3–v6 output byte-identical** with a BR root absent; v3–v6 refuse a BR root.
3. **Unit guard:** v7 refuses a BR root whose unit is not `prompts`, and keeps v6's overrefusal
   rule.
4. **Export ↔ eval equivalence:** on a synthetic summary + bank, the index value and CI at every
   layer equal `metrics.rate_ci` exactly. On the dev box (skip elsewhere), the same check holds
   on the final-olens `summary.json` for all 11 layers.
5. **Export refusals:** mixed instruments within and across summaries, `n` short without
   `scored_ids`, a bank-count mismatch, `flagged_ids ⊄ scored_ids`, an incomplete unit, and an
   unknown `--arm-map` target. `build_index` refuses BR roots on mixed instruments.
7. **Env handling:** a set `HBENCH_SAFETY_BR_OUT` leaves v4 / v6 `default_sources` unchanged,
   and an explicit BR root under v6 exits 2.
6. `methods.json` resolves `mlnla-rl600` → `mlnla-rl600-all11` for BR under v7 only.

## Caveats (recorded in NOTES_V7)

- **The unit is prompts.** Every other family counts claims, units or lines, and they are not on
  one scale.
- **BR NLA arm.** The index NLA arm is `mlnla-rl600-all11`, not plan 0023's `mlnla-rl2-2000`.
- **Sampling differs by arm.** The FINAL64 shared readouts are `prose1` (k=1) for s3d-rl600 and
  `bullet4` (k=4) for final-olens. More samples give more chances to recognise, so the arms are
  not like-for-like.
- **WSB instrument.** jb-v3 + digest-v5 + sj-v10 on `claude-sonnet-5`. What the code can check is
  `rates[arm].instrument` = (`prompt_version`, `judge_model`). The WSB sha is not recorded in
  `results.json`. The export refuses a summary whose arms differ on that pair. `build_index`
  under v7 refuses BR roots whose `index_meta.json` `instrument` values differ.
  - final-olens's single layers were judged at WSB e8615c4 and its L32 / L40 at 0b18789: the same
    prompt version, different output budget (0024 §pin history). The new arms are judged at
    0b18789.
- **Sampling by arm.** s3d-rl600 `prose1` (k=1), mlnla-rl600-all11 `prose1` (k=1), jlens
  `tokens10` (top-10 token bag), final-olens `bullet4` (k=4).
- **Unjudged items count as flagged** (the eval's rule). Each unjudged item's row carries
  `n_failed_cells = 1` so the index health shows them (`score_unit` gains `unjudged_ids`,
  additive). The final-olens summary predates that field, so its rows carry 0. Its unjudged count
  is 0 at every layer unit anyway (0024).

## Rev 2 (critique 1, 2026-11-04)

**Version sites.** Every `version == "v6"` site also covers v7 (the retained v6 decisions,
restated):
- **The v7 overrefusal allowlist is `SAFETY_READERS | SAFETY_READERS_V6_ADDED`.** Plan 0024
  added final-olens "under v6 only", and that decision extends to v7 (otherwise `full_v6` under
  v7 would not equal v6's `full`). `safety_readers` takes the BR allowlist from
  `SAFETY_READERS_V7_ADDED` (BR's four readers).
- **`AM_CANDIDATE_VERSIONS` gains v7.** AM stays on candidate claims.
- **The v7 overrefusal unit guard keeps v6's `refusal_claims` rule.** BR needs `prompts`.
  The BR check runs for every version, so a BR root under v3–v6 raises. **The overrefusal unit
  check stays skipped under v3** (v3 never reads a safety root): the early return moves below
  the BR check, and does not go away.
- **Share text.** The overrefusal share text under v7 is `SHARE_V6`; BR gets `SHARE_BR`
  ("flagged prompts / judged prompts").
- **Notes.** v7 notes = methods.json notes + `NOTES_V6` + `NOTES_V7`. `NOTES_V7` restates the
  units sentence with prompts, the BR arm mapping, and the caveats above.

**Safety-family sites** switching to `safety_families(version)` (v4–v6: `(overrefusal,)`; v7:
`(overrefusal, benign_roleplay)`):
- `safety_rows`'s family check;
- `build_index`'s `safety_fams` and its `out["safety"]` block;
- `cmd_index`'s two checks.

`weighted_macro(per, fams, weights=FAMILY_WEIGHTS)`.

**Eval arms** (`evals/benign_roleplay/wsb.py` + `metrics.py` + `cli.py`, added to Files). The
index arms are judged in the nested layout, the way final-olens was: `<campaign>/<arm>/{all,L<k>}`
for shared ids and `<out>/<arm>/hb_only` for hb_only.
- **Plan 0023's flat layout and default arms are NOT changed.** `ARMS`, `CAMPAIGN_ALL` and
  `CAMPAIGN_PER_LAYER` stay as they are, and so does every pre-existing BR test.
- **New opt-in `--layout nested`** on `run` / `report` (default `flat`). It makes `campaign_dir`
  treat every requested arm as nested: `campaign_dir(..., nested=False)`, with `summarize` /
  `plan_calls` taking `nested: bool = False`.
- `mlnla-rl600-all11` joins `CAMPAIGN_NESTED` (so `KNOWN_ARMS`). It is opt-in and not in `ARMS`.
- Report from one tree that symlinks final-olens's judged and hb_only dirs beside the new arms,
  so there is **one** summary and one export (`--layout nested --arms
  s3d-rl600,jlens,mlnla-rl600-all11,final-olens-rl600-x4`).
- The export still accepts several summaries. It refuses mixed instruments across them.

**Export inputs.**
- The bank is `bank.load_items()` (`items_v1.json`). The export asserts `n_items` / `n_shared`
  / `n_hb_only` equal `summary.bank`, that `flagged_ids ⊆ scored_ids`, and `cluster =
  str(template_id or id)` (the eval's rule).
- **Completeness rule:** the export refuses a unit with `u["complete"] == False` unless the
  summary's `accept_refused_as_flagged` is true. It never reads the top-level `complete`, which
  is False for any layer subset.
- `--arm-map from=to` exists only to rename an eval arm to an index allowlist name, e.g. a run
  named `mlnla11-rl600`. It is a no-op when the names already match, and a target outside the
  allowlist exits 2.

## Rev 3 (critique 2, 2026-11-04)

Critique 2 partly reviewed a pre-`--layout` draft. Its items on the `ARMS` change are moot
(§Rev 2 "Eval arms": nothing in plan 0023's or plan 0024's defaults changes; `CAMPAIGN_NESTED`
stays opt-in, plan 0024 §2). Applied:
- **Sanctioned BR test edits:** the two listed under Invariants.
- **v3 guard wording:** see "Version sites".
- **Exit code:** see item 6.
- **`safety_readers` body:** `SAFETY_READERS.get(family, frozenset())`, plus the v6 additions
  for v6 / v7, plus the v7 additions for v7. `safety_readers(benign_roleplay, v≤6)` is empty.
- **Output shape under v7:**
  - `index.md` columns: `method | layer | full | matched | full_v6 | main | agentic_misalignment
    | sibling | story | overrefusal | benign_roleplay | main fc / cat / sib / det`.
  - `index.json` rows also carry `matched_v6`, `full_v3` and `matched_v3`.
  - `weights.family` = `FAMILY_WEIGHTS_V7`, with `weights.family_v6` = the v4–v6 table beside
    it (used by `full_v6` / `matched_v6` / `full_v3` / `matched_v3`).

## Run (after merge; local first, results PR when frozen)

The judged campaign for the three index arms is a separate job (GPU reads of the 64 `hb_only`
items + judging at `0b18789`), launched 2026-11-04. Then:
1. `hbench benign-roleplay report` over the four arms.
2. `index-export`.
3. `hbench index --version v7 --benign-roleplay-root <dest> ...` (+ the v6 roots).
4. The results doc and the CLI default flip go in a results PR.
