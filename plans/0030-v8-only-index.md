# Plan 0030: hindex-v8 only (remove index versions v3–v7)

Author: Camila Blank (planned with Claude), 2026-11-18. **Revision 3** (rev 3 = second critique: the rev-3 rules section below; rev 2 = critique of rev 1: judge-noise tooling moved out of scope; every `version` parameter kept; the retained-names rule; the test files are split, none deleted whole; no plan is deleted). Follows PR #68 (the repo trimmed to the
v8 benchmark by deletions). Camila: "this should be the minimal hallucination bench eval
benchmark"; asked whether to also remove the legacy scoring code, she said yes.

## Goal

`main` carries one index definition, hindex-v8, and no tooling that v8 does not use. The frozen
v8 numbers and every file a v8 run writes stay byte-identical.

## Hard invariants (checked before merge)

1. **Frozen index.** `hbench final --manifest data/index/v8_manifest.json --methods
   data/index/methods.json` against the archived judge outputs writes `index.json` (apart from
   `git_sha` and the manifest's absolute path), `index.md`, `excluded.md` and the three plots
   byte-identical to the same command on `main` @ 37d92be.
2. **The judge pipeline is not touched.** No change to `evals/*/run.py`, `evals/*/score*.py`,
   `evals/overrefusal/*`, `evals/benign_roleplay/*`,
   `evals/sibling_probe/*`, `judges/*`, `taxonomy.py`, `taxonomy_v2.py`, `units.py`,
   `readouts.py`, `summarizer.py`, `llm.py`, `cache.py`, or any file under `data/` other than
   those listed for deletion. So prompts, fingerprints, cache keys and rebuilt `cells.jsonl` /
   `summary*.json` cannot change. The v5c span judge, taxonomy v1 and the claimmap-v2.7 / v2.8
   registry entries therefore STAY: a v8 run still imports and partly executes them, v2.9 is
   layered on their text, and removing them would change what a rebuild writes.
3. **The v8 index output keeps every column it has today**, including `full_v7` / `matched_v7`
   and `full_j6`. **Retention rule:** a name in `index.py` is deleted only if, after the change,
   no kept source file, script or kept test references it AND the v8 build / renderer does not
   reach it. Known to be retained *as internals of the v8 build* even though hindex-v3 … v7
   are no longer buildable: `FAMILIES_V7`, `MATCHED_V7`, `FAMILY_WEIGHTS_V7`,
   `SAFETY_FAMILIES_V7`, `SAFETY_READERS_V7_ADDED` (read by
   `evals/benign_roleplay/index_export.py`), `AM`, `AM_LIKE`, `ALL_FAMILIES`,
   `CLAIM_FAMILIES_ANY`, `AM_CANDIDATE_VERSIONS`, `REFUSAL_CLAIMS_VERSIONS`, `SHARE_V6`,
   `SHARE_BR`, `ABSENT_*`, `_md_footer`, `_md_absent`, `_pct`, `NOTES_V8_TEMPLATE` /
   `notes_v8`. When in doubt, keep.
4. `data/index/methods.json`, `v8_manifest.json`, `v8_persistent_failures.json` are unchanged
   (the manifest hashes them; v8 merges `methods.json`'s notes into its own).

## Changes

### A. `index.py`: v8 only

- `build_index(sources, methods, workers, version)` builds only v8: `version` defaults to
  `"v8"` and any other value raises `ValueError("hindex-v8 is the only index version")`. Remove
  the v3–v7 body of `build_index`, `_render_md_v3` … `_render_md_v7` (and their dispatch in
  `render_md`), `NOTES_V3` … `NOTES_V7`, `_claim_job` if only the v3–v7 body calls it, and the
  constants only they reach (subject to invariant 3's retention rule).
- **Every public helper keeps its `version` parameter** (`safety_families`, `families_for`,
  `claim_families`, `safety_readers`, `safety_rows`, `check_safety_units`, `default_sources`,
  `claim_metric`, `weighted_macro`, …): the default becomes `"v8"`, the v8 behaviour is
  unchanged, and any other value raises the same `ValueError`. Kept tests and
  `evals/*/index_export.py` call them with `"v8"`.
- `VERSIONS` keeps only `"v8"`; `INDEX_VERSION` becomes `"hindex-v8"` and `INDEX_VERSION_V8`
  stays. `AM_CANDIDATE_VERSIONS` / `REFUSAL_CLAIMS_VERSIONS` become `frozenset({"v8"})`.
- The unit guard keeps its v8 rule (overrefusal root `unit == "refusal_claims"`,
  benign_roleplay `prompts`), and a cheating root is still refused.
- Module docstring rewritten for v8 only.
- `final.py` is edited only if it calls a changed signature (it calls `build_index_v8`
  directly, so probably not at all).

### B. CLI

- `hbench index`: `--version` keeps a single choice, `v8` (default), so existing command lines
  keep working; the v3–v7 branches of `cmd_index` go.
- `hbench report` no longer refreshes the index (that refresh built hindex-v4). `cmd_report`
  becomes `_cmd_report`; `_index_version` goes if unused. `hbench index` / `hbench final` are
  the only index builders.
- **Out of scope, unchanged:** `consistency-v2` and `consistency_v2.py`, the judge-noise code in
  `report_v2.py`, `calibrate`, `pilot-dump`, `pilot-dump-v2`, `units-v2`, `calibrate.py`,
  `calibrate_v2.py`, `data/calibration/`, `scripts/build_dev_v2.py`. Kept parser tests import
  them (five `test_claimmap_v2_rev*.py` files import `consistency_v2` at module level, about
  ten read the gold), `run.py` imports `gate_passed`, and `report_v2.json` carries a
  `judge_noise` key.

### C. Tests and fixtures

No index test file is deleted whole: each is split by what its tests cover.

- **Delete** tests whose subject is a v3–v7 index *output* or the report refresh: the golden
  comparisons (`tests/fixtures/index_v3/`, `index_v3_devbox.*`, `index_v4/`,
  `index_v4_devbox.*`, `index_v5_devbox.*` go with them), `--version v3…v7` CLI behaviour,
  `NOTES_V4 == methods.json`, the v3–v7 parametrisation at `tests/test_index_v8.py:153-167`,
  `hbench report` writing / refusing to overwrite an index, and the v6-only reader allowlist
  tests in `tests/test_final_olens_arm.py` (with `SAFETY_READERS_V6_ADDED`, if nothing else
  reads it).
- **Keep** every test of code that stays, in `tests/test_index_core.py` (or the original file
  if it still reads well): um label / sibling metric, `weighted_macro`, `load_methods` errors,
  `resolve_methods` (from `test_index.py`); `safety_metric`, the absent rules, the sorted
  cluster bootstrap, allowlist + columns, two roots (from `test_index_v4.py`); `is_candidate`,
  candidate `claim_rows`, merged mentions (from `test_index_v5.py`); the overrefusal
  `offtopic-export` tests (from `test_index_v6.py`); the benign_roleplay `index-export` tests
  (from `test_index_v7.py`); `test_benign_roleplay_nested_campaign_arm`.
- **Sanctioned rewrites of kept tests** (and only these): a `version="v3"…"v7"` argument
  becomes `"v8"` when the test's subject is not the version and the assertion still holds
  under v8 unchanged; if it does not hold under v8 (e.g. `test_index_v7.py:377` reads
  `final-olens-rl600-x4`, which the v8 allowlist excludes; a v4 `unit: items` overrefusal
  root), delete that test and list it in the hand-back. `tests/test_benign_roleplay.py`
  `test_on_the_index_from_v7_only` becomes: benign_roleplay is in `safety_families("v8")` and
  a non-v8 version raises `ValueError`. In `tests/test_index_v8.py`: `:173-174` drop the
  `"v7"` comparison; the `:271` / `:282` loops over old versions reduce to v8; `:291-304`
  asserts that `hbench index --version v7` exits non-zero (argparse `choices`).
  `tests/test_index_cli_default_v8.py` gains the same `--version v6` refusal.
- Helpers imported across test files (`_or_row`, `_v6_root` from `test_index_v6`; `_summary`,
  `_items`, `_unit`, `INST`, `ITEMS` from `test_index_v7`; anything `test_joint` /
  `test_final*` / `test_continuation_verifier` import) move to `tests/index_helpers.py`.
  Helpers with no surviving caller (`_sources_v5`, `_canon`, `_golden_methods`,
  `_pinned_roots_v4`) are deleted, not moved.
- **Every other pre-existing test must pass unmodified** apart from the import path of a moved
  helper. If one fails, stop and report it rather than editing its assertions.

### D. Docs

- `README.md` "Hallucination index": remove the sentence about v3–v7 remaining in the code
  and "`hbench report` refreshes it"; state that v8 is the only version.
- `CLAUDE.md`: shorten the hindex-v3 … v7 paragraphs in the intro and the "Hallucination
  index" gotcha to what v8 inherits (definitions by plan), dropping `--version v3…v7`, the
  library-default-v4 statements, the goldens and anything naming a deleted test / fixture;
  keep everything about v8. Plan numbers: next free is 0031.
- **No plan is deleted.** Plan 0029 defines the v8 families by reference to 0013, 0019, 0021,
  0022 and 0027, and the frozen output's notes cite 0014, 0019, 0021 and 0027. README /
  CLAUDE.md keep their links to them as the definitions v8 inherits.

### Rev-3 rules (these override anything above they conflict with)

- **`weighted_macro` and `check_safety_units` keep their signatures unchanged** (neither has a
  `version` default to flip). Also retained, because kept tests and `weighted_macro`'s default
  use them: `FAMILY_WEIGHTS`, `FAMILY_WEIGHTS_V3`, `SAFETY_FAMILIES`, `FAMILIES_V3`,
  `MATCHED_V3`.
- **A call that relied on the implicit old default (`"v4"`) is treated like an explicit old
  version** under the sanctioned-rewrite rule: keep the test if its assertions hold unchanged
  under v8, otherwise handle it as listed here.
  - `tests/test_index_v4.py` `test_safety_arm_under_two_roots`: keep; the fixture gains
    `meta={"unit": "refusal_claims"}` so the v8 unit guard passes and the "under two roots"
    error is still the one asserted.
  - `tests/test_index.py` `test_unweighted_share_and_am_reader` (old unweighted story / AM
    metric): delete.
  - `tests/test_index_v5.py` `test_versions` (asserts `INDEX_VERSION == "hindex-v4"` and the
    full `VERSIONS` dict): delete.
  - `tests/test_index_v8.py:146-167`: `test_claim_families_v3_to_v7_unchanged` keeps only its
    `claim_families("v8")` assertion; the v3–v7 parametrised test after it is deleted.
- **CLI:** delete the `cmd_report` wrapper and point the `report` subparser's
  `set_defaults(fn=...)` at `_cmd_report`.
- **README:** do not touch the `## benign_roleplay` section; `tests/test_benign_roleplay.py`
  asserts its sentence "on the index from hindex-v7 (plan 0027)".
- Any other kept test that fails and is not named in this plan: stop and list it in the
  hand-back with the failure; do not edit or delete it.

## File tree (changed / deleted)

```
src/hallucination_bench/index.py            M  v8 only
src/hallucination_bench/cli.py              M  index: v8 only; report: no index refresh
src/hallucination_bench/final.py            M  only if a signature changed
tests/index_helpers.py                      A  shared helpers
tests/test_index_core.py                    A  kept unit tests from test_index{,_v4,_v5}.py
tests/test_index.py, test_index_v4.py, test_index_v5.py, test_index_v6.py, test_index_v7.py
                                            split (kept tests moved; the rest deleted)
tests/test_index_v8.py, test_index_cli_default_v8.py, test_benign_roleplay.py,
tests/test_final_olens_arm.py               M  the sanctioned rewrites
tests/fixtures/index_v3*, index_v4*, index_v5_devbox.*                  D
README.md, CLAUDE.md                        M
```

## Verification

`uv run pytest`, `uv run ruff check`, `uv run ruff format --check`; invariant 1's rebuild is
run by the main session against `/dev/shm/hb-v8-full` (no API calls, about 10 minutes); a
fresh-context review of the diff with `git diff origin/main -- tests/` read separately.
