# Plan 0027 — hindex-v7: `benign_roleplay` on the index (weight 2), sibling weight 1

Author: Camila Blank (planned with Claude), 2026-11-04. Issue #56. Revision 1.

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
   - Items the eval drops as refused (WSB `refused` state, prompt-only refusals) are not rows:
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
   - The v4 notes guard (`NOTES_V4` == methods.json notes) is unaffected: notes are unchanged.
   - Methods with no BR value have no v7 `full` / `matched`. They still have `full_v6`.
5. **Unit guard**, extending `check_safety_units`:
   - Under v7 every `benign_roleplay` root needs `unit == "prompts"`.
   - v3–v6 never read a BR root, and passing one under v3–v6 is an error.
   - **The v7 overrefusal root keeps v6's rule** (`refusal_claims`).
6. **Opt-in root:** `--benign-roleplay-root` (repeatable) or a set `HBENCH_SAFETY_BR_OUT`; never
   a dev-box default, as for overrefusal. Under v7 without a BR root, BR is absent everywhere, so
   `full` / `matched` are absent. The CLI prints a warning naming the missing root.
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
    metrics.py                                     score_unit: + scored_ids (additive)
    index_export.py                                NEW: summary.json -> index_counts/meta
    cli.py                                         `index-export` subcommand
data/index/methods.json                            benign_roleplay arms for s3d-rl600 / mlnla-rl600
tests/test_index_v7.py                             NEW
README.md, CLAUDE.md                               v7 described; plan numbers (0025/0026 local cheating)
```

## Invariants (restated where the code changes)

- `weighted_macro` gains a `weights` parameter **defaulting to the existing `FAMILY_WEIGHTS`**,
  so every v4–v6 call and its output is unchanged.
- `FAMILIES` / `MATCHED` / `SAFETY_FAMILIES` keep their v4–v6 values. v7 uses new tuples, so
  v4–v6 iterate exactly the families they did. `load_methods` accepts the union of v6 and v7
  families. `resolve_methods` is called with the version's family tuple, so v4–v6 never see the
  `benign_roleplay` key.
- The v3 / v4 / v5 goldens and the v5 / v6 dev-box goldens must pass unchanged, and **no
  pre-existing test is modified**.

## Tests (`tests/test_index_v7.py`)

1. **Synthetic v7 build:** six families on a tiny fixture.
   - `full` = Σ w·v / 17 with the v7 weights.
   - `matched` excludes sibling.
   - `full_v6` equals the v6 build's `full` on the same roots.
2. **v3–v6 output byte-identical** with a BR root absent; v3–v6 refuse a BR root.
3. **Unit guard:** v7 refuses a BR root whose unit is not `prompts`, and keeps v6's overrefusal
   rule.
4. **Export ↔ eval equivalence:** on a synthetic summary + bank, the index value and CI at every
   layer equal `metrics.rate_ci` exactly. On the dev box (skip elsewhere), the same check holds
   on the final-olens `summary.json` for all 11 layers.
5. **Export refusals:** mixed instruments, `n` short without `scored_ids`, and an
   unknown `--arm-map` target.
6. `methods.json` resolves `mlnla-rl600` → `mlnla-rl600-all11` for BR under v7 only.

## Caveats (recorded in NOTES_V7)

- **The unit is prompts.** Every other family counts claims, units or lines, and they are not on
  one scale.
- **BR NLA arm.** The index NLA arm is `mlnla-rl600-all11`, not plan 0023's `mlnla-rl2-2000`.
- **Sampling differs by arm.** The FINAL64 shared readouts are `prose1` (k=1) for s3d-rl600 and
  `bullet4` (k=4) for final-olens. More samples give more chances to recognise, so the arms are
  not like-for-like.
- **WSB instrument.** jb-v3 + digest-v5 + sj-v10 on `claude-sonnet-5`, WSB pin `0b18789`. Every
  BR row on the index must come from that instrument, and the export enforces it.

## Run (after merge; local first, results PR when frozen)

The judged campaign for the three index arms is a separate job (GPU reads of the 64 `hb_only`
items + judging at `0b18789`), launched 2026-11-04. Then:
1. `hbench benign-roleplay report` over the four arms.
2. `index-export`.
3. `hbench index --version v7 --benign-roleplay-root <dest> ...` (+ the v6 roots).
4. The results doc and the CLI default flip go in a results PR.
