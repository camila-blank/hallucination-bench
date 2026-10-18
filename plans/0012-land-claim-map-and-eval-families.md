# Plan 0012: land the claim-map judge and the new eval families on `main`, production-ready

Issue #14. Author: Camila Blank (planned with Claude), 2026-10-18. Revision 2. Critique 1 led to
four changes: PRs A–C land as merge commits, not squash; the country ladder's lexicon is pinned;
CI dependencies are pinned down; the path rules are split by kind.

## Why

`main` holds only plan 0001 (the bootstrap, PR #2). Every later piece of work is on a stack of
unmerged branches, so a fresh clone of the repo has none of it. Several branches also only run
on the dev box: they hardcode `/workspace/...` paths, and some tests skip silently elsewhere.

## Scope (Camila, 2026-10-18)

**In:** the claim-map judge (taxonomy v2, plan 0002) and the eval families:

| Plan | Eval | Source branch (tip) |
|---|---|---|
| 0002 | claim map / taxonomy v2 judge | `camila/taxonomy-v2` **at `f969f83`** (see below) |
| 0003–0005 | sibling-concept probes, 11-layer verdict judge, basic-readout mention ladder | `camila/sibling-probes` (`527378c`) |
| 0006 + "0007" | user-modelling country ladder, 4-domain ladder | `camila/um-ladder-v2` (`653f0eb`) |
| 0007 | story eval | `camila/story-eval` (contained in `am-halluc`) |
| 0008 | agentic-misalignment claim map | `camila/am-halluc` (`d8d1acb`) |

**Out, do not land:** the resampling sibling gate and its pipeline step (plans 0009/0010, PRs #11
and #13), the readout viewer (`camila/readout-viewer`), and the WIP layer-plot snapshot
(`camila/taxonomy-v2-layer-plots`). Their branches stay on GitHub untouched.

`camila/taxonomy-v2`'s tip already has #11 and #13 merged into it, so the judge lands from
`f969f83` ("Judge-tree infographic…"), its last commit before #11.
- `am-halluc` and `story-eval` contain `f969f83`.
- `sibling-probes` and `um-ladder-v2` fork from an earlier taxonomy-v2 commit (`81a37e3`).
- No in-scope branch contains or imports sibling-gate code.

**Check:** after each landing PR, no file from `src/hallucination_bench/evals/sibling_gate/`,
`data/sibling_gate/`, `scripts/sibling_gate/` or plans 0009/0010 is on the branch. This command
must print nothing:

```
git ls-tree -r HEAD --name-only | grep -E 'sibling_gate|^plans/00(09|10)'
```

## Landing order (one PR per step)

Each PR is built the same way:
1. Build the branch on the previous PR's branch (PR A on `origin/main`).
2. Merge the source branch(es) in and resolve conflicts.
3. Pass the gates (§Gates).
4. Get a fresh-eye review, then merge.

**PRs A–C land with merge commits (`gh pr merge --merge`), never squash.** Squashing A would
leave `f969f83`'s history off `main`. Git would then treat B's and C's copies of the taxonomy-v2
history as competing edits, which means ~20 conflicts that don't need to exist (claim-map code
and tests, `pyproject.toml`, `uv.lock`). Only PR D is squash-merged. Because each branch is built
on the previous one, its diff against `main` shrinks to its own content once the earlier PR
lands.

### PR A: claim-map judge

Merge `f969f83` into `main`. There are no conflicts. This plan file rides along in PR A. Close #4
with a comment pointing at PR A that explains the #11/#13 work stays on `camila/taxonomy-v2`.

### PR B: sibling probes + user-modelling ladders

Merge `camila/sibling-probes`, then `camila/um-ladder-v2`.
- Both fork from `9eaeecb` (plan 0005 rev 2). `sibling-probes` then implemented plan 0005
  (`ladder.py`, `ladder_probe.py`, `tests/test_sibling_ladder.py`), and `um-ladder-v2`
  implemented plans 0006/0007 (`um_*.py`). Neither contains the other, so both must land.
- Known conflicts: `README.md` when merging `sibling-probes`; then `CLAUDE.md`,
  `data/sibling_probe/README.md` and `br_lexicons.json` when merging `um-ladder-v2`.
  `README.md`, `CLAUDE.md` and `data/sibling_probe/README.md` keep both sides.

`data/sibling_probe/br_lexicons.json` (add/add):
- **Take the `sibling-probes` version for plan 0005.** It is plan 0005's review fix (`527378c`),
  which adds the bare-name aliases Jordan, Georgia, Guinea, Luxembourg and Niger.
- **But the country ladder (plan 0006) also reads this file**, through `um_ladder.load_lexicon`
  via `ladder_lex.LEXICON_PATH`. The new aliases change its matcher: "Guinea is lovely" goes from
  `{}` to `{'guinea'}`. `test_um_ladder.py` does not catch this.
- Results 0006 must not change silently, so:
  - The `um-ladder-v2` version is kept as `data/sibling_probe/br_lexicons_um_v1.json`.
  - `um_ladder.load_lexicon` reads that file for the country domain
    (`UM_COUNTRY_BASE_LEXICON_PATH`).
  - A new test, `tests/test_um_lexicon_pin.py`, pins the file's sha256 and the Guinea behaviour.
  - The docs say that adopting the newer lexicon is a results change.

The `um-ladder-v2` tip commit (`451b1e3`) adds WIP adjacent-layer (L±1) code paths, and its run
was stopped. Keep the code, because it is opt-in (`um_run_adj.sh`). PR D's CLAUDE.md entry for it
says it is unvalidated and that no results depend on it.

Close #6 as superseded by PR B.

### PR C: story + agentic-misalignment evals

Merge `camila/am-halluc` (which contains `camila/story-eval`) on top of PR B's branch.
- The only conflicts are `cli.py`, `CLAUDE.md` and `README.md`. `cli.py` keeps B's
  `cmd_sibling` and C's `_eval` helper, both whole. `CLAUDE.md` and `README.md` keep both sides.
- Both branches' tests pass unmodified.

Close #7 and #9 as superseded by PR C.

### PR D: production hardening

PR D starts only after A–C have landed, so it touches one tree.

#### D1. Plan numbering

Two plans are numbered 0007. The user-modelling 4-domain plan
(`plans/0007-user-modelling-ladder-4-domains.md`) becomes **0011**. 0009 and 0010 are skipped
because they belong to the unlanded resample/gate branches.

Update every reference to it: run `git grep -n "0007"` across plans, docs, CLAUDE.md, code
comments and results docs, and change only the hits that mean the UM 4-domain plan.

#### D2. CLI registry

`cli.py` (~600 lines after C) is a merge-conflict hotspot, because every eval adds its subcommands
and branches to it.
- Move each eval's CLI code into `evals/<eval>/cli.py`, which exposes `register(sub)`.
- The top-level `cli.py` keeps the shared helpers (`_common`, `_models`, `_layers`, …) and a
  list of eval modules to register.

**Retained invariants:**
- Every existing command line keeps working byte-for-byte: the same subcommand names, flags,
  defaults, exit codes (0 / 3 = `JudgeConfigError` / 4 = failed calls remain) and output paths.
- `hbench run --eval agentic_misalignment` and `hbench report --eval agentic_misalignment` stay
  `--eval` switches on `run`/`report`; they do not become new subcommands.
- The `--eval` choices list (`evals.EVALS`) is unchanged.
- The module-level names the tests import stay importable from `hallucination_bench.cli`:
  `cli.main`, `cli.build_parser`, `cli._models`, and anything else
  `git grep -n "cli\." tests/` finds.

This is a pure move: no behaviour change and no test edits.

#### D3. Paths

Add `src/hallucination_bench/paths.py`. It resolves each external location from an environment
variable and falls back to today's path:
- `HBENCH_READOUTS_ROOT` (default `/workspace/wsbench-paper-data/readouts`)
- `HBENCH_WSBENCH_PUB` (default `/workspace/root/wsbench-pub`)
- one more variable for any other `/workspace/...` root found by
  `git grep -n "/workspace/" -- src scripts tests`

Rules, by kind of hit:
- **This repo's own worktree** used as "the repo" (`/workspace/hb-am-halluc`,
  `/workspace/hb-sibling-probes`, `/workspace/hb-um-ladder`, ...) becomes the repo root.
  - A path to a **tracked** file uses the repo-relative path (`hallucination_bench.DATA` /
    `REPO_ROOT`). Example: `scripts/sibling_probe/um_lexicons.py` reads
    `/workspace/hb-sibling-probes/data/sibling_probe/concepts.json`.
  - Shell scripts `cd` to `"$(cd "$(dirname "$0")/<..>" && pwd)"`.
- **Another checkout's gitignored outputs, and other data roots**, become `HBENCH_*` env vars
  with today's path as the default. They are NOT replaced by the repo root. Examples:
  - `um_ladder.py`'s `SRC_DEFAULT=/workspace/hb-sibling-probes`, which points at that
    worktree's `outputs/`
  - `run_am_mlnla.sh`'s `GROOT=/workspace/hb-am-gen` and `RROOT=/workspace/hb-readouts-am`
  - `/workspace/hb-readouts-story`
- Where each kind of file gets the value:
  - `src/` hits go through `paths.py`.
  - `scripts/*.py` import from `paths.py`.
  - Shell scripts use `${HBENCH_X:-<today's path>}`.

Defaults stay identical, so nothing changes on the dev box.

#### D4. Honest skips

A test that needs local data skips with a reason naming the missing path and the environment
variable. CI runs pytest with `-rs`, so the skip list is visible.

#### D5. CI

Add `.github/workflows/ci.yml`. It runs on push and on PRs, on ubuntu-latest with Python 3.12:

```
uv sync --locked
uv run ruff check
uv run ruff format --check
uv run --with scikit-learn pytest -rs
```

- scikit-learn stays a non-dependency, following the repo's own convention. `--with` makes CI
  run the probe tests instead of skipping them.
- The workflow sets `HF_HUB_OFFLINE=1`, so `test_sibling_probe.py`'s Qwen tokenizer tests skip
  with their stated reason instead of downloading.
- No API keys.

#### D6. Docs

- The README gets an "Evals" table: eval, plan, entrypoint command, what it measures, results
  doc.
- The CLAUDE.md layout covers every landed module, including the unvalidated L±1 code paths
  (see PR B).
- `docs/results/README.md` indexes the results docs.

## Gates (every PR)

```
uv run ruff check && uv run ruff format --check && uv run pytest -rs
```

The ruff config must exclude only what the source branches already excluded. Also:

- **Tests are not edited to pass a merge or the refactor.** `git diff origin/main...HEAD -- tests/`
  is reviewed separately. Each changed assertion in a test that already exists on `main` or on a
  source branch must be justified by this plan. No PR justifies any; PR B only *adds*
  `tests/test_um_lexicon_pin.py`.
- The sibling-gate check under §Scope prints nothing.
- PR D only:
  - `hbench --help` and each subcommand's `--help` are diffed before and after the refactor and
    are identical.
  - A smoke run that needs no API calls (`hbench run --rebuild` on one arm with existing outputs,
    or the equivalent offline path) gives byte-identical outputs before and after.

## Not in scope

- Moving data out of git. Without the sibling-gate outputs (~76 MB), the largest in-scope branch
  carries ~17 MB of inputs (probe texts, banks). That is acceptable to keep in the repo.
- Re-running any eval or changing any result.
- Rewriting history or deleting any branch.
