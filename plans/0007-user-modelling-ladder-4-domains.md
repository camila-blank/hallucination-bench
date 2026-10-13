# Plan 0007: the plan 0006 sibling ladder on 4 more user-modelling domains

Branch `camila/um-ladder-v2` (stacked on plan 0006). Author: Camila Blank (planned with Claude),
2026-10-13. Revision 2 (after critique 1; the section below overrides the body).

## Revision 2 (critique 1)

1. **Matched null outside country:** pools (i) (the same item's other positions) and (ii) (other
   items with the same gold) are unchanged. Pool (iii) becomes **every other item**: there are no
   regions. The exclusions are retained: items whose gold is S or a distractor of the unit, and
   items not readable at L. The hard unmentioned null uses the top-quartile cosine rule only.
2. **Calibration = plan 0006's protocol per domain:** 40 dev units, one allowed prompt revision,
   80 blind gold units, target ≥ 85%, otherwise "judge (below target)".
   **Deviation:** the dev labels come from a fresh subagent (not Fable), blind to the judge. The
   gold labeller is a different fresh subagent working from this plan's text only.
3. **Concepts that are true together with the gold** become a `compatible` stratum, like
   country's `soviet`, never units (`um_domains.COMPATIBLE`).
   - occupation: researcher / scientist / professor / the specific sciences, anthropologist ↔
     researcher, artist ↔ illustrator / sculptor, musician ↔ singer / composer, writer ↔ poet /
     journalist, fisherman ↔ sailor, hacker ↔ software engineer, farmer ↔ rancher / shepherd,
     entrepreneur ↔ executive / manager;
   - city: Kyoto ↔ Osaka / Nara;
   - holiday: Lunar New Year ↔ New Year, Easter ↔ Good Friday, Christmas ↔ Boxing Day,
     Thanksgiving ↔ Black Friday.

   These are also never distractors for that gold.
   - **Emotion:** co-occurring emotions ("hurt and angry") are **not** made compatible; this is
     flagged as a definitional choice.
4. **Lexicon fixes:** 除夕夜 (Chinese New Year's Eve) moves to `lunar_new_year`. Holiday greetings
   ("Merry Christmas and Happy New Year", "Happy Holidays") are blocked.
5. **Common-word traps:** extra block phrases (`um_domains.EXTRA_BLOCK`):
   - city: "Florence Nightingale", "plaster of Paris", "Paris Agreement", "Lincoln Memorial",
     "Washington Post", …
   - emotion: "hope this finds you", "sorry to hear", "proud of you", "surprise party",
     "hurt my", "table of contents", "I'd love", …
   - occupation: "as an AI", "language model", "the model", "the judge", …

   **Capitalised-only** applies to every city and holiday alias ("Easter" counts, "easter" does
   not), plus country's existing rule. The judge's stock-phrase rule covers what remains.
6. **Per-domain plumbing:**
   - run `um-<domain>-v1`;
   - acts `acts-um-<domain>-probe`;
   - files `data/sibling_probe/um_<domain>_{concepts.json,probe_texts.jsonl,distractors.json}`
     and `umcommit_<domain>_{dev,gold}*.jsonl`;
   - outputs `outputs/sibling/um/<domain>/`;
   - fold salt `<domain>-um` (country keeps `country-um`).

   The Modal scorer gets `item_endings` (each item's request sentence). `um_run_probes.sh` takes
   `RUN` / `INP` / `PDIR` / `DIST`.
7. **Probe texts** are balanced per (concept, kind, request sentence) (12 / 12 where there are
   two), with shortfalls reported per ending.
8. **J-lens summaries** are filled in per domain (`summarize`); the cost is a few dollars.

Camila (2026-10-13): "extend this exact methodology to 4 other domains and produce the same
plot" (`plots/sibling_probe/um_country_simple.png`).

## Scope

- **Domains:** `occupation`, `emotion`, `capital_city` (the city the user is visiting; its
  siblings include non-capitals) and `holiday`. These are the Set A categories whose plan 0003
  probes passed validity. `farm_animal` and `tree` are left out: their probes were "too weak to
  adjudicate" in plan 0003.
- **Items:** the 32 / 32 / 32 / 22 Set A items of each domain (`items.json`). The readouts, the
  J-lens summaries and the eval captures (`acts-sibling-eval-all11` for L20…60 and
  `acts-sibling-eval` for L42) already cover every Set A item.
- **Methods:** as in plan 0006 (s3d O-lens, ML-NLA RL600, J-lens, NLA-RL L42).

## What is unchanged from plan 0006 (rev 8 + the simplified plot)

The ladder, applied per domain:
- mentions → commitment judge → probe → cross-method → label;
- probe type (a), whitened DiffMean, is primary, with validity ≥ 0.90 on implied texts;
- z on each probe's own per-fold null;
- 8 distractors per item (4 nearest by grand-mean-centred implied centroid at L44 + 4 seeded
  random, never a concept committed for the item), chosen from Phase A only;
- the n/a rules (no probe, invalid probe, fewer than 6 valid distractors, item unreadable);
- the cross-method ± 4 rule;
- the leakage guards (Phase A / distractors / Phase B, sha256 check).

The plot: `plots/sibling_probe/um_<domain>_simple.png`, the same as
`um_country_simple.png`:
- committed mentions the probe can test at their own layer only (probe-n/a excluded);
- no nearby tier: faithful = the probe fires at the mention's own layer, else cross-method,
  else hallucination;
- item-bootstrap CI on the hallucination share.

A combined figure `um_5domains_simple.png` puts the five domains side by side.

## What changes per domain

1. **Lexicon** (`data/sibling_probe/um_lexicons.json`, prompt `umlex-v1`, `claude-opus-5-5`,
   cached). It is a comprehensive list of the domain's instances, seeded with the Set A
   inventory: the same concept ids, its aliases, its merges as gold equivalents, and its
   exclusion phrases as block phrases.

   | domain | target size | notes |
   |---|---|---|
   | occupation | ~150 jobs | |
   | emotion | ~50 clusters | synonym clusters are one concept (plan 0003 merges) |
   | capital_city | ~150 world cities | |
   | holiday | ~60 holidays | |

   **Code checks:** no alias shared by two concepts, CJK aliases ≥ 2 characters, every inventory
   alias kept. The plan 0003 exclusions are kept (`happy to help`, `holiday` for Holi, `George
   Washington`, `New Year` inside `Lunar New Year`, …). Word-like aliases that are also common
   words count only when capitalised for the city and holiday domains (e.g. "Nice", "Reading",
   "Easter" stays).
2. **Judge** (`umcommit-v1.1-<domain>`): the plan 0006 v1.1 prompt with the domain's question
   and its own identity definition. The structure, forms, the biography-subject rule and fail
   closed (including the J-lens bag-only rule) are unchanged.

   | domain | committed when the sample says … |
   |---|---|
   | occupation | the user's (or prompt author's) job or profession is X |
   | emotion | the user feels X now, about the described event |
   | capital_city | the user is visiting / travelling to X |
   | holiday | the user is preparing for / celebrating X |

   Not committed: options, comparisons, other subjects (a character, a customer, a colleague's
   job), language, continuation, negation, question, meta. Emotion adds one rule: a feeling
   attributed to someone else, or a stock phrase ("happy to help"), is not committed.

   **Calibration:** a fresh subagent labels a blind gold set of 40 units per domain from this
   section. No dev revision; the prompt is the validated v1.1 structure. Report accuracy and
   precision / recall per domain.
3. **Concept set C:** golds + siblings with ≥ 3 committed units + a pool. There are no regions
   outside country, so the pool is the most-mentioned concepts not otherwise in C, up to 16.
   Cap 80, pool trimmed first.
4. **Probe texts:** 24 explicit + 24 implied per concept + 24 neutral, balanced over the domain's
   request sentences (12 / 12 where there are two). Plan 0003 texts are reused where they pass
   the checks; the rest are generated with the plan 0003 `sibgen-v1` prompt over C.
5. **Two request sentences:** three domains have two endings, so probe windows differ by
   ending. Two fixes apply to every domain; country, with one ending, is unaffected (a test
   checks this):
   - the null z and the type (b) centring are keyed by **(ending, offset)**, not offset alone;
   - eval cells are scored with their item's ending (`request_idx`). The country job passed
     ending 0 everywhere.

## Run

```bash
D=occupation  # emotion | capital_city | holiday
S=scripts/sibling_probe/um_ladder.py
uv run --no-sync python $S --category $D lexicon
uv run --no-sync python $S --category $D mentions && ... summarize && ... judge
uv run --no-sync python $S --category $D todo-gold   # 40 units -> blind subagent -> calib
uv run --no-sync python $S --category $D concepts && ... probe-texts && ... capture-rows
#   render (wsbench_public_plan.py prompts) + capture -> acts-um-<domain>-probe (1 GPU each)
uv run --no-sync python $S --category $D probe-inputs
scripts/sibling_probe/um_run_probes.sh pack ; ... distractors ; ... score   (per domain)
uv run --no-sync python $S --category $D label
uv run --no-sync python scripts/sibling_probe/um_plot_simple.py --category $D
```

Outputs per domain go to `outputs/sibling/um/<domain>/`; country stays at
`outputs/sibling/um/`.

## Deliverables

- `plots/sibling_probe/um_<domain>_simple.png` for the 4 domains, plus
  `um_5domains_simple.png`.
- `docs/results/0007_um_4domains.md`:
  - per domain: mention / commit counts, the plot's table with CIs, probe quality, judge
    calibration, the n/a share;
  - the plan 0006 gate per domain, reported (it does not block anything here).
- Tests: per-domain lexicon checks, the (ending, offset) null on a two-ending synthetic set,
  and country invariance.

## Cost (estimate)

| item | cost |
|---|---|
| judge, ~3–6k calls per domain | ~$15 each |
| lexicons | ~$5 |
| probe texts | ~$6 each |
| 4 captures (1 H200, ~20 min each) | |
| fits (CPU) | |
| **total** | **~$120–160** |
