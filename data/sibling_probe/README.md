# data/sibling_probe — the user-modelling sibling ladder (plans 0006 + 0011)

The final sibling eval is the 3-domain user-modelling ladder (country, capital_city = the city the
user is visiting, holiday): plans `plans/0006-user-modelling-country-ladder.md` +
`plans/0011-user-modelling-ladder-4-domains.md`. Code: `src/hallucination_bench/evals/sibling_probe/`
and `scripts/sibling_probe/um_*`. Plan 0013 removed the superseded evals (plan 0003 claim-map
sibling bank, plan 0004 verdicts, plan 0005 basic-readout ladder, the occupation / emotion UM
domains) from the tree; their data and builders are in git history. This directory holds only
what the ladder (and its tests) read.

| file | what | written by |
|---|---|---|
| `pilot/` | plan 0003 Phase A pilot (40 prompts, gold meta, rough counter); read by the `bank` / `concepts` tests | 2026-10-11 pilot |
| `concepts.json` | plan 0003 concept inventory (7 categories, 11–16 concepts each); frozen input, seeds `um_lexicons.json` | plan 0003 (builder removed in plan 0013) |
| `items.json` | plan 0003 eval bank, 204 items with Qwen responses; frozen input (`um_ladder.py probe-texts` leak / n-gram dedupe) | plan 0003 `gen_items.py` + Qwen job (job removed in plan 0013) |
| `probe_texts.jsonl` | plan 0003 probe-training + neutral texts (5,064); frozen input, reused by `probe-texts` | plan 0003 `gen_items.py probes` |
| `br_lexicons.json` | plan 0005 lexicon (`ladder_lex.LEXICON_PATH`); read by the lexicon-pin and `ladder_lex` tests | plan 0005 |
| `br_lexicons_um_v1.json`, `umcommit_*`, `um_country_*` | the country ladder (plan 0006, below) | `um_ladder.py` / labellers |
| `um_lexicons.json`, `umcommit_<domain>_*`, `um_<domain>_*` | the capital_city / holiday domains (plan 0011, below) | `um_lexicons.py` / `um_ladder.py --category` / labellers |

The `concepts.json` / `items.json` / `probe_texts.jsonl` sections below describe how those frozen
plan 0003 inputs were built; the scripts named there were removed in plan 0013.

## concepts.json

`{version, provenance, categories: {cat: {question, requests, merges, concepts: [{id, label, gold,
aliases, exclude, token_aliases, token_eligible}]}}}`.

- **Concept choice.** Every gold of the wsbench verbatim / pilot items (`gold: true`) plus every
  sibling seen in >= 2 cells of the category's readouts, ranked by cell count, cap 16.
  Counts: a plan 0003 counting script (broad candidate vocabulary, package matcher) over the pilot s3d-rl600 readouts (new categories) and the wsbench `user_modeling`
  s3d-rl600 + jlens (L28/36/44/52) + nla-rl-L42 readouts of the selfdescribe Country /
  Occupation items -> `outputs/sibling/sibling_counts.json`. After generation every concept is
  the gold of >= 1 item, so `gold` only marks the original (pilot / wsbench) golds.
- **Left out as generic non-instances** (plan: pets, generic uses are not concepts): emotion
  `love` (29 cells, "I'd love to help"), `calm` (22, a tone word: "calm but firm"), `empathy`
  (the assistant's stance); farm_animal `dog`, `cat` (pets); tree `evergreen` (a class, not a
  species) and `orange` (alias is the colour word); holiday `New Year's Eve` merged into
  `new_year`. Over the cap: country japan/nigeria/germany…, occupation thief/actor/teacher….
- **Category notes.** `capital_city` keeps its name but its siblings include non-capital cities
  (New York, Boston, Kyoto, Osaka, Chicago, Luxor, Florence, Prague are the readouts' frequent
  alternatives; the implication question is "which city is the user visiting?"). farm_animal
  and holiday have 11 concepts (only 2 / 3 frequent siblings); farm_animal adds turkey and
  donkey to reach 10+.
- **Aliases.** English: case-folded, curly apostrophes folded, ASCII-alphanumeric word
  boundaries, plurals / demonyms listed explicitly. CJK: >= 2 characters, substring, every CJK
  alias has an exclusion list (often empty). Exclusion phrases void a hit inside them
  (`happy to help`, `North Korea`, `东京都` for 京都, `罗马尼亚` for 罗马, `nail polish`,
  `George Washington`). A hit strictly inside a longer hit of another concept is dropped
  (`new year` inside `Lunar New Year`). Tests: every alias and label matches exactly its own
  concept.
- **Merges** (recorded per category): emotion clusters (embarrassed/ashamed/humiliated/
  mortified; angry/furious/frustrated/irritated/annoyed/outraged; sad/grieving/heartbroken;
  nervous/anxious/worried; …), occupation writer/author/novelist, software engineer/developer/
  programmer, hacker/penetration tester, artist/painter, entrepreneur/founder; farm_animal
  bovines (cow/cattle/bull/ox/calf/steer), horse/pony; tree palm species; holiday New Year's Eve.
  painter, founder, penetration tester, steer, coconut tree / date palm and flowering cherry were
  added after the first implication round showed them as Qwen's answers for those golds.
- **Token aliases** (J-lens): `" " + alias` for English aliases (the bare form only for
  capitalised names of 4+ characters, since bare lowercase pieces are subword fragments: `hen`
  in `Stephen`), CJK aliases as-is, plus a few token-only extras (` palm`, ` cherry`, ` US`);
  kept iff ONE Qwen3.6-27B token. Matching is exact after strip + lower-case (no first-subtoken
  matching). Not token-eligible: costa_rica, flight_attendant, anthropologist, donkey, luxor,
  birch, spruce, aspen, sycamore, independence_day, diwali, passover.
- **Questions / requests**: the plan's strings; "…" in the plan's elided questions is read as
  "Answer in one or two words." (emotion: "one word").

## items.json

`{version, provenance, counts, items: [{id, category, gold, source, messages, cue, request,
request_idx, named, implication, response}]}`. Messages are always `[system "You are a helpful
assistant.", user cue + " " + request]`.

- **wsbench** (country 16, occupation 14): `/workspace/root/wsbench-pub/evals/user_modeling/
  items.json` @92d763e selfdescribe Country / Occupation, verbatim (id = wsbench `name`, gold =
  `attr`). Not leak- or implication-checked; `named` lists inventory concepts named in the turn.
- **pilot** (8 per new category): `pilot/specs.json`, request sentence 0.
- **generated**: cues from `claude-opus-5-5` (`sibgen-v1` prompt in `gen_items.py`; the hacker
  occupation from `claude-sonnet-5-5`, because Opus 5.5 refused all 4 attempts). Allocation =
  coverage rule (`bank.allocate`: total max(20, 2 x concepts) capped at 32, fewest-items-first,
  <= 3 generated per gold), pool = allocation + 3 spare per concept, topped up after failures.
  Checks: format (ends in `.!?`, no trailing quote), leak (no alias of any category concept in
  the user turn), duplicates, implication gate, window identity. Each candidate is checked with
  both request sentences; the request is then chosen to balance the category (16/16 or 11/11).
- **Implication gate**: Qwen3.6-27B, greedy, thinking off, 16 new tokens, on a fresh chat
  `[system, user: <item user turn> + "\n\n" + <category question>]`; pass iff the answer's
  alias matches include the gold.
- **response**: Qwen3.6-27B greedy reply (chat template, `enable_thinking=False`, 256 new tokens;
  135/204 stop at the 256-token limit).

| category | concepts | items | wsbench / pilot | generated | candidates dropped: leak / implication |
|---|---|---|---|---|---|
| country | 16 | 32 | 16 | 16 | 0 / 8 |
| occupation | 16 | 32 | 14 | 18 | 2 / 38 |
| emotion | 16 | 32 | 8 | 24 | 3 / 76 |
| farm_animal | 11 | 22 | 8 | 14 | 2 / 8 (+1 format) |
| capital_city | 16 | 32 | 8 | 24 | 3 / 1 |
| tree | 16 | 32 | 8 | 24 | 2 / 46 |
| holiday | 11 | 22 | 8 | 14 | 0 / 0 |

Every concept is the gold of >= 1 item (country philippines/australia/belarus/poland and
occupation writer…scientist have 1, because the 32 cap is reached; all others 2+).

## probe_texts.jsonl

`{id, category, concept (null = neutral), kind: explicit|implied|neutral, request_idx, context,
messages}`; 24 explicit + 24 implied per concept and 24 neutral per category, endings balanced
(12/12 for the two-sentence categories). Generated by `claude-opus-5-5` (hacker: Sonnet 5.5),
never shown to the judge. Checks: explicit texts name exactly their concept, implied / neutral
name no concept of the category; no 8-gram (word) shared with any eval cue (all items and all
candidates); window identity; no duplicates.

## Plan 0005 lexicon

- `br_lexicons.json` (`brlex-v1`): per category (`sport` 84, `country` 200 = 193 UN members +
  Taiwan, Palestine, Kosovo, Scotland, Wales, Northern Ireland + Vatican City, `musical_instrument`
  61) `{label, en, cjk (>= 2 chars), tokens}`; `block` (capital cities, "Mexico City", "Indian
  Ocean", 印度洋, ...) match but name no concept; `ambiguous` ("football" -> the item's gold or
  `ladder_lex.FOOTBALL_DEFAULT`); `capitals`; `checks` (code: duplicates removed, short CJK,
  substring pairs resolved by longest match). Generated once by `claude-opus-5-5` (effort low),
  then code-forced equivalences / drops (plan 0005's ladder script, removed in plan 0013).

## Plan 0006 (user-modelling ladder v2, country)

- `br_lexicons_um_v1.json`: the country ladder's base lexicon, a frozen copy of plan 0005's
  `br_lexicons.json` as of 2026-10-12, before plan 0005's review fix added the bare-name aliases
  Jordan / Georgia / Guinea / Luxembourg / Niger. Results 0006 were produced with it, and
  `tests/test_um_lexicon_pin.py` pins its sha256. `ladder_lex.py` (plan 0005's matcher) is
  kept unchanged. The Set A patch is code (`um_ladder.SET_A_LEXICON_PATCH`).
- `umcommit_dev.jsonl` (40, Fable, before any judge call), `umcommit_dev2.jsonl` (60, Fable,
  blind to the judge, used for the one prompt revision v1 -> v1.1), `umcommit_gold.jsonl` (80,
  fresh subagent from the plan's Step 3 text only, never used for tuning); the `*_todo.jsonl`
  files hold the units as shown to the labellers (dev2's holds uids only).
- `um_country_concepts.json` (`umconcepts-v1`): set C = 16 golds + siblings with >= 3 committed
  units + <= 2 most-mentioned same-region countries per gold; cap 80, pool trimmed first.
- `um_country_probe_texts.jsonl`: 24 explicit + 24 implied per concept + 24 neutral; plan 0003
  texts reused where they pass the Set A checks (gold map applied: uk / usa), the rest generated
  with the plan 0003 `sibgen-v1` prompt over C; drop counts by cause in
  `outputs/sibling/um/probe_texts_report.json`.
- `um_country_distractors.json` (`umdist-v1`): D_i per item, chosen from Phase A only.

## Plan 0011 (the capital_city / holiday domains)

- `um_lexicons.json` (`umlex-v1`): per domain `{concepts, block}` + `capitalised_only` + `checks`,
  built by `scripts/sibling_probe/um_lexicons.py` (seeded with `concepts.json`); `um_ladder`
  applies `um_domains` alias moves / drops / extra block phrases on load. Plan 0013 removed the
  occupation / emotion categories (no hash or fingerprint covers the whole file).
- `umcommit_<domain>_{dev,gold}.jsonl` + `*_todo.jsonl`: commitment-judge calibration labels.
- `um_<domain>_concepts.json`, `um_<domain>_probe_texts.jsonl`, `um_<domain>_distractors.json`:
  as for country (the distractors' sha256 is checked against the probe score run).
