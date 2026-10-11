# data/sibling_probe — plan 0003 (sibling-concept probes)

Spec: `plans/0003-sibling-concept-probes.md`. Code: `src/hallucination_bench/evals/sibling_probe/`
and `scripts/sibling_probe/`.

| file | what | written by |
|---|---|---|
| `pilot/` | Phase A pilot (40 prompts, gold meta, rough counter); readouts in `outputs/sibling/pilot/` | 2026-10-11 pilot |
| `concepts.json` | concept inventory (7 categories, 11–16 concepts each) | `scripts/sibling_probe/build_concepts.py` |
| `items.json` | eval bank, 204 items with Qwen responses | `scripts/sibling_probe/gen_items.py` + `qwen_gen_modal.py` |
| `probe_texts.jsonl` | 5,064 probe-training + neutral texts | `scripts/sibling_probe/gen_items.py probes` |

## concepts.json

`{version, provenance, categories: {cat: {question, requests, merges, concepts: [{id, label, gold,
aliases, exclude, token_aliases, token_eligible}]}}}`.

- **Concept choice.** Every gold of the wsbench verbatim / pilot items (`gold: true`) plus every
  sibling seen in >= 2 cells of the category's readouts, ranked by cell count, cap 16.
  Counts: `scripts/sibling_probe/count_siblings.py` (broad candidate vocabulary, package matcher)
  over the pilot s3d-rl600 readouts (new categories) and the wsbench `user_modeling`
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
