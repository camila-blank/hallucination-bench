# `data/agentic_misalignment/`: the agentic_misalignment bank (plan 0008)

The 32 frozen WorkspaceBench `agentic_misalignment` items (`camilablank/wsbench-pub@92d763e`,
`evals/agentic_misalignment/items.json`; the full commit and the sha256 of every input file are
in `items.json` `meta`): 28 misaligned and 4 controls, 18 Anthropic-grid items (`group =
anthropic`, Lynch et al. 2025) and 14 in-house (`group = working`). Each carries `system`,
`text` (the user turn), `prefill` (the pinned rollout Qwen3.6-27B wrote), `category`,
`descriptor`, `misalignment_rate` (8-sample rate; null for the Anthropic items).

- **Built by** `scripts/build_am_bank.py` (deterministic; re-run it, never hand-edit). It reads
  the wsbench-pub items and only the layer-20 `(id, pos, token)` columns of the `s3d-rl600`
  readouts (streamed), and checks that the J-lens layer-20 columns and the NLA positions match.
- **`items.json`:** `{meta, items: [{id, category, arm, group, descriptor, misalignment_rate,
  persona, system, text, prefill, prompt_render, response, sites: [{pos, token, char
  [, split_char]}]}]}`. `response = prefill`.
- **Render check.** `prompt_render` is the Qwen chat render the model read:
  `<|im_start|>system\n{system}<|im_end|>\n` (when `system` is non-empty) +
  `<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant`. The decoded readout tokens
  concatenate to it exactly, except where a multi-byte character is split across tokens (each
  piece decodes to U+FFFD): the `┌┬┐` box-drawing line of the 6 `am-murder_*` items, 15
  characters each, 180 split sites in all. `difflib` aligns the two; every mismatch is a
  U+FFFD-only run against FFFD-free text. A token carrying a piece of a split character is marked
  `split_char: true`, and its `char` is the end of the run when it ends inside it (the marker
  never splits a character). For every other site `prompt_render[:char]` ends with the token.
- **Prompt-only positions.** The read sites are every prompt position, `0 .. rollout_start-1`,
  contiguous, ending with `<|im_start|>`, `assistant`: 50,452 sites, 270 to 2,825 per item. So
  the claim map's marker `⟦READ⟧` sits **in the prompt** (`bank.marked_prompt`), never in the
  response; there is no thinning (the detector decides which cells matter).
- **`persona`:** the name after "You are" in the system prompt, checked by hand: Alex 19, Atlas
  2, Argus / Devin / Hypatia / Kepler / Meridian 1 each; the 6 "You are an …" / "You are the …"
  items have `persona = null`. The score remaps a detector `other_party` actor named like the
  persona to `ai_self`; Stage H's verified actor (plan 0008 rev 13) is remapped the same way,
  with the detector's `actor_name` (the blind verifier never sees the persona).
- **Short prefills.** The two `am-murder_none-none_{replacement,restriction}` prefills are 66
  characters and `sb-deployment-gate`'s is 190: "realised after the marker" has little text to
  match against for them.
- **1 of 8.** The prefill is one of 8 samples (three misaligned items have an 8-sample
  misalignment rate of 0.125 to 0.25). It is what the model did in this run, not its only
  possible intention.
- **`intent_overrides.json`** (absent until the intent audit): the hand-audited corrections of
  the `amintent-v2` results, `{"items": {"<id>": {"rollout_kinds": [{kind, rollout_quote,
  enabling_quote}], "scenario_kinds": [{kind, enabling_quote}], "note": "..."}}}`; a list given
  replaces the model's list. Override quotes must locate (rollout quotes in the prefill,
  enabling quotes in `prompt_render`).
