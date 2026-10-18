# `data/story/`: the story bank (plan 0007)

20 "tell me a story" items: stories, a poem, a letter, a dialogue scene, a place description. They
come from WorkspaceBench's frozen `hallucination_creative` bank
(`camilablank/global-workspace:evals/workspace-bench/hillclimbing_evals/hallucination/creative/`;
the commit and sha256s are in `items.json` `meta`). Qwen3.6-27B answered each item on-policy.

- **Built by** `scripts/build_story_bank.py`. Never hand-edit these files: re-run the builder.
- **Thinning:** 8 read sites per item, using the old judge's rule (`step = len(sites) / 8`,
  `sites[int(i * step)]`). That gives 160 cells per layer, the same cells the old Sonnet v5b
  creative run judged.
- **`items.json`:** the `wsbench_chat` item schema (`id, stratum, source, prompt, response,
  response_start, sites[{pos, token, kind, char}]`). `char` is the offset just after the site
  token, where the judge's `⟦READ⟧` marker goes.
- **`capture_rows.json`:** the source capture rows (exact token ids), with `eval_positions` set to
  the 8 kept sites. This is the capture record for `wsbench_capture_modal.py` (plan 0007
  §Readouts). The judge never reads it.
- **Truncation:** responses were capped at ~512 tokens, so most stories stop mid-plot. The final
  `<|im_end|>` site follows that cut and is never kept. The text after the marker is the
  reference, and the claim map's reasonable-continuation ruling covers readouts that extrapolate
  past it.
