# data/overrefusal (plan 0015)

Built by `hbench overrefusal candidates|judge-gt|bank`; never hand-edited.

| file | what | built by |
|---|---|---|
| `meta.json` | OR-Bench repo + pinned revision `e36d8b80…`, CSV sha256s, seed, candidate counts | `candidates` |
| `candidates.jsonl` | `id, config (hard/toxic), row (0-based CSV row), category, prompt` — a seeded prefix of a seeded permutation of each config (prefix-stable: more candidates never change earlier ids) | `candidates` |
| `gt_labels.jsonl` | per candidate: `n_valid, n_refuse, n_full, n_partial, greedy_label, labels[10], reasons[10], heads[10]` (first 300 chars of each sampled reply), `greedy_reason/greedy_head` (hard only); judge `orgt-v1` (Opus 5.5) | `judge-gt` |
| `items.json` | `meta` (eligible / selected counts) + `items`: `id, population, config, category, prompt, about_refusal, n_refuse, prompt_render, sites[{pos, token, kind}]` | `bank` |
| `capture_rows.json` | one `render: "plain"` row per item for global-workspace `wsbench_capture_modal.py` (no `eval_*` key = every position) | `bank` |

Populations: `never_refused` (hard, 0/10 sampled refusals at T = 1 and a complying greedy
reply), `hard_refused` (hard, >= 8/10 and a non-complying greedy reply; relaxed to >= 5 below 25
prompts, flagged in `items.json` meta), `always_refused` (toxic, 10/10). refuse =
`partial_refusal` or `full_refusal`.

The full on-policy samples (`outputs/overrefusal/samples*.jsonl`, vLLM 0.25.1, n=10, T=1.0,
top_p=0.95, max_tokens=512, seed=0; greedy n=1, T=0) are not in git.

## probes/ (plan 0015 rev 5-6): refusal probe artifacts, reusable by other evals

- `refusal_probe_v1.npz`: keys `{variant}/L{layer}/{U, muU, sigma, null_mean, null_sd, tau,
  c_mean, c_sd, c_tau, validity}`; arrays `[fold=4, offset=4]` (offset 0 = `assistant`, 1 =
  `<|im_start|>`, 2 = `\n`, 3 = `<|im_end|>`), `U` `[4, 5120]`. Score a residual `h` (Qwen3.6-27B,
  layer L, a tail offset o of the render `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant`):
  per fold `s = (h.U[f] - muU[f,o]) / sigma[f,o]`, `z = (s - null_mean) / null_sd`, present if
  `s > tau[f,o]`; average over folds. Variants `refusal-dm-v1` (primary, validated) and
  `refusal-beh-v1` (underpowered, reported only).
- `refusal_probe_v1.json`: provenance (datasets + pinned sources, counts, validity, lams, acts
  subdirs, scoring rule, git commit).
- `refusal_probe_v1_calib.{npz,json}`: exploratory thresholds `tau_calib` / `c_tau_calib`
  re-calibrated on prompts Qwen answered (use these for OR-Bench-like prompts; the Alpaca `tau`
  fires on almost any harm-looking prompt).
- `train_texts.jsonl`, `train_meta.json`, `orbench80k_*`: training data. `calib_texts.jsonl` (+
  `calib_capture_rows.json`): the v2 threshold's calibration pool, written by
  `hbench overrefusal probe-data calib` (byte-identical regeneration; in `MANIFEST.sha256`).

## Rev 7 (grown populations, threshold v2)

- `grow/orbench80k2_candidates.jsonl` (450, 45 per OR-Bench category, `or-bench-80k` at the
  pinned revision, excluding by exact + near-duplicate match every prompt used before),
  `grow/orbench80k2_gt_labels.jsonl` (10 samples + greedy, `orgt-v2`), `grow/capture_rows_v2add.json`
  (the 213 added items). `items.json` now holds 360 items (never_refused 160, hard_refused 100,
  always_refused 100); the 147 rev-3 items are unchanged; every item has a `source`; the
  category / source mix is in `items.json` `meta.rev7`.
- **`probes/refusal_probe_v2.{npz,json}` (use this one):** the `refusal-dm-v1` fold arrays
  (`U, muU, sigma, ...` as in v1) plus `tau_v2, mu_v2, sd_v2` `[fold, offset]` and
  `c_tau_v2, c_mu_v2, c_sd_v2` `[fold]`, the primary threshold: q95 over the calibration half
  of on-distribution negatives (OR-Bench-80k prompts Qwen answered 0/10, not in the read set).
  Present at a tail offset o iff the mean over folds of `s > tau_v2[f,o]` is >= 0.5, with
  `s = (h.U[f] - muU[f,o]) / sigma[f,o]`; z = `(s - mu_v2) / sd_v2`, averaged. The JSON has the
  held-out-half realised FPR per (layer, offset), the split ids and provenance.
