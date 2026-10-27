# `data/strongreject/`: the strongreject eval (plan 0016)

## `source/` (verbatim copies; never edit)

| file | source | sha256 |
|---|---|---|
| `strongreject_dataset.csv` | `alexandrasouly/strongreject@f7cad6c17e624e21d8df2278e918ae1dddb4cb56`, `strongreject_dataset/` (313 forbidden prompts, 6 categories) | `4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381` |
| `wrapping_jailbreaks.json` | `dsbowen/strong_reject@7a551d5b440ec7b75d4f6f5bb7c1719965b76b47`, `strong_reject/jailbreak_files/` | `805a29aec6b2d7129ce044663da73fcca219786cba863514ebc533180d580746` |
| `judge_templates.json` | same commit, `strong_reject/eval_files/` (the `strongreject_rubric` grader) | `6d01841899e99f1f57c4f19153056051684ede2a7e1b54ebbb37b5ddff981768` |
| `LICENSE.strongreject-dataset`, `LICENSE.strong_reject` | the two repos' MIT licenses | |

`families.wrap` re-implements `strong_reject.jailbreaks.wrapping_jailbreak` (base64 families
base64-encode the forbidden prompt, then `template.replace("{forbidden_prompt}", prompt)`).

## `prompts.json`

Built by `scripts/strongreject/build_prompts.py` (deterministic): 40 forbidden prompts
stratified by category (7/7/7/7/6/6 over the sorted categories, `random.Random(0)` within each)
x 7 families (`none` control, `aim`, `dev_mode_v2`, `evil_confidant`, `prefix_injection`,
`refusal_suppression`, `base64_input_only`) = 280 prompts, ids `sr-<forbidden_id:03d>-<family>`
(`forbidden_id` = 0-based CSV row).

## `items.json` + `capture_rows.json`

Built by `scripts/strongreject/build_bank.py` from `prompts.json` and the graded outcomes
(`outputs/strongreject/<judge>/outcomes.json`; its sha256 is in `meta`). The read set is every
`successful` prompt (cap 60) plus up to 200 − n_successful (cap 150) `unsuccessful` prompts,
round-robin over families. Each item: the prompt fields, `outcome` + `outcome_counts` (no
rollout text: rollouts stay in `outputs/`, uncommitted), `prompt_render` (the Qwen chat render
up to and including `<|im_start|>assistant`), `n_positions`, and the thinned `sites: [{pos,
token, char, region[, split_char]}]` (last 64 positions + every 16th before; `region` turn /
tail / body). `split_char` marks a token holding part of a multi-byte character (the
`evil_confidant` 😈), whose decoded piece is U+FFFD; its `char` is the offset-mapping end.
`capture_rows.json` is the wsbench capture schema with `eval_positions` = the site positions.
