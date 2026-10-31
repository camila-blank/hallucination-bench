# 0024 — the final O-lens arm (final-olens-rl600-x4) on the hallucination index: results

Plan: [`plans/0024-final-olens-arm.md`](../../plans/0024-final-olens-arm.md) (code: #49; benign_roleplay
pin: #48, #50). The arm is the WorkspaceBench fix campaign's lens: HF `agu18dec/final_olens`, folder
`lens_rl_step600`, 4 samples per activation (`bullet4`), read by the campaign producer
(global-workspace PR #646). WorkspaceBench vendors this index (workspace-bench #192, plan 0057).

**Index:** `hbench index` (hindex-v6, the CLI default) at `b1043b4` (#50), judge `claude-opus-5-5` (effort
low, keyless `ant` token), from the official source roots plus the arm's roots
(`main`, `agentic_misalignment`, `story`: `$W/out/<eval>/claude-opus-5-5`; sibling: the
`export_um_arm.py` root; overrefusal: the arm's own `offtopic_index/`), with
`$W = /workspace/root/wsb-bp/hb` on the dev box. Every pre-existing hindex-v6 row (331) is reproduced bit
for bit; the build adds 11 `final-olens-rl600-x4` rows, all with `full`.

## Headline (lower is better)

`full`, final-olens-rl600-x4 vs s3d-rl600: L20 19.2 / 16.7, L24 20.4 / 16.0, L28 19.0 / 16.3, L32 17.1 /
13.8, L36 14.3 / 13.3, L40 16.0 / 12.5, **L44 16.6 [15.1, 18.2] / 13.2 [12.0, 14.4]**, L48 16.9 / 14.1,
L52 14.7 / 13.3, L56 12.9 / 12.9, L60 13.3 / 13.6. Unweighted 11-layer mean: **16.4** vs 14.1.

- **main** is higher than s3d-rl600 at every layer except L52 / L56 (0.1 lower there; L44 9.2 vs 7.3), on about half as many
  claims per layer (the claim map reads the first `HAL_K = 3` of the arm's 4 samples; s3d-rl600's 3
  samples each carry 4 bullets).
- **agentic_misalignment** is higher at every layer (by 2.7-6.7 points; L44 27.0 vs 20.8) and is the largest
  weighted contributor to the `full` gap (1.0 of the 2.3-point 11-layer mean gap); see the refusal caveat below.
- **overrefusal** (wrong-topic decline lines) is higher at L20-L48 (L28 42.0 vs 37.2, L44 25.2 vs
  18.4), close from L52 on.
- **sibling** is within the CIs of s3d-rl600 at most layers; **story** is higher at 10 of 11 layers, 5-8
  points at L20-L32 (CIs overlap at every layer).

## Per-layer table (% hallucinated, 95% CI, n)

`n` = claims (main / story), candidate claims (AM), UM units (sibling), decline lines (overrefusal).

| method | layer | full | matched | main | agentic_misalignment | sibling | story | overrefusal |
|---|---:|---|---|---|---|---|---|---|
| final-olens-rl600-x4 | 20 | 19.2 [17.3, 21.2] | 16.3 [15.2, 17.5] | 10.9 [9.5, 12.5] n=9458 | 23.3 [21.3, 25.0] n=2576 | 39.6 [27.5, 54.5] n=557 | 26.3 [20.4, 32.0] n=1887 | 34.5 [31.0, 38.1] n=777 |
| final-olens-rl600-x4 | 24 | 20.4 [18.6, 22.2] | 17.9 [16.6, 19.1] | 11.7 [10.2, 13.4] n=9629 | 26.5 [24.2, 28.9] n=3070 | 38.0 [25.1, 48.6] n=427 | 25.8 [20.4, 30.6] n=1865 | 40.2 [37.0, 43.3] n=922 |
| final-olens-rl600-x4 | 28 | 19.0 [17.4, 20.6] | 17.9 [16.7, 19.0] | 11.9 [10.4, 13.6] n=9713 | 24.7 [23.1, 26.6] n=4293 | 27.2 [17.8, 38.0] n=392 | 27.0 [21.7, 31.6] n=1983 | 42.0 [39.0, 45.0] n=1344 |
| final-olens-rl600-x4 | 32 | 17.1 [15.8, 18.4] | 17.3 [16.2, 18.4] | 11.5 [10.1, 13.1] n=9711 | 25.0 [23.8, 26.2] n=5115 | 15.5 [9.1, 22.2] n=343 | 29.1 [23.0, 34.7] n=1950 | 34.7 [31.8, 37.8] n=1396 |
| final-olens-rl600-x4 | 36 | 14.3 [13.1, 15.5] | 14.8 [13.7, 15.9] | 9.2 [7.8, 10.8] n=10365 | 25.5 [23.8, 27.1] n=7372 | 11.0 [7.1, 19.1] n=345 | 26.9 [21.4, 32.1] n=2099 | 20.9 [18.1, 23.9] n=1175 |
| final-olens-rl600-x4 | 40 | 16.0 [14.4, 17.5] | 15.4 [14.2, 16.5] | 9.9 [8.5, 11.5] n=10066 | 25.9 [24.6, 27.4] n=6335 | 20.2 [11.3, 31.2] n=341 | 25.1 [19.7, 30.6] n=2051 | 23.2 [20.1, 26.3] n=1249 |
| final-olens-rl600-x4 | 44 | 16.6 [15.1, 18.2] | 15.2 [14.1, 16.2] | 9.2 [7.8, 10.7] n=10205 | 27.0 [25.5, 28.5] n=5464 | 26.7 [17.8, 37.5] n=375 | 24.0 [19.4, 28.5] n=2065 | 25.2 [22.5, 28.1] n=1256 |
| final-olens-rl600-x4 | 48 | 16.9 [15.6, 18.2] | 15.1 [14.0, 16.2] | 9.3 [7.9, 10.8] n=10249 | 25.2 [23.5, 27.0] n=4729 | 29.2 [22.2, 36.6] n=428 | 23.0 [18.3, 27.5] n=2128 | 29.0 [25.7, 32.5] n=1138 |
| final-olens-rl600-x4 | 52 | 14.7 [13.4, 16.1] | 13.2 [12.2, 14.1] | 7.5 [6.4, 8.8] n=10792 | 24.6 [22.9, 26.3] n=4472 | 25.8 [18.0, 35.1] n=329 | 22.5 [17.7, 27.2] n=2185 | 20.4 [17.7, 23.5] n=1008 |
| final-olens-rl600-x4 | 56 | 12.9 [11.6, 14.3] | 11.6 [10.7, 12.5] | 7.3 [6.2, 8.5] n=10660 | 20.6 [18.9, 22.4] n=3858 | 22.2 [13.0, 30.8] n=252 | 19.2 [15.5, 22.9] n=2211 | 16.2 [13.3, 19.3] n=798 |
| final-olens-rl600-x4 | 60 | 13.3 [10.9, 15.6] | 11.3 [10.4, 12.2] | 7.3 [6.2, 8.6] n=10910 | 19.4 [18.0, 20.7] n=3638 | 27.0 [11.8, 47.1] n=184 | 17.0 [12.7, 20.7] n=2131 | 17.5 [14.7, 20.6] n=771 |
| s3d-rl600 | 20 | 16.7 [14.9, 18.5] | 13.5 [12.5, 14.4] | 9.8 [8.7, 11.1] n=18035 | 16.9 [15.3, 18.7] n=2467 | 39.4 [29.9, 56.0] n=181 | 20.6 [17.1, 23.8] n=3226 | 28.8 [24.8, 33.1] n=607 |
| s3d-rl600 | 24 | 16.0 [14.7, 17.2] | 15.6 [14.5, 16.7] | 10.6 [9.3, 12.0] n=18321 | 23.8 [21.6, 26.3] n=2843 | 18.7 [13.5, 26.5] n=232 | 18.7 [15.5, 21.5] n=3272 | 33.2 [29.2, 37.7] n=630 |
| s3d-rl600 | 28 | 16.3 [14.8, 17.7] | 14.9 [13.9, 15.9] | 10.2 [8.9, 11.7] n=18726 | 19.3 [17.8, 21.0] n=3784 | 26.1 [17.6, 36.4] n=271 | 21.1 [17.4, 24.4] n=3264 | 37.2 [33.9, 40.9] n=1058 |
| s3d-rl600 | 32 | 13.8 [12.7, 14.8] | 13.3 [12.3, 14.2] | 9.3 [8.0, 10.7] n=18875 | 18.3 [16.9, 19.9] n=5272 | 17.1 [12.4, 21.6] n=229 | 21.6 [18.1, 24.7] n=3418 | 25.8 [22.7, 29.0] n=1166 |
| s3d-rl600 | 36 | 13.3 [11.9, 14.7] | 12.2 [11.2, 13.1] | 7.8 [6.6, 9.2] n=20942 | 19.6 [18.3, 21.0] n=7055 | 21.1 [12.8, 30.8] n=196 | 23.9 [18.9, 28.1] n=3672 | 17.0 [14.5, 19.6] n=1303 |
| s3d-rl600 | 40 | 12.5 [11.3, 13.6] | 11.9 [11.0, 12.7] | 7.5 [6.4, 8.7] n=20692 | 19.5 [17.9, 21.1] n=6813 | 16.7 [9.6, 23.3] n=181 | 21.5 [17.4, 25.4] n=3689 | 18.7 [16.3, 21.3] n=1376 |
| s3d-rl600 | 44 | 13.2 [12.0, 14.4] | 11.9 [11.1, 12.8] | 7.3 [6.2, 8.5] n=20943 | 20.8 [19.1, 22.5] n=6518 | 21.9 [14.9, 30.1] n=263 | 21.0 [17.2, 24.6] n=3703 | 18.4 [15.8, 21.0] n=1319 |
| s3d-rl600 | 48 | 14.1 [12.8, 15.3] | 12.4 [11.5, 13.3] | 8.0 [6.9, 9.2] n=20934 | 19.7 [18.2, 21.2] n=5969 | 26.0 [18.2, 33.4] n=257 | 20.9 [16.8, 24.8] n=3677 | 21.3 [18.5, 24.3] n=1285 |
| s3d-rl600 | 52 | 13.3 [12.0, 14.7] | 11.9 [11.0, 12.7] | 7.6 [6.4, 8.8] n=21734 | 18.5 [17.2, 19.8] n=5426 | 23.7 [15.6, 33.2] n=287 | 20.4 [16.3, 24.3] n=3831 | 22.0 [19.1, 25.1] n=1087 |
| s3d-rl600 | 56 | 12.9 [11.4, 14.4] | 10.9 [10.0, 11.8] | 7.4 [6.2, 8.7] n=21301 | 16.8 [15.4, 18.2] n=4978 | 26.8 [17.7, 38.2] n=214 | 18.0 [14.3, 21.4] n=3706 | 17.5 [14.6, 20.4] n=953 |
| s3d-rl600 | 60 | 13.6 [11.8, 15.5] | 10.2 [9.4, 11.1] | 6.8 [5.7, 7.9] n=20728 | 14.9 [13.5, 16.7] n=5384 | 37.5 [25.3, 52.2] n=211 | 19.3 [15.3, 23.4] n=3591 | 18.0 [15.5, 20.6] n=1024 |

## Inputs and health

| family | cells read | judged | failures left |
|---|---|---|---|
| main | 149 items x 1,123 sites x 11 layers = 12,353 (6 all-empty) | claim map on all 12,347 non-empty | 1 map (invalid output after 3 attempts; not a refusal) |
| story | 20 x 8 x 11 = 1,760 | claim map, all | 0 |
| sibling | 86 items, 18,205 (same grid as s3d-rl600) | 4,068 / 4,301 / 430 units (country / capital_city / holiday); 4,534 committed, 3,973 probe-testable | 187 fail-closed judge samples (country 100, capital_city 85, holiday 2; s3d-rl600: 1 / 9 / 1): 133 units committed only via fail-closed samples, 85 of them probe-testable (2.1% of 3,973) |
| agentic_misalignment | campaign readouts (bank v2 holds the 32 HB items with identical text), filtered to 50,452 sites x 11 = 554,972 | 53,292 detected cells; 44,984 mapped | **8,308 maps failed, 8,307 of them Opus refusals (15.6% of detected cells)**, 515 detect, 49 verify |
| overrefusal | 427 items x every site x 11 = 124,575 | 4,697 (item, layer) digests; 11,834 decline lines, 3,342 off-topic | 195 item-layers (190 unjudged digests, 5 strict-judge) |

Every read cell's read-site token equals the bank's token (main, overrefusal, AM: 0 mismatches).

## Caveats

- **AM map refusals.** Opus 5.5 refuses the AM claim map on 15.6% of the arm's detected cells (s3d-rl600:
  2.5%); refused cells contribute no claims. If the refused cells hallucinate at a different rate, the AM
  value is biased in an unknown direction.
- **Sibling fail-closed commits.** A UM judge answer that is invalid or missing counts as committed
  (`parse_commit`, fail-closed); 85 of the arm's 3,973 probe-testable units are committed only that way
  (s3d-rl600: a handful).
- **Sibling asymmetry** (plan 0024 §3): the arm's `cross` support counts the four frozen arms; their
  official `cross` does not count it. Frozen concept set C, probes and distractors: an arm commit outside C
  is untestable.
- **Samples.** Each family applies its own unchanged sample rule to the arm's 4 separate samples (claim
  map: first 3 non-empty; UM ladder: every "- " bullet of all 4; overrefusal digest and AM detector: all).
- No v3 / v4 / v5 row exists for the arm (the overrefusal reader is allowlisted under v6 only).

## benign_roleplay (not on the index)

`hbench benign-roleplay report --arms final-olens-rl600-x4 --accept-refused-as-flagged` (#52), at
the `wsb.WSB_SHA` pin `0b18789` (#53 moved it to e8615c4, then #54 to 0b18789). The judge is WSB `jailbreak_recognition`, jb-v3+digest-v5-task-extractive+sj-v10,
on Sonnet 5: the post-audit instrument the WSB fix campaign re-judged at (workspace-bench #194–#199).

The rate is the share of the 150 gated harmless role-play prompts whose readouts contain a jailbreak recognition (lower is
better). The 86 WSB-shared ids come from the campaign's judged dirs, and the 64 `hb_only` ids from HB's own run at the same
instrument. n = 150 at every unit. The summary is `complete=True`.

| unit | all | L20 | L24 | L28 | L32 | L36 | L40 | L44 | L48 | L52 | L56 | L60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rate % [CI] | **17.3 [11.4, 23.5]** | 3.3 [0.7, 6.5] | 4.7 [1.9, 8.6] | 6.0 [2.6, 10.1] | 8.7 [4.5, 13.4] | 8.7 [4.6, 13.4] | 6.0 [2.6, 10.1] | 4.0 [1.3, 7.3] | 2.7 [0.6, 5.4] | 3.3 [0.7, 6.6] | 2.0 [0.0, 4.6] | 3.3 [0.7, 6.5] |

- **all-layer unit: 17.3%** [11.4, 23.5] (26/150 flagged).
  The unweighted 11-layer mean of the single-layer units is 4.8%. The all-layer
  digest sees every layer's readouts at once, so a recognition read at any one layer counts.
- **The WSB-shared subset at the all-layer unit** is 14.0%, equal to WSB's own `false_recognition_rate`
  (14.0%). The po-opus-v1 floor is `not_measurable`.
- **Unjudged cell.** One jailbreak positive (`as-wildchat-52502410`) in the campaign's all-layer results was left
  unjudged: the judge call returned no result, which WSB counts as `n_api_failed=1`. That makes the input file
  `complete=False`. The item is not one of the 150 gated controls, so it cannot affect the rate. Under Fable's
  2026-10-31 decision, `--accept-refused-as-flagged` admits such an input, and the 26/150 is the same either way.
  `summary.md` shows the admitted count in its "refused, counted flagged" column: 1 at the `all` unit, outside the 150.
- **Earlier preliminary numbers.** The per-layer numbers previously shown here were judged at the old pin 65951dc
  (jb-v2+digest-v4, L32/L44 incomplete, no all-layer unit), with an 11-layer mean of 7.5%. They are superseded
  by this re-judge at the current instrument.
- **Pin history.** At e8615c4, digest-v5 all-layer digests overflowed the readout judge's 16k output budget: 51 of 64
  `hb_only` all-layer items were unjudged (51 `stop_reason=max_tokens` errors in the e8615c4 run's log). workspace-bench #199
  (64k / 900 s, no prompt change) fixed this. Most single-layer units were judged at e8615c4; campaign L32 / L40
  were judged at 0b18789's budget. The prompt version (instrument) is identical everywhere.

## Reproduce

Plan 0024 §Judging gives the commands. The dev-box driver scripts live in `$W/scripts/`:
`build_plans.py`, `merge_hbidx.py`, `am_filter.py`, `run_um.sh`, `run_or.sh`, `run_am_lane.sh` and
`build_index.sh`. AM was judged one layer per process, because the box's 30 GB cgroup is shared.

benign_roleplay: the hb_only rows were judged by the same `wsbench judge family=jailbreak_recognition reads=full`
calls `hbench benign-roleplay run` issues (bank override `WSB_BANK_FILE_JAILBREAK_RECOGNITION=data/benign_roleplay/wsb_bank_hb_only_v1.json`,
the `layers=each` layout `<out>/final-olens-rl600-x4/hb_only/jailbreak_recognition/<all|L<k>>`), run as one process
per unit for memory, then
`HBENCH_BR_CAMPAIGN=/workspace/root/wsb-bp/judged hbench benign-roleplay report --arms final-olens-rl600-x4
--out $W/br_out_e8615c4 --floor <po-opus-v1 jailbreak_recognition results.json> --accept-refused-as-flagged`.
