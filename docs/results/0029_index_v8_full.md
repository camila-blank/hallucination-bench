# Hallucination index (hindex-v8)

% hallucinated per (method, layer), 95% CIs. Family weights main 9, am_claims 3, am_situation 3, sibling 1, story 1, overrefusal 1, benign_roleplay 2; `full` = all seven families, `matched` = all but sibling, `full_j6` = the six families with a joint unit, `full_joint` = that macro on the joint row (layer `all`: sample 0 of all 11 layers in one judge call). Per-layer uses k = 3 samples (J-lens 1), joint 1 per layer. Leaf weights: main / am_claims false_claim 1, substitution_category 0.5, substitution_sibling 0.5, detail 0.2; story false_claim 1, substitution_category 1, detail 0.2, substitution_sibling 0.

| method | layer | full | matched | full_j6 | full_joint | main | am_claims | am_situation | sibling | story | overrefusal | benign_roleplay | main fc / cat / sib / det |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| jlens | 20 | — | 10.0 [9.1, 11.0] | 10.0 [9.1, 11.0] | — | 14.1 [13.3, 15.0] | 4.2 [0.0, 9.0] | 12.1 [9.4, 15.0] | — | 14.5 [9.7, 20.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 13.9 / 0.3 / 0.1 / 0.3 |
| jlens | 24 | — | 7.0 [6.4, 7.6] | 7.0 [6.4, 7.6] | — | 10.5 [9.8, 11.2] | 5.2 [3.4, 7.4] | 4.5 [2.7, 6.4] | — | 9.5 [6.8, 13.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 10.2 / 0.2 / 0.2 / 0.3 |
| jlens | 28 | — | 6.1 [5.6, 6.6] | 6.1 [5.6, 6.6] | — | 8.9 [8.2, 9.5] | 4.4 [2.9, 6.0] | 4.9 [2.9, 7.0] | — | 8.2 [5.5, 11.2] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 8.6 / 0.2 / 0.2 / 0.3 |
| jlens | 32 | — | 6.2 [5.6, 6.8] | 6.2 [5.6, 6.8] | — | 7.8 [7.2, 8.3] | 7.9 [6.1, 9.9] | 6.2 [3.7, 9.0] | — | 6.3 [3.5, 9.4] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 7.5 / 0.2 / 0.1 / 0.3 |
| jlens | 36 | — | 3.3 [2.9, 3.6] | 3.3 [2.9, 3.6] | — | 4.0 [3.6, 4.4] | 4.8 [3.5, 6.3] | 2.0 [0.8, 3.1] | — | 5.9 [3.7, 8.1] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 3.7 / 0.1 / 0.3 / 0.2 |
| jlens | 40 | — | 3.3 [3.0, 3.7] | 3.3 [3.0, 3.7] | — | 4.2 [3.8, 4.6] | 6.5 [5.4, 7.9] | 0.4 [0.0, 1.0] | — | 3.7 [1.7, 5.8] | 0.0 [0.0, 0.0] | 0.7 [0.0, 2.1] | 3.9 / 0.1 / 0.3 / 0.3 |
| jlens | 44 | — | 3.1 [2.7, 3.5] | 3.1 [2.7, 3.5] | — | 3.6 [3.2, 3.9] | 5.8 [4.3, 7.5] | 1.0 [0.2, 1.8] | — | 4.4 [2.7, 6.4] | 2.3 [0.0, 5.8] | 0.0 [0.0, 0.0] | 3.3 / 0.1 / 0.3 / 0.4 |
| jlens | 48 | — | 3.4 [3.0, 3.9] | 3.4 [3.0, 3.9] | — | 4.4 [4.0, 4.8] | 5.3 [3.3, 7.4] | 0.6 [0.0, 1.2] | — | 5.0 [2.7, 7.5] | 0.0 [0.0, 0.0] | 1.3 [0.0, 3.3] | 4.2 / 0.0 / 0.3 / 0.4 |
| jlens | 52 | — | 2.9 [2.5, 3.3] | 2.9 [2.5, 3.3] | — | 3.7 [3.3, 4.1] | 3.0 [1.9, 4.2] | 2.1 [1.2, 3.1] | — | 6.8 [2.9, 11.9] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 3.4 / 0.1 / 0.3 / 0.4 |
| jlens | 56 | — | 2.9 [2.6, 3.3] | 2.9 [2.6, 3.3] | — | 4.4 [4.1, 4.8] | 2.0 [0.9, 3.5] | 1.4 [0.6, 2.3] | — | 5.5 [2.8, 8.6] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 4.1 / 0.1 / 0.3 / 0.6 |
| jlens | 60 | — | 3.4 [3.1, 3.8] | 3.4 [3.1, 3.8] | — | 5.4 [5.0, 5.9] | 2.5 [1.1, 4.2] | 1.2 [0.4, 2.0] | — | 5.4 [3.4, 7.5] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 5.1 / 0.2 / 0.2 / 0.7 |
| jlens | all | — | — | — | 5.1 [4.6, 5.5] | 6.7 [6.4, 7.0] | 3.2 [2.4, 4.1] | 6.4 [4.1, 9.2] | — | 7.0 [5.5, 8.5] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 6.4 / 0.2 / 0.3 / 0.2 |
| mlnla-rl600 | 20 | 32.4 [31.1, 33.7] | 31.8 [30.5, 33.1] | 31.8 [30.5, 33.1] | — | 24.0 [23.4, 24.6] | 63.8 [61.5, 65.6] | 42.2 [35.7, 48.6] | 42.8 [34.3, 51.0] | 21.3 [19.1, 23.3] | 37.4 [27.5, 47.3] | 6.0 [2.6, 10.1] | 15.1 / 3.1 / 5.8 / 22.7 |
| mlnla-rl600 | 24 | 33.6 [32.3, 34.9] | 33.7 [32.4, 34.9] | 33.7 [32.4, 34.9] | — | 24.8 [24.2, 25.5] | 65.5 [63.4, 67.1] | 43.9 [37.1, 50.4] | 32.6 [19.4, 44.5] | 21.8 [19.5, 23.9] | 52.3 [45.5, 58.9] | 6.7 [3.2, 10.8] | 15.7 / 3.2 / 5.9 / 22.8 |
| mlnla-rl600 | 28 | 33.9 [32.6, 35.2] | 33.9 [32.5, 35.2] | 33.9 [32.5, 35.2] | — | 23.5 [22.8, 24.1] | 63.3 [61.2, 65.2] | 51.7 [44.3, 58.7] | 34.8 [28.0, 41.5] | 23.0 [20.9, 25.0] | 42.9 [36.9, 49.3] | 10.7 [6.0, 15.5] | 14.1 / 3.6 / 5.9 / 23.1 |
| mlnla-rl600 | 32 | 33.3 [31.8, 34.9] | 33.4 [31.9, 34.9] | 33.4 [31.9, 34.9] | — | 23.4 [22.7, 24.1] | 66.5 [64.4, 68.3] | 54.2 [45.3, 62.7] | 32.3 [22.4, 43.8] | 22.8 [20.7, 24.9] | 31.1 [26.1, 36.3] | 4.0 [1.3, 7.5] | 13.8 / 3.6 / 6.0 / 23.5 |
| mlnla-rl600 | 36 | 30.3 [28.9, 31.8] | 30.4 [28.9, 31.8] | 30.4 [28.9, 31.8] | — | 21.1 [20.4, 21.7] | 62.6 [60.3, 64.7] | 51.4 [43.2, 59.6] | 29.1 [20.8, 38.1] | 19.1 [17.3, 21.1] | 13.5 [10.0, 17.2] | 6.7 [2.7, 11.0] | 12.7 / 2.0 / 5.3 / 23.4 |
| mlnla-rl600 | 40 | 31.3 [29.9, 32.7] | 31.5 [30.2, 32.9] | 31.5 [30.2, 32.9] | — | 22.2 [21.6, 22.8] | 64.9 [62.7, 66.8] | 50.0 [42.5, 57.6] | 27.2 [19.4, 36.2] | 20.6 [18.9, 22.5] | 18.5 [14.5, 22.7] | 7.3 [3.4, 11.6] | 13.9 / 2.3 / 5.1 / 23.4 |
| mlnla-rl600 | 44 | 30.3 [29.0, 31.7] | 30.4 [29.1, 31.8] | 30.4 [29.1, 31.8] | — | 22.0 [21.3, 22.6] | 65.9 [63.5, 67.8] | 42.4 [35.0, 50.0] | 28.5 [20.8, 37.7] | 20.7 [18.6, 22.9] | 21.5 [17.7, 25.6] | 6.7 [3.2, 10.9] | 13.2 / 2.4 / 5.7 / 23.7 |
| mlnla-rl600 | 48 | 30.9 [29.5, 32.2] | 30.9 [29.6, 32.3] | 30.9 [29.6, 32.3] | — | 23.1 [22.4, 23.7] | 68.4 [65.9, 70.3] | 39.8 [32.7, 47.3] | 29.3 [22.5, 37.1] | 21.2 [19.0, 23.4] | 19.8 [15.7, 24.2] | 7.3 [3.8, 11.6] | 14.2 / 2.6 / 5.9 / 23.1 |
| mlnla-rl600 | 52 | 29.9 [28.4, 31.4] | 29.8 [28.4, 31.3] | 29.8 [28.4, 31.3] | — | 22.6 [21.9, 23.2] | 66.3 [63.9, 68.1] | 38.5 [30.5, 46.7] | 30.7 [20.8, 42.5] | 19.9 [17.6, 22.2] | 13.5 [9.9, 17.1] | 8.0 [4.0, 12.6] | 14.8 / 1.4 / 4.8 / 23.4 |
| mlnla-rl600 | 56 | 30.5 [29.3, 31.8] | 30.3 [29.1, 31.5] | 30.3 [29.1, 31.5] | — | 24.8 [24.2, 25.4] | 66.1 [63.6, 68.0] | 36.7 [30.5, 43.4] | 35.0 [24.6, 45.1] | 21.4 [19.3, 23.4] | 14.4 [10.6, 18.4] | 4.0 [1.3, 7.3] | 17.4 / 1.2 / 4.3 / 23.5 |
| mlnla-rl600 | 60 | 32.5 [31.3, 33.7] | 31.7 [30.6, 32.9] | 31.7 [30.6, 32.9] | — | 26.9 [26.2, 27.6] | 67.8 [65.6, 69.4] | 37.9 [31.8, 43.9] | 47.4 [39.4, 57.9] | 22.6 [20.6, 24.7] | 12.0 [8.8, 15.5] | 4.7 [1.4, 8.2] | 19.7 / 1.2 / 3.8 / 23.4 |
| mlnla-rl600 | all | — | — | — | 36.0 [34.6, 37.4] | 22.1 [21.6, 22.7] | 55.8 [53.1, 58.1] | 73.8 [66.0, 80.3] | — | 21.1 [19.5, 22.9] | 41.9 [36.4, 47.0] | 16.7 [10.7, 22.7] | 12.8 / 2.4 / 6.0 / 25.7 |
| s3d-rl600 | 20 | 11.0 [10.1, 11.9] | 9.5 [8.9, 10.1] | 9.5 [8.9, 10.1] | — | 9.3 [8.6, 10.0] | 18.7 [15.9, 21.5] | 0.2 [0.0, 0.6] | 39.4 [29.9, 56.0] | 11.2 [8.7, 13.8] | 29.3 [25.4, 33.6] | 0.0 [0.0, 0.0] | 5.0 / 1.2 / 3.9 / 8.5 |
| s3d-rl600 | 24 | 10.7 [10.1, 11.4] | 10.3 [9.7, 10.9] | 10.3 [9.7, 10.9] | — | 9.0 [8.4, 9.7] | 23.3 [20.6, 25.7] | 0.6 [0.0, 1.4] | 18.7 [13.5, 26.5] | 10.5 [8.3, 12.8] | 32.6 [28.6, 36.9] | 0.0 [0.0, 0.0] | 4.8 / 1.2 / 3.9 / 8.1 |
| s3d-rl600 | 28 | 11.6 [10.8, 12.4] | 10.8 [10.2, 11.5] | 10.8 [10.2, 11.5] | — | 9.3 [8.6, 10.0] | 22.4 [19.7, 24.9] | 0.6 [0.0, 1.2] | 26.1 [17.6, 36.4] | 11.6 [9.5, 14.0] | 38.0 [34.7, 41.6] | 2.0 [0.0, 4.6] | 5.2 / 1.2 / 3.8 / 7.6 |
| s3d-rl600 | 32 | 10.9 [10.2, 11.5] | 10.5 [9.9, 11.2] | 10.5 [9.9, 11.2] | — | 8.8 [8.1, 9.5] | 23.5 [21.0, 25.8] | 1.6 [0.6, 2.7] | 17.1 [12.4, 21.6] | 10.4 [8.6, 12.2] | 27.5 [24.6, 30.7] | 4.0 [1.3, 7.4] | 4.7 / 1.2 / 3.7 / 8.4 |
| s3d-rl600 | 36 | 9.1 [8.4, 9.8] | 8.5 [7.9, 9.1] | 8.5 [7.9, 9.1] | — | 7.3 [6.6, 8.0] | 22.1 [19.5, 24.7] | 0.4 [0.0, 1.0] | 21.1 [12.8, 30.8] | 10.0 [7.3, 13.0] | 17.3 [14.9, 19.8] | 0.7 [0.0, 2.1] | 3.3 / 0.9 / 3.8 / 8.2 |
| s3d-rl600 | 40 | 9.6 [8.9, 10.2] | 9.2 [8.6, 9.8] | 9.2 [8.6, 9.8] | — | 7.3 [6.7, 7.9] | 24.6 [22.1, 26.9] | 1.8 [0.8, 2.7] | 16.7 [9.6, 23.3] | 7.8 [5.9, 9.8] | 19.8 [17.4, 22.5] | 1.3 [0.0, 3.4] | 3.3 / 0.9 / 3.8 / 8.0 |
| s3d-rl600 | 44 | 9.7 [9.0, 10.4] | 9.1 [8.5, 9.7] | 9.1 [8.5, 9.7] | — | 7.4 [6.8, 8.1] | 23.4 [20.6, 25.8] | 0.8 [0.2, 1.6] | 21.9 [14.9, 30.1] | 8.6 [6.6, 10.8] | 18.8 [16.3, 21.3] | 2.7 [0.6, 5.4] | 3.4 / 0.9 / 3.9 / 8.3 |
| s3d-rl600 | 48 | 10.0 [9.3, 10.7] | 9.2 [8.6, 9.8] | 9.2 [8.6, 9.8] | — | 7.4 [6.7, 8.0] | 23.0 [20.2, 25.5] | 1.6 [0.8, 2.5] | 26.0 [18.2, 33.4] | 9.6 [7.5, 11.6] | 21.9 [19.2, 24.8] | 1.3 [0.0, 3.4] | 3.3 / 0.9 / 3.8 / 8.3 |
| s3d-rl600 | 52 | 8.5 [7.9, 9.1] | 7.7 [7.2, 8.2] | 7.7 [7.2, 8.2] | — | 6.5 [5.9, 7.1] | 19.3 [17.2, 21.1] | 0.4 [0.0, 1.0] | 23.7 [15.6, 33.2] | 6.9 [5.1, 8.6] | 21.8 [19.0, 24.9] | 0.0 [0.0, 0.0] | 2.8 / 0.6 / 3.6 / 7.9 |
| s3d-rl600 | 56 | 8.6 [7.9, 9.3] | 7.6 [7.1, 8.2] | 7.6 [7.1, 8.2] | — | 6.5 [5.9, 7.1] | 18.7 [16.4, 20.7] | 0.4 [0.0, 1.0] | 26.8 [17.7, 38.2] | 6.8 [5.0, 8.6] | 18.8 [15.9, 21.6] | 2.0 [0.0, 4.5] | 3.0 / 0.5 / 3.5 / 7.3 |
| s3d-rl600 | 60 | 8.9 [8.1, 9.8] | 7.4 [6.9, 7.9] | 7.4 [6.9, 7.9] | — | 6.5 [5.9, 7.1] | 17.2 [15.5, 18.9] | 0.8 [0.2, 1.6] | 37.5 [25.3, 52.2] | 6.6 [5.1, 8.3] | 18.2 [15.7, 20.8] | 2.0 [0.0, 4.6] | 3.1 / 0.5 / 3.4 / 7.2 |
| s3d-rl600 | all | — | — | — | 9.6 [8.9, 10.2] | 7.2 [6.6, 7.9] | 18.3 [16.4, 20.1] | 3.1 [2.0, 4.3] | — | 6.7 [5.5, 7.8] | 31.6 [29.1, 34.4] | 7.3 [3.4, 11.6] | 3.5 / 0.9 / 3.7 / 7.0 |

## Mean over layers (plot 3; shared item bootstrap across layers, not the per-layer normal approximation)

| method | macro | mean [95% CI] | layers | dropped draws |
|---|---|---|---|---:|
| jlens | matched | 4.7 [4.5, 5.0] | 20,24,28,32,36,40,44,48,52,56,60 | 58.6% |
| jlens | full_j6 | 4.7 [4.5, 5.0] | 20,24,28,32,36,40,44,48,52,56,60 | 58.6% |
| mlnla-rl600 | full | 31.7 [30.5, 32.9] | 20,24,28,32,36,40,44,48,52,56,60 | 2.2% |
| mlnla-rl600 | matched | 31.6 [30.3, 32.8] | 20,24,28,32,36,40,44,48,52,56,60 | 0.0% |
| mlnla-rl600 | full_j6 | 31.6 [30.3, 32.8] | 20,24,28,32,36,40,44,48,52,56,60 | 0.0% |
| s3d-rl600 | full | 9.9 [9.4, 10.4] | 20,24,28,32,36,40,44,48,52,56,60 | 23.7% |
| s3d-rl600 | matched | 9.1 [8.6, 9.5] | 20,24,28,32,36,40,44,48,52,56,60 | 0.0% |
| s3d-rl600 | full_j6 | 9.1 [8.6, 9.5] | 20,24,28,32,36,40,44,48,52,56,60 | 0.0% |

Arms used under another name (at one or more layers): mlnla-rl600.am_claims = mlnla-rl600-all11; mlnla-rl600.am_situation = mlnla-rl600-all11; mlnla-rl600.benign_roleplay = mlnla-rl600-all11; mlnla-rl600.main = mlnla-rl600-all11; mlnla-rl600.overrefusal = mlnla-rl600-all11; mlnla-rl600.story = mlnla-rl600-all11; s3d-rl600.main = s3d-rl600-all11; s3d-rl600.sibling = s3d; s3d-rl600.story = s3d-rl600-all11.

Excluded arms: nla-rl-L42, nla-sft-L42.

- hindex-v8: plan 0029. Families / weights: main 9, am_claims 3, am_situation 3, sibling 1, story 1, overrefusal 1, benign_roleplay 2 (Σ 20; cheating is not on the index: archived as a diagnostic, plan 0029 amendment 3); `matched` = all but sibling (Σ 19); full_v7 / matched_v7 = hindex-v7's families and weights on v8 values (AM = am_claims). Claim maps judged with claimmap-v2.9; the continuation excuse is decided by the continuation verifier cverify-v1, not by the judge's tags (plan 0029 amendment 2: a segment contradicted only after the marker is excused iff the verifier finds it a literal continuation that fills the same slot and is on topic; excused segments score `continuation_miss`, weight 0, still claims). main and am_claims use the main leaf weights (false_claim 1, substitution_category 0.5, substitution_sibling 0.5, detail 0.2); story down-weights detail only (false_claim 1, substitution_category 1, detail 0.2, substitution_sibling 0). am_claims counts candidate claims (plan 0021); am_situation is the amsit-v2 share of cells whose readouts misrepresent the situation as a whole (prose k = 3 samples, J-lens 1 summary: the rate rises with k). The joint row (layer `all`) judges sample 0 of all 11 layers of a read site in one call (J-lens: the 11 per-layer summaries); `full_joint` and every per-layer row's `full_j6` are the six-family macro over the families with a joint unit (main, story, am_claims, am_situation, overrefusal, benign_roleplay; Σ 19; the same families as `matched`). Per-layer uses k = 3 (J-lens 1), joint 1 per layer; the joint am_situation ('any of 11') is not comparable to per-layer. Reader allowlists exclude final-olens-rl600-x4. Pre-gate.
- jlens.sibling: no sibling score: J-lens has committed, probe-testable UM units in at most two of the three domains at every layer, so only `matched` compares it
- mlnla-rl600.safety: overrefusal reads the mlnla-rl600-all11 arm (plan 0019); it uses the allowlisted readers s3d-rl600, mlnla-rl600-all11 and jlens only
- nla-rl-L42: single-layer L42 NLA arms (nla-rl-L42, nla-sft-L42) excluded (Camila 2026-10-20)
- s3d-rl600.main: the s3d-rl600 arm wins at the layers it has, s3d-rl600-all11 fills the rest
- v4: DeepSeek-V4-Flash arms (v4-*, plan 0014) are matched by name; their roots are passed beside the Qwen ones. V4 AM is read at layers 21/27/33 only, so V4 `full` / `matched` exist at 21/27/33 only
- v4-jlens-std-n1000.sibling: no sibling score: no committed, probe-testable country units at any layer (as for jlens)

## Cells excluded as persistent judge failures (plan 0029 Amendment 4)

A cell that stayed failed after at least 3 attempts (both models refused, or an invalid answer after the in-run retries) is dropped from the index, never scored; counted here.

Exclusions are not random: judges refuse harmful-sounding readouts, so each method is scored on its own (method-dependent) item set; safety `n_items` / bootstrap clusters still include the excluded all-zero rows. `review` = excluded share above 5%.

| family | method | unit | cells | excluded | % | reasons | review |
|---|---|---|---:|---:|---:|---|---|
| main | s3d-rl600 | L20 | 5922 | 1 | 0.02 | refused 1 |  |
| main | s3d-rl600 | L28 | 5923 | 1 | 0.02 | refused 1 |  |
| main | s3d-rl600 | L32 | 5921 | 2 | 0.03 | refused 2 |  |
| main | s3d-rl600 | L36 | 5923 | 2 | 0.03 | invalid 2 |  |
| main | s3d-rl600 | L44 | 5924 | 2 | 0.03 | invalid 1, refused 1 |  |
| main | s3d-rl600 | L48 | 5923 | 1 | 0.02 | refused 1 |  |
| main | s3d-rl600 | L52 | 5924 | 2 | 0.03 | invalid 2 |  |
| main | s3d-rl600 | L56 | 5923 | 3 | 0.05 | invalid 1, refused 2 |  |
| main | s3d-rl600 | L60 | 5924 | 5 | 0.08 | refused 5 |  |
| main | s3d-rl600 | all | 5924 | 4 | 0.07 | refused 4 |  |
| am_claims | s3d-rl600 | L20 | 50452 | 2 | 0.00 | refused 2 |  |
| am_claims | s3d-rl600 | L24 | 50452 | 10 | 0.02 | refused 10 |  |
| am_claims | s3d-rl600 | L28 | 50452 | 5 | 0.01 | refused 5 |  |
| am_claims | s3d-rl600 | L32 | 50452 | 5 | 0.01 | refused 5 |  |
| am_claims | s3d-rl600 | L36 | 50452 | 16 | 0.03 | refused 16 |  |
| am_claims | s3d-rl600 | L40 | 50452 | 17 | 0.03 | refused 17 |  |
| am_claims | s3d-rl600 | L44 | 50452 | 13 | 0.03 | refused 13 |  |
| am_claims | s3d-rl600 | L48 | 50452 | 9 | 0.02 | refused 9 |  |
| am_claims | s3d-rl600 | L52 | 50452 | 7 | 0.01 | refused 7 |  |
| am_claims | s3d-rl600 | L56 | 50452 | 2 | 0.00 | refused 2 |  |
| am_claims | s3d-rl600 | L60 | 50452 | 8 | 0.02 | refused 8 |  |
| am_claims | s3d-rl600 | all | 14760 | 20 | 0.14 | invalid 1, refused 19 |  |
| am_situation | s3d-rl600 | all | 512 | 1 | 0.20 | refused 1 |  |
| overrefusal | s3d-rl600 | L20 | 427 | 13 | 3.04 | refused 13 |  |
| overrefusal | s3d-rl600 | L24 | 427 | 21 | 4.92 | refused 21 |  |
| overrefusal | s3d-rl600 | L28 | 427 | 19 | 4.45 | refused 19 |  |
| overrefusal | s3d-rl600 | L32 | 427 | 30 | 7.03 | refused 30 | review |
| overrefusal | s3d-rl600 | L36 | 427 | 12 | 2.81 | refused 12 |  |
| overrefusal | s3d-rl600 | L40 | 427 | 15 | 3.51 | refused 15 |  |
| overrefusal | s3d-rl600 | L44 | 427 | 18 | 4.22 | refused 18 |  |
| overrefusal | s3d-rl600 | L48 | 427 | 18 | 4.22 | refused 18 |  |
| overrefusal | s3d-rl600 | L52 | 427 | 19 | 4.45 | refused 19 |  |
| overrefusal | s3d-rl600 | L56 | 427 | 15 | 3.51 | refused 15 |  |
| overrefusal | s3d-rl600 | L60 | 427 | 12 | 2.81 | refused 12 |  |
| overrefusal | s3d-rl600 | all | 427 | 28 | 6.56 | refused 28 | review |
| main | mlnla-rl600 | L28 | 5924 | 1 | 0.02 | refused 1 |  |
| main | mlnla-rl600 | L44 | 5924 | 1 | 0.02 | refused 1 |  |
| main | mlnla-rl600 | L56 | 5924 | 1 | 0.02 | refused 1 |  |
| main | mlnla-rl600 | all | 5924 | 2 | 0.03 | refused 2 |  |
| am_claims | mlnla-rl600 | L20 | 50452 | 8 | 0.02 | refused 8 |  |
| am_claims | mlnla-rl600 | L24 | 50452 | 15 | 0.03 | refused 15 |  |
| am_claims | mlnla-rl600 | L28 | 50452 | 23 | 0.05 | refused 23 |  |
| am_claims | mlnla-rl600 | L32 | 50452 | 29 | 0.06 | refused 29 |  |
| am_claims | mlnla-rl600 | L36 | 50452 | 38 | 0.08 | refused 38 |  |
| am_claims | mlnla-rl600 | L40 | 50452 | 47 | 0.09 | refused 47 |  |
| am_claims | mlnla-rl600 | L44 | 50452 | 66 | 0.13 | refused 66 |  |
| am_claims | mlnla-rl600 | L48 | 50452 | 32 | 0.06 | refused 32 |  |
| am_claims | mlnla-rl600 | L52 | 50452 | 25 | 0.05 | refused 25 |  |
| am_claims | mlnla-rl600 | L56 | 50452 | 35 | 0.07 | refused 35 |  |
| am_claims | mlnla-rl600 | L60 | 50452 | 26 | 0.05 | refused 26 |  |
| am_claims | mlnla-rl600 | all | 35443 | 398 | 1.12 | invalid 391, refused 7 |  |
| am_situation | mlnla-rl600 | L28 | 512 | 1 | 0.20 | refused 1 |  |
| am_situation | mlnla-rl600 | L32 | 512 | 1 | 0.20 | refused 1 |  |
| am_situation | mlnla-rl600 | L40 | 512 | 2 | 0.39 | refused 2 |  |
| am_situation | mlnla-rl600 | L48 | 512 | 2 | 0.39 | refused 2 |  |
| overrefusal | mlnla-rl600 | L20 | 427 | 14 | 3.28 | refused 14 |  |
| overrefusal | mlnla-rl600 | L24 | 427 | 25 | 5.85 | refused 25 | review |
| overrefusal | mlnla-rl600 | L28 | 427 | 17 | 3.98 | refused 17 |  |
| overrefusal | mlnla-rl600 | L32 | 427 | 16 | 3.75 | refused 16 |  |
| overrefusal | mlnla-rl600 | L36 | 427 | 14 | 3.28 | refused 14 |  |
| overrefusal | mlnla-rl600 | L40 | 427 | 16 | 3.75 | refused 16 |  |
| overrefusal | mlnla-rl600 | L44 | 427 | 16 | 3.75 | refused 16 |  |
| overrefusal | mlnla-rl600 | L48 | 427 | 15 | 3.51 | refused 15 |  |
| overrefusal | mlnla-rl600 | L52 | 427 | 13 | 3.04 | refused 13 |  |
| overrefusal | mlnla-rl600 | L56 | 427 | 14 | 3.28 | refused 14 |  |
| overrefusal | mlnla-rl600 | L60 | 427 | 11 | 2.58 | refused 11 |  |
| overrefusal | mlnla-rl600 | all | 427 | 37 | 8.67 | refused 37 | review |
| main | jlens | L24 | 5924 | 2 | 0.03 | refused 2 |  |
| main | jlens | L28 | 5924 | 1 | 0.02 | refused 1 |  |
| main | jlens | all | 5924 | 2 | 0.03 | refused 2 |  |
| am_claims | jlens | L24 | 50452 | 2 | 0.00 | refused 2 |  |
| am_claims | jlens | L28 | 50452 | 16 | 0.03 | refused 16 |  |
| am_claims | jlens | L32 | 50452 | 10 | 0.02 | refused 10 |  |
| am_claims | jlens | L36 | 50452 | 9 | 0.02 | refused 9 |  |
| am_claims | jlens | L40 | 50452 | 8 | 0.02 | refused 8 |  |
| am_claims | jlens | L44 | 50452 | 6 | 0.01 | refused 6 |  |
| am_claims | jlens | L48 | 50452 | 4 | 0.01 | refused 4 |  |
| am_claims | jlens | L52 | 50452 | 3 | 0.01 | refused 3 |  |
| am_claims | jlens | L56 | 50452 | 4 | 0.01 | refused 4 |  |
| am_claims | jlens | L60 | 50452 | 3 | 0.01 | refused 3 |  |
| overrefusal | jlens | L20 | 427 | 1 | 0.23 | refused 1 |  |
| overrefusal | jlens | L28 | 427 | 1 | 0.23 | refused 1 |  |
| overrefusal | jlens | L52 | 427 | 1 | 0.23 | refused 1 |  |
| overrefusal | jlens | L56 | 427 | 4 | 0.94 | refused 4 |  |
| overrefusal | jlens | L60 | 427 | 5 | 1.17 | refused 5 |  |
| overrefusal | jlens | all | 427 | 3 | 0.70 | refused 3 |  |

Total excluded: 1404
