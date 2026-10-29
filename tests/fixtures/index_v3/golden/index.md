# Hallucination index (hindex-v3)

% of claims hallucinated per (method, layer), 95% CIs. Family weights main 9, agentic_misalignment 3, sibling 2, story 1; `matched` = main + agentic_misalignment + story. main leaf weights false_claim 1, substitution_category 0.5, substitution_sibling 0.5, detail 0.2; agentic_misalignment / story: unweighted false_claim + substitution_category + detail; sibling: 3-domain UM ladder.

| method | layer | full | matched | main | agentic_misalignment | sibling | story | main fc / cat / sib / det |
|---|---:|---|---|---|---|---|---|---|
| M | 20 | — | — | 45.0 [37.5, 57.7] | — | 36.1 [12.5, 57.7] | 33.3 [25.0, 66.7] | 25.0 / 25.0 / 12.5 / 6.2 |
| M | 24 | — | — | 38.5 [25.0, 75.0] | — | 13.3 [0.0, 33.3] | 0.0 [0.0, 0.0] | 25.0 / 15.0 / 10.0 / 5.0 |
| M | 28 | — | — | 30.0 [21.9, 41.9] | — | 8.3 [0.0, 33.3] | 40.0 [nan, nan] | 0.0 / 18.8 / 31.2 / 25.0 |
| c | 20 | 36.9 [31.3, 42.5] | 38.5 [32.4, 44.6] | 32.3 [26.0, 40.0] | 50.0 [36.7, 63.3] | 26.7 [8.3, 38.3] | 60.0 [35.7, 85.7] | 13.6 / 9.1 / 22.7 / 13.6 |
| c | 24 | 32.8 [nan, nan] | 35.8 [nan, nan] | 29.3 [19.2, 40.0] | 47.1 [28.6, 60.0] | 13.3 [0.0, 33.3] | 60.0 [nan, nan] | 13.8 / 20.7 / 3.4 / 17.2 |
| c | 28 | 26.8 [19.1, 34.4] | 27.7 [19.4, 36.0] | 20.0 [11.2, 28.0] | 50.0 [20.0, 70.0] | 20.6 [4.2, 44.4] | 30.0 [0.0, 33.3] | 8.0 / 4.0 / 12.0 / 20.0 |
| d | 20 | — | — | 25.0 [nan, nan] | 50.0 [37.9, 59.0] | — | — | 0.0 / 50.0 / 0.0 / 0.0 |
| d | 24 | — | — | 32.8 [22.0, 46.2] | 43.8 [28.6, 52.6] | — | — | 22.2 / 11.1 / 5.6 / 11.1 |
| e | 20 | — | — | — | 70.0 [18.2, 100.0] | — | 36.8 [16.7, 41.7] | — |
| e | 24 | — | — | — | 34.8 [11.1, 57.1] | — | 0.0 [0.0, 0.0] | — |
| e | 28 | — | — | — | 47.4 [36.4, 62.5] | — | 43.8 [31.2, 83.3] | — |

Arms used under another name (at one or more layers): M.main = a; M.sibling = s; M.story = a2.

Excluded arms: x.

- jlens.sibling: no sibling score: J-lens has committed, probe-testable UM units in at most two of the three domains at every layer, so only `matched` compares it
- nla-rl-L42: single-layer L42 NLA arms (nla-rl-L42, nla-sft-L42) excluded (Camila 2026-10-20)
- s3d-rl600.main: the s3d-rl600 arm wins at the layers it has, s3d-rl600-all11 fills the rest
- v4: DeepSeek-V4-Flash arms (v4-*, plan 0014) are matched by name; their roots are passed beside the Qwen ones. V4 AM is read at layers 21/27/33 only, so V4 `full` / `matched` exist at 21/27/33 only
- v4-jlens-std-n1000.sibling: no sibling score: no committed, probe-testable country units at any layer (as for jlens)
