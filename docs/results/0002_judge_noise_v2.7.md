## Claim-map judge consistency (claimmap-v2.7), 220 cells, 3058 units compared (537 failed in a run)

| scope | units | κ | raw agreement |
|---|---|---|---|
| overall | 3058 | 0.860 | 0.905 |
| logit | 62 | 0.906 | 0.952 |
| mlnla-rl2-800-all11 | 532 | 0.837 | 0.891 |
| mlnla-rl600-all11 | 531 | 0.866 | 0.908 |
| nla-sft-L42 | 969 | 0.849 | 0.953 |
| patchscopes | 36 | 0.851 | 0.917 |
| rlens | 41 | 0.799 | 0.878 |
| s3d-rl600-all11 | 403 | 0.780 | 0.836 |
| s3d-sft251 | 356 | 0.848 | 0.893 |
| sae-l42 | 35 | 0.725 | 0.857 |
| template-v3 | 47 | 0.827 | 0.872 |
| tuned | 46 | 0.696 | 0.783 |
| family logit | 62 | 0.906 | 0.952 |
| family mlnla | 1063 | 0.852 | 0.899 |
| family nla | 969 | 0.849 | 0.953 |
| family patchscopes | 36 | 0.851 | 0.917 |
| family rlens | 41 | 0.799 | 0.878 |
| family s3d | 759 | 0.812 | 0.863 |
| family sae | 35 | 0.725 | 0.857 |
| family template | 47 | 0.827 | 0.872 |
| family tuned | 46 | 0.696 | 0.783 |

### detail vs false_claim, on units both runs call hallucination

| scope | units | κ | raw agreement |
|---|---|---|---|
| overall | 233 | 0.828 | 0.914 |
| logit | 10 | 1.000 | 1.000 |
| mlnla-rl2-800-all11 | 57 | 0.769 | 0.895 |
| mlnla-rl600-all11 | 69 | 0.845 | 0.928 |
| nla-sft-L42 | 62 | 0.799 | 0.903 |
| patchscopes | 0 | — | — |
| rlens | 0 | — | — |
| s3d-rl600-all11 | 10 | 1.000 | 1.000 |
| s3d-sft251 | 13 | 0.552 | 0.769 |
| sae-l42 | 7 | 1.000 | 1.000 |
| template-v3 | 3 | 1.000 | 1.000 |
| tuned | 2 | 1.000 | 1.000 |

### Headline metrics from run r0 vs run r1 (readout rate / character share, %)

| scope | metric | rate r0 | rate r1 | diff | ± | share r0 | share r1 | diff | ± |
|---|---|---|---|---|---|---|---|---|---|
| overall | hallucination | 42.1 | 43.1 | -1.0 | 0.5 | 17.6 | 17.5 | 0.1 | 0.1 |
| overall | false_claim | 25.1 | 25.9 | -0.8 | 0.4 | 9.5 | 9.1 | 0.4 | 0.2 |
| overall | detail | 25.1 | 26.1 | -1.0 | 0.5 | 8.0 | 8.3 | -0.3 | 0.1 |
| overall | probe_candidate | 13.5 | 12.2 | 1.3 | 0.6 | 4.3 | 4.2 | 0.2 | 0.1 |
| overall | correct | 67.5 | 69.5 | -2.0 | 1.0 | 33.0 | 33.9 | -0.8 | 0.4 |
| overall | unresolved | 62.2 | 58.9 | 3.3 | 1.6 | 26.0 | 26.0 | 0.0 | 0.0 |
| logit | hallucination | 50.0 | 50.0 | 0.0 | 0.0 | 21.2 | 21.2 | 0.0 | 0.0 |
| logit | false_claim | 50.0 | 50.0 | 0.0 | 0.0 | 21.2 | 21.2 | 0.0 | 0.0 |
| logit | detail | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| logit | probe_candidate | 5.0 | 5.0 | 0.0 | 0.0 | 0.7 | 0.7 | 0.0 | 0.0 |
| logit | correct | 15.0 | 10.0 | 5.0 | 2.5 | 5.1 | 3.8 | 1.4 | 0.7 |
| logit | unresolved | 35.0 | 25.0 | 10.0 | 5.0 | 14.0 | 10.6 | 3.4 | 1.7 |
| mlnla-rl2-800-all11 | hallucination | 73.7 | 71.9 | 1.8 | 0.9 | 27.1 | 27.2 | -0.1 | 0.1 |
| mlnla-rl2-800-all11 | false_claim | 29.8 | 35.1 | -5.3 | 2.6 | 8.6 | 11.4 | -2.8 | 1.4 |
| mlnla-rl2-800-all11 | detail | 61.4 | 56.1 | 5.3 | 2.6 | 18.5 | 15.8 | 2.7 | 1.3 |
| mlnla-rl2-800-all11 | probe_candidate | 7.0 | 8.8 | -1.8 | 0.9 | 1.6 | 1.8 | -0.1 | 0.1 |
| mlnla-rl2-800-all11 | correct | 87.7 | 89.5 | -1.8 | 0.9 | 39.9 | 40.6 | -0.7 | 0.3 |
| mlnla-rl2-800-all11 | unresolved | 63.2 | 56.1 | 7.0 | 3.5 | 22.2 | 20.5 | 1.7 | 0.8 |
| mlnla-rl600-all11 | hallucination | 75.0 | 75.0 | 0.0 | 0.0 | 32.0 | 29.8 | 2.2 | 1.1 |
| mlnla-rl600-all11 | false_claim | 37.5 | 39.3 | -1.8 | 0.9 | 10.5 | 9.7 | 0.8 | 0.4 |
| mlnla-rl600-all11 | detail | 60.7 | 58.9 | 1.8 | 0.9 | 21.5 | 20.1 | 1.4 | 0.7 |
| mlnla-rl600-all11 | probe_candidate | 23.2 | 16.1 | 7.1 | 3.6 | 3.7 | 2.5 | 1.2 | 0.6 |
| mlnla-rl600-all11 | correct | 82.1 | 89.3 | -7.1 | 3.6 | 32.7 | 36.8 | -4.1 | 2.0 |
| mlnla-rl600-all11 | unresolved | 80.4 | 82.1 | -1.8 | 0.9 | 19.9 | 20.5 | -0.6 | 0.3 |
| nla-sft-L42 | hallucination | 73.8 | 81.0 | -7.1 | 3.6 | 31.7 | 32.5 | -0.9 | 0.4 |
| nla-sft-L42 | false_claim | 42.9 | 54.8 | -11.9 | 6.0 | 17.4 | 18.4 | -1.0 | 0.5 |
| nla-sft-L42 | detail | 50.0 | 50.0 | 0.0 | 0.0 | 14.3 | 14.2 | 0.2 | 0.1 |
| nla-sft-L42 | probe_candidate | 16.7 | 11.9 | 4.8 | 2.4 | 4.6 | 3.5 | 1.1 | 0.6 |
| nla-sft-L42 | correct | 64.3 | 76.2 | -11.9 | 6.0 | 25.0 | 26.9 | -1.9 | 1.0 |
| nla-sft-L42 | unresolved | 54.8 | 47.6 | 7.1 | 3.6 | 10.7 | 9.5 | 1.2 | 0.6 |
| patchscopes | hallucination | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| patchscopes | false_claim | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| patchscopes | detail | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| patchscopes | probe_candidate | 0.0 | 5.0 | -5.0 | 2.5 | 0.0 | 5.0 | -5.0 | 2.5 |
| patchscopes | correct | 5.0 | 5.0 | 0.0 | 0.0 | 5.0 | 5.0 | 0.0 | 0.0 |
| patchscopes | unresolved | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| rlens | hallucination | 5.0 | 5.0 | 0.0 | 0.0 | 5.0 | 3.9 | 1.1 | 0.6 |
| rlens | false_claim | 5.0 | 5.0 | 0.0 | 0.0 | 5.0 | 3.9 | 1.1 | 0.6 |
| rlens | detail | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| rlens | probe_candidate | 5.0 | 5.0 | 0.0 | 0.0 | 2.5 | 2.5 | 0.0 | 0.0 |
| rlens | correct | 85.0 | 90.0 | -5.0 | 2.5 | 56.2 | 62.3 | -6.1 | 3.1 |
| rlens | unresolved | 40.0 | 40.0 | 0.0 | 0.0 | 25.5 | 23.5 | 2.0 | 1.0 |
| s3d-rl600-all11 | hallucination | 16.7 | 26.7 | -10.0 | 5.0 | 5.2 | 10.5 | -5.3 | 2.7 |
| s3d-rl600-all11 | false_claim | 10.0 | 8.3 | 1.7 | 0.8 | 2.8 | 2.9 | -0.1 | 0.1 |
| s3d-rl600-all11 | detail | 6.7 | 18.3 | -11.7 | 5.8 | 2.4 | 7.6 | -5.2 | 2.6 |
| s3d-rl600-all11 | probe_candidate | 33.3 | 33.3 | 0.0 | 0.0 | 12.8 | 13.0 | -0.2 | 0.1 |
| s3d-rl600-all11 | correct | 63.3 | 60.0 | 3.3 | 1.7 | 27.0 | 24.0 | 3.0 | 1.5 |
| s3d-rl600-all11 | unresolved | 93.3 | 85.0 | 8.3 | 4.2 | 48.9 | 45.8 | 3.1 | 1.6 |
| s3d-sft251 | hallucination | 23.3 | 21.7 | 1.7 | 0.8 | 7.4 | 6.1 | 1.3 | 0.6 |
| s3d-sft251 | false_claim | 16.7 | 13.3 | 3.3 | 1.7 | 4.6 | 2.7 | 1.9 | 0.9 |
| s3d-sft251 | detail | 8.3 | 10.0 | -1.7 | 0.8 | 2.7 | 3.3 | -0.6 | 0.3 |
| s3d-sft251 | probe_candidate | 6.7 | 5.0 | 1.7 | 0.8 | 2.4 | 2.1 | 0.3 | 0.1 |
| s3d-sft251 | correct | 76.7 | 76.7 | 0.0 | 0.0 | 37.8 | 36.2 | 1.6 | 0.8 |
| s3d-sft251 | unresolved | 81.7 | 80.0 | 1.7 | 0.8 | 41.7 | 45.2 | -3.6 | 1.8 |
| sae-l42 | hallucination | 35.0 | 25.0 | 10.0 | 5.0 | 26.1 | 20.8 | 5.3 | 2.7 |
| sae-l42 | false_claim | 35.0 | 25.0 | 10.0 | 5.0 | 26.1 | 20.8 | 5.3 | 2.7 |
| sae-l42 | detail | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| sae-l42 | probe_candidate | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| sae-l42 | correct | 80.0 | 75.0 | 5.0 | 2.5 | 63.4 | 66.8 | -3.5 | 1.7 |
| sae-l42 | unresolved | 20.0 | 20.0 | 0.0 | 0.0 | 10.5 | 12.3 | -1.8 | 0.9 |
| template-v3 | hallucination | 15.8 | 26.3 | -10.5 | 5.3 | 8.6 | 13.8 | -5.2 | 2.6 |
| template-v3 | false_claim | 15.8 | 26.3 | -10.5 | 5.3 | 8.6 | 13.8 | -5.2 | 2.6 |
| template-v3 | detail | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| template-v3 | probe_candidate | 15.8 | 10.5 | 5.3 | 2.6 | 12.4 | 7.2 | 5.3 | 2.6 |
| template-v3 | correct | 47.4 | 52.6 | -5.3 | 2.6 | 31.8 | 36.9 | -5.1 | 2.5 |
| template-v3 | unresolved | 31.6 | 36.8 | -5.3 | 2.6 | 17.6 | 20.5 | -2.9 | 1.5 |
| tuned | hallucination | 30.0 | 15.0 | 15.0 | 7.5 | 15.1 | 5.7 | 9.3 | 4.7 |
| tuned | false_claim | 30.0 | 15.0 | 15.0 | 7.5 | 15.1 | 5.7 | 9.3 | 4.7 |
| tuned | detail | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| tuned | probe_candidate | 0.0 | 5.0 | -5.0 | 2.5 | 0.0 | 2.4 | -2.4 | 1.2 |
| tuned | correct | 65.0 | 65.0 | 0.0 | 0.0 | 39.0 | 39.1 | -0.1 | 0.0 |
| tuned | unresolved | 55.0 | 55.0 | 0.0 | 0.0 | 31.8 | 36.6 | -4.8 | 2.4 |

### Per-leaf self-agreement (Dice of the two runs)

| leaf | r0 | r1 | both | self-agreement |
|---|---|---|---|---|
| non_claim | 1558 | 1561 | 1533 | 0.983 |
| correct | 560 | 564 | 500 | 0.890 |
| detail | 143 | 155 | 107 | 0.718 |
| false_claim | 141 | 147 | 106 | 0.736 |
| substitution_sibling | 53 | 43 | 40 | 0.833 |
| substitution_category | 18 | 25 | 13 | 0.605 |
| model_state | 1 | 0 | 0 | 0.000 |
| unverifiable | 520 | 510 | 423 | 0.821 |
| supplementary_true | 26 | 20 | 17 | 0.739 |
| supplementary_false | 8 | 6 | 6 | 0.857 |
| ambiguous_topic | 3 | 2 | 1 | 0.400 |
| off_topic | 27 | 25 | 23 | 0.885 |

### Flagged pairs (swapped in > 10% of either leaf's units)

| pair | swaps | share of A | share of B |
|---|---|---|---|
| correct ↔ unverifiable | 68 | 0.109 | 0.112 |
| false_claim ↔ unverifiable | 37 | 0.203 | 0.061 |
| detail ↔ unverifiable | 33 | 0.173 | 0.054 |
| correct ↔ detail | 28 | 0.045 | 0.147 |
| detail ↔ false_claim | 20 | 0.105 | 0.110 |
| supplementary_true ↔ unverifiable | 10 | 0.345 | 0.016 |
| false_claim ↔ substitution_category | 7 | 0.038 | 0.233 |
| substitution_category ↔ substitution_sibling | 7 | 0.233 | 0.125 |
| non_claim ↔ off_topic | 3 | 0.002 | 0.103 |
| ambiguous_topic ↔ unverifiable | 2 | 0.500 | 0.003 |
| model_state ↔ unverifiable | 1 | 1.000 | 0.002 |
| ambiguous_topic ↔ off_topic | 1 | 0.250 | 0.034 |
| supplementary_false ↔ unverifiable | 1 | 0.125 | 0.002 |
| supplementary_false ↔ supplementary_true | 1 | 0.125 | 0.034 |

### Top swaps

| pair | swaps | share of A | share of B |
|---|---|---|---|
| correct ↔ unverifiable | 68 | 0.109 | 0.112 |
| false_claim ↔ unverifiable | 37 | 0.203 | 0.061 |
| detail ↔ unverifiable | 33 | 0.173 | 0.054 |
| correct ↔ detail | 28 | 0.045 | 0.147 |
| non_claim ↔ unverifiable | 25 | 0.016 | 0.041 |
| correct ↔ non_claim | 21 | 0.034 | 0.013 |
| detail ↔ false_claim | 20 | 0.105 | 0.110 |
| supplementary_true ↔ unverifiable | 10 | 0.345 | 0.016 |
| correct ↔ false_claim | 7 | 0.011 | 0.038 |
| false_claim ↔ substitution_category | 7 | 0.038 | 0.233 |

### Between-run confusion (rows r0, columns r1)

| r0 \ r1 | non_claim | correct | detail | false_claim | substitution_sibling | substitution_category | model_state | unverifiable | supplementary_true | supplementary_false | ambiguous_topic | off_topic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non_claim | 1533 | 12 | 0 | 1 | 0 | 0 | 0 | 10 | 1 | 0 | 0 | 1 |
| correct | 9 | 500 | 8 | 3 | 0 | 0 | 0 | 40 | 0 | 0 | 0 | 0 |
| detail | 0 | 20 | 107 | 8 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 |
| false_claim | 2 | 4 | 12 | 106 | 1 | 3 | 0 | 13 | 0 | 0 | 0 | 0 |
| substitution_sibling | 0 | 0 | 3 | 1 | 40 | 7 | 0 | 2 | 0 | 0 | 0 | 0 |
| substitution_category | 0 | 0 | 0 | 4 | 0 | 13 | 0 | 1 | 0 | 0 | 0 | 0 |
| model_state | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| unverifiable | 15 | 28 | 25 | 24 | 2 | 1 | 0 | 423 | 1 | 0 | 0 | 1 |
| supplementary_true | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 17 | 0 | 0 | 0 |
| supplementary_false | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 6 | 0 | 0 |
| ambiguous_topic | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 |
| off_topic | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 23 |
