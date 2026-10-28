# Results

One results doc per plan (`NNNN_<name>.md`); `*_report_full*.md` are the generated full reports
the results docs cite.

| plan | results | full report |
|---|---|---|
| [0001](../../plans/0001-bootstrap-wsbench-chat-taxonomy.md) wsbench_chat + taxonomy v1 | [0001_wsbench_chat_taxonomy.md](0001_wsbench_chat_taxonomy.md) | [0001_report_full.md](0001_report_full.md) |
| [0002](../../plans/0002-claim-map-taxonomy-v2.md) claim map (taxonomy v2) | [0002_claim_map.md](0002_claim_map.md), judge noise: [0002_judge_noise_v2.7.md](0002_judge_noise_v2.7.md) | [0002_report_full_v2.7.md](0002_report_full_v2.7.md) |
| [0006](../../plans/0006-user-modelling-country-ladder.md) user-modelling ladder, country | [0006_um_country_ladder.md](0006_um_country_ladder.md) | |
| [0007](../../plans/0007-story-eval.md) story eval | [0007_story.md](0007_story.md) | [0007_report_story_v2.7.md](0007_report_story_v2.7.md) |
| [0008](../../plans/0008-agentic-misalignment-claim-map.md) agentic misalignment | [0008_agentic_misalignment.md](0008_agentic_misalignment.md) | [0008_report_full.md](0008_report_full.md) |
| [0011](../../plans/0011-user-modelling-ladder-4-domains.md) user-modelling ladder, capital city + holiday | no results doc (figures: `scripts/sibling_probe/um_plot_macro*.py`; the 3-domain sibling score is in 0013) | |
| [0016](../../plans/0016-strongreject-eval.md) strongreject (failed jailbreaks), pilot | [0016_strongreject_pilot.md](0016_strongreject_pilot.md) | |
| [0013](../../plans/0013-final-sibling-eval-and-hallucination-index.md) hallucination index (hindex-v3) | [0013_hallucination_index.md](0013_hallucination_index.md) | |
| [0015](../../plans/0015-overrefusal-eval.md) overrefusal (OR-Bench) pilot | [0015_overrefusal_pilot.md](0015_overrefusal_pilot.md) | (generated: `outputs/overrefusal/claude-sonnet-5-5/report.md`) |
| [0017](../../plans/0017-mask-eval.md) mask (pilot) | [0017_mask_pilot.md](0017_mask_pilot.md) | [0017_report_full.md](0017_report_full.md) |
| [0015](../../plans/0015-overrefusal-eval.md)–[0017](../../plans/0017-mask-eval.md) safety evals, combined summary | [safety_evals_summary.md](safety_evals_summary.md) | |

Plans 0009/0010 (resampling sibling gate) are not on `main`; plan 0012 is the landing plan
(no results). Plans 0003–0005 (older sibling evals) were removed from the tree in plan 0013.
