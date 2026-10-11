# Results 0003: are same-category sibling mentions hallucinations? (DiffMean probe check)

Plan `plans/0003-sibling-concept-probes.md` (rev 10). Issue #5, PR #6. Run 2026-10-11.
Judge: claim map `claimmap-v2.5` (plan 0002 rev 15), `claude-opus-5-5`, effort low.
Full machine report: `outputs/sibling/score/report.md` (regenerate with `hbench sibling report`).

## Question

When a lens names several instances of one category where the context implies only one
(the user is Korean and the readout says "Korea … China"; the user is embarrassed and the
readout says "angry"), are the extra **siblings** hallucinations, or does the model actually
represent them? Each concept gets one DiffMean probe, and we test whether the sibling's probe
fires where the lens names it.

## Answer in one paragraph

**Partly.** Sibling mentions do carry a real probe signal: the pre-registered verdict is
*supported* for the s3d O-lens and for NLA, in every subset, with the J-lens positive control
passing. The effect is small, though. In the s3d O-lens, a mentioned sibling's probe fires 14%
of the time. An unmentioned sibling at the same item and layer fires 9% of the time. The gold
concept fires 72% of the time at cells that mention it. So **about 86% of s3d sibling mentions
are not backed by a probe-detectable representation**, which makes them hallucinations by the
definition Camila set. The rate is 82% for NLA and 72% for J-lens (the positive control, which
reads the same residual linearly). The exceptions are close siblings: near-synonyms
(resentful → angry, 81% detected) and closely associated places and animals (Osaka → Kyoto 60%,
New York → Boston 61%, goose → duck 57%). There, the sibling really is in the activation.
Distant substitutions are almost never detected (fisherman → musician/writer 0–4%,
turkey → chicken 2%, Osaka → Tokyo 7%).

![probe-verified hallucination rate](../../plots/sibling_probe/halluc_rate.png)

## Headline numbers

The probe-verified hallucination rate is the share of sibling mentions (judged not correct)
whose probe does not fire. The headline threshold is τ_cat. The strict threshold is τ(c | g),
the 95th percentile of the sibling's probe on ordinary contexts of the item's gold concept.
95% CIs are from an item-clustered bootstrap.

| arm | sibling mentions (M) | halluc rate, τ_cat | strict, τ(c\|g) | all sibling mentions (judge ignored) | det(M) | det(unmentioned, matched) | gold ceiling |
|---|---|---|---|---|---|---|---|
| **s3d O-lens** | 2,629 | **85.8%** [82.7, 88.6] | 86.7% [85.0, 88.6] | 84.1% | 14.2% | 8.7% | 71.9% |
| **NLA-RL L42** | 564 | **82.4%** [78.7, 86.1] | 84.5% [81.5, 87.5] | 78.0% | 17.6% | 13.0% | 71.8% |
| **J-lens** (control) | 165 | **72.0%** [64.3, 79.3] | 78.6% [72.3, 84.1] | 68.4% | 28.0% | 16.3% | 93.2% |

**Not detected is not the same as certainly absent.** The probes miss the gold itself 28% of
the time at gold-mention cells (ceiling 72%), so part of the 86% is probe miss. A
sensitivity-corrected view is (det(M) − floor) / (ceiling − floor): **0.09** for s3d [0.06,
0.12] and 0.08 for NLA. In words, about 9% of sibling mentions carry gold-strength evidence
beyond baseline. The pre-registered "strongly supported" bar was > 0.5; it is not met.

### The verdict tests (pre-registered)

β₁ is the matched test: the same item, layer and concept, at cells that mention the concept vs
cells that do not, with offset fixed effects, on the z-scale of the probe. β₂ is the
within-cell test: at the same activation, mentioned siblings vs unmentioned siblings.

| arm | β₁ (z) | β₂ (z) | verdict |
|---|---|---|---|
| s3d O-lens | 0.25 [0.19, 0.31] | 0.28 [0.17, 0.40] | supported |
| s3d, probes with validity ≥ 0.85 (robustness) | 0.34 [0.26, 0.43] | 0.34 [0.17, 0.52] | supported |
| NLA-RL L42 | 0.27 [0.17, 0.36] | 0.52 [0.39, 0.64] | supported |
| J-lens (positive control) | 0.58 [0.39, 0.77] | 1.17 [0.98, 1.36] | control passes |

The within-item AUROC (mention cells vs non-mention cells, same item, layer and concept) is
0.57 for s3d, 0.58 for NLA and 0.65 for J-lens.

## By category (s3d O-lens)

![detection by mention status](../../plots/sibling_probe/detection_s3d-rl600.png)

| category | halluc rate τ_cat | gold ceiling | floor | reading |
|---|---|---|---|---|
| emotion | **73%** | 87% | 14% | the most "real" siblings (synonym clusters, e.g. resentful → angry) |
| capital_city | 83% | 64% | 11% | near cities detected (Osaka → Kyoto); far ones not |
| tree | 86% | 81% | 5% | maple → birch, the sap trees, are sometimes detected |
| holiday | 90% | 83% | 7% | |
| country | 91% | 51% | 9% | Ukraine → Russia and Korea → China are detected only at some cells |
| occupation | 94% | 62% | 4% | fisherman → musician / writer: not detected |
| farm_animal | 98% | 80% | 5% | turkey / donkey → chicken / goat: not detected |

## Caveats

- **Probe validity.** Out-of-fold, position-level AUROC passed the original 0.85 gate for only
  165 of 510 (concept, layer) probes. Plan rev 10 lowered the gate to 0.70 before any judge
  result existed; 364 of 510 pass both gates. This dropped 26% of s3d M mentions. The 0.85-only
  robustness row gives the same verdict.
- **Inventory coverage.** Rates cover the 10–16 inventory concepts per category. Estimated
  sibling mentions outside the inventory: low for capital_city, country, farm_animal and
  emotion (0–8%); higher for tree (35%) and occupation (55%, a small sample).
- **Scope.** Many NLA sibling mentions sit inside reconstructed or imagined prompts. On the
  `scope = user` slice the rates are similar: s3d 83%, NLA 81%.
- 85 judged cells (0.8%) still failed after 4 attempts, mostly refusals, and are excluded.
- The items are user-modelling prompts with an implied attribute, read from the request
  sentence through the template tail at L28/36/44/52 (NLA L42). Responses are Qwen3.6-27B greedy.

## Pilot (manual check of how common sibling mentions are)

There were 40 hand-written prompts (8 per new category), read with s3d at 4 layers. Every
category had sibling mentions in 8/8 items (21–37% of cells under a rough matcher). A hand read
confirmed the phenomenon:

- Berlin landmarks → "central London";
- Tokyo → "Tokyo & Washington D.C.";
- a maple-syrup tree → "likely a birch or oak";
- goats on the roof → "cows on the roof";
- Easter → a Thanksgiving / Halloween invitation;
- Diwali → a Rosh Hashanah invitation.

Country and occupation showed the same pattern in the wsbench readouts (Ukraine → Russia,
New Zealand → Australia, fisherman → farmer / sailor).

## Further settings worth trying

Other categories:

- languages (a French speaker read as Spanish or Italian);
- cuisines or dishes;
- sports and teams;
- musical instruments;
- dog breeds;
- car brands;
- religions and denominations;
- universities;
- programming languages;
- chemical elements;
- planets;
- historical figures of the same era or role (the Picasso / da Vinci substitution).

The results suggest where to look: categories with graded similarity (cities, synonym-rich
emotions) carry real sibling representation, while categorical swaps (occupations, farm
animals) are mostly lens confabulation.

## Cost

| item | cost |
|---|---|
| Judge (claim map, scope and coverage calls, J-lens summaries; 3 arms) | $356 |
| Item and probe-text generation | $10 |
| Modal (captures, s3d / J-lens / NLA readouts, probe job, Qwen generation) | about $30–40 |
| **Total** | **about $400** |
