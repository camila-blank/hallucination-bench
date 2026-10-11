# Results 0003: are same-category sibling mentions hallucinations? (DiffMean probe check)

Plan `plans/0003-sibling-concept-probes.md` (rev 10). Issue #5, PR #6. Run 2026-10-11 (probe run
`sibling-v2`, after review fixes). Judge: claim map `claimmap-v2.5` (plan 0002 rev 15),
`claude-opus-5-5`, effort low. The full machine report is `outputs/sibling/score/report.md`;
regenerate it with `hbench sibling report`.

## Question

Sometimes a lens names several instances of one category where the context implies only one:
the user is Korean and the readout says "Korea … China", or the user is embarrassed and the
readout says "angry". Are the extra **siblings** hallucinations, or does the model actually
represent them? We give each concept one DiffMean probe and test whether the sibling's probe
fires at the activations where the lens names it.

## Answer in one paragraph

**Mostly hallucinations, with a small real signal.** The pre-registered verdict is *supported*
for the s3d O-lens and for NLA in every subset, and the J-lens positive control passes. Where
the lens names a sibling, that sibling's probe scores higher than at the same item's other
cells (β₁ = 0.27 sd for s3d), and higher than the unmentioned siblings at the same activation
(β₂ = 0.35 sd). The effect is small, though:

- a mentioned sibling's probe fires 14% of the time;
- a matched unmentioned sibling's fires 9% of the time;
- the gold concept's fires 72% of the time at cells that name it.

So **86% of s3d sibling mentions are not backed by a probe-detectable representation**, and
count as hallucinations by Camila's definition. The rate is 82% for NLA and 72% for J-lens (the
positive control, which reads the same residual linearly). Close siblings (resentful → angry,
Osaka → Kyoto, New York → Boston) do fire the lenient threshold 50–80% of the time. But under
the strict threshold, which asks whether the probe fires more than it does in an ordinary
gold-concept context, they drop to about 20%. So their detection mostly reflects concepts that
overlap with the gold everywhere, not something specific to where the lens names them.

![probe-verified hallucination rate](../../plots/sibling_probe/halluc_rate.png)

## Headline numbers

The **probe-verified hallucination rate** is the share of sibling mentions (judged not correct)
whose probe does not fire. Two thresholds are used:

- **τ_cat** (headline): the 95th percentile of the sibling's probe over the category's other
  concepts.
- **τ(c | g)** (strict): the 95th percentile over ordinary contexts of the item's gold concept.

Only probes passing both the validity gate and the eval-site transfer gate count. CIs are 95%,
from an item-clustered bootstrap.

| arm | M pairs | halluc rate, τ_cat | strict, τ(c\|g) | all sibling mentions (judge ignored) | det(M) | det(unmentioned, matched) | gold ceiling |
|---|---|---|---|---|---|---|---|
| **s3d O-lens** | 2,767 | **86.0%** [83.1, 88.7] | 86.3% [84.5, 88.1] | 84.4% | 14.0% | 8.6% | 71.7% |
| **NLA-RL L42** | 626 | **82.1%** [78.6, 85.6] | 82.5% [79.3, 85.6] | 78.2% | 17.9% | 12.7% | 71.2% |
| **J-lens** (control) | 176 | **72.0%** [64.8, 78.9] | 78.8% [73.3, 83.6] | 68.2% | 28.0% | 15.8% | 92.4% |

**"Not detected" does not mean "certainly absent".** The probes miss the gold itself 28% of the
time at gold-mention cells (ceiling 72%), so some of the 86% is probe miss. A
sensitivity-corrected view is (det(M) − floor) / (ceiling − floor):

- s3d: **0.085** [0.056, 0.117]
- NLA: 0.089 [0.041, 0.137]
- J-lens: 0.16 [0.08, 0.24]

In words, under 10% of O-lens and NLA sibling mentions carry gold-strength evidence beyond the
baseline. The pre-registered "strongly supported" bar (> 0.5) is not met.

### Verdict tests (pre-registered)

- **β₁, matched test:** the same item, layer and concept, comparing cells that name the concept
  with cells that do not. It uses offset fixed effects and is measured in z units of the probe.
- **β₂, within-cell test:** at the same activation, named siblings vs unnamed siblings.

| arm | β₁ (z) | β₂ (z) | verdict |
|---|---|---|---|
| s3d O-lens | 0.27 [0.21, 0.32] | 0.35 [0.25, 0.46] | supported |
| s3d, probes with validity ≥ 0.85 (robustness) | 0.36 [0.27, 0.45] | 0.40 [0.22, 0.57] | supported |
| NLA-RL L42 | 0.32 [0.23, 0.41] | 0.58 [0.46, 0.70] | supported |
| J-lens (positive control) | 0.61 [0.44, 0.80] | 1.21 [1.04, 1.38] | control passes |

The within-item AUROC (mention cells vs non-mention cells of the same item, layer and concept)
is 0.57 for s3d, 0.60 for NLA and 0.66 for J-lens.

## By category (s3d O-lens)

![detection by mention status](../../plots/sibling_probe/detection_s3d-rl600.png)

| category | halluc τ_cat | halluc τ(c\|g) | probe pass share | status |
|---|---|---|---|---|
| emotion | **73%** | 80% | 100% | adjudicable; lowest rate |
| capital_city | 83% | 87% | 94% | adjudicable |
| holiday | 91% | 89% | 98% | adjudicable |
| country | 92% | 90% | 72% | adjudicable |
| occupation | 94% | 90% | 86% | adjudicable |
| farm_animal | (92%) | (83%) | 23% | **probes too weak to adjudicate** |
| tree | (85%) | (81%) | 39% | **probes too weak to adjudicate** |

Over adjudicable categories only, the macro rate is s3d 86.6%, NLA 84.2% and J-lens 62.0%.
Farm-animal and tree probes mostly fail the validity gate (out-of-fold AUROC about 0.65), so
those two rates rest on the few passing probes and are not interpreted.

## Most frequent sibling pairs (s3d, gate-passing probes)

| gold → sibling | mentions | items | det τ_cat | det τ(c\|g) | mean z(c\|g) |
|---|---|---|---|---|---|
| resentful → angry | 43 | 2 | 0.82 | 0.17 | 0.83 |
| fisherman → musician | 43 | 5 | 0.04 | 0.28 | 1.08 |
| Osaka → Tokyo | 32 | 2 | 0.08 | 0.02 | −0.21 |
| Ireland → UK | 28 | 2 | 0.15 | 0.08 | 0.40 |
| Osaka → Kyoto | 26 | 2 | 0.52 | 0.21 | 0.76 |
| jealous → angry | 26 | 2 | 0.14 | 0.24 | 0.95 |
| locksmith → musician | 26 | 5 | 0.04 | 0.21 | 0.75 |
| artist → musician | 25 | 2 | 0.03 | 0.06 | 0.18 |
| fisherman → writer | 25 | 5 | 0.00 | 0.01 | −0.30 |
| New York → Boston | 23 | 2 | 0.59 | 0.20 | 0.95 |

Most pairs come from 2–5 items, so per-pair rates are anecdotal. The pattern across them:

- **High τ_cat, low τ(c | g)** (resentful → angry): the sibling is near the gold, and its probe
  fires in any gold context.
- **Low τ_cat, positive z** (fisherman → musician): the probe is above its gold-context level
  but far below detection.

## Caveats

- **Probe validity.** Out-of-fold, position-level AUROC passed the original 0.85 gate for only
  a third of (concept, layer) probes. Plan rev 10 lowered the gate to 0.70 before any judge
  result existed; 382 of 510 now pass both gates. This drops 22% of s3d M mentions. The
  0.85-only robustness row gives the same verdict.
- **Inventory coverage.** Rates cover the 10–16 inventory concepts per category. Estimated
  sibling mentions outside the inventory are low (0–8%) for s3d capital_city, country, emotion,
  farm_animal and holiday, and higher for tree (35%) and occupation (55%, small sample).
- **Scope.** Many NLA sibling mentions sit inside reconstructed or imagined prompts. On the
  `scope = user` slice the rates are similar: s3d 83%, NLA 81%.
- **Review fixes before this version.** The occupation alias "author" no longer matches generic
  uses ("the author of", "author bio"); this removed 25 s3d M pairs. Golds named outright in
  the prompt (5 pilot items) no longer count toward the ceiling or transfer. Folds are
  stratified by explicit / implied text.
- **Failed cells.** 85 judged cells (0.8%) still failed after 4 attempts, mostly refusals, and
  are excluded.
- **Setup.** The items are user-modelling prompts with an implied attribute, read from the
  request sentence through the template tail at L28/36/44/52 (NLA at L42). Responses are
  Qwen3.6-27B greedy.

## Pilot (manual check of how common sibling mentions are)

We wrote 40 prompts by hand (8 per new category) and read them with s3d at 4 layers. Every
category had sibling mentions in 8/8 items (21–37% of cells under a rough matcher), and a hand
read confirmed the phenomenon:

- Berlin landmarks → "central London";
- Tokyo → "Tokyo & Washington D.C.";
- a maple-syrup tree → "likely a birch or oak";
- goats on the roof → "cows on the roof";
- Easter → a Thanksgiving / Halloween invitation;
- Diwali → a Rosh Hashanah invitation.

Country and occupation showed the same pattern in the wsbench readouts (Ukraine → Russia,
New Zealand → Australia, fisherman → farmer / sailor).

## Further settings worth trying

- **Other categories:** languages (a French speaker read as Spanish or Italian), cuisines,
  sports and teams, musical instruments, dog breeds, car brands, religions and denominations,
  universities, programming languages, and historical figures of the same era or role (the
  Picasso / da Vinci substitution).
- **Better probes before adjudicating farm animals and trees:** more and longer probe texts, or
  token-position-specific probes, since their DiffMean probes are near chance across the read
  window.
- **A direct test of "the model represents both":** steer or ablate along the sibling direction
  at mention cells and check whether the lens mention moves.

## Cost

| item | cost |
|---|---|
| Judge (claim map, scope and coverage calls, J-lens summaries; 3 arms) | $356 |
| Item and probe-text generation | $10 |
| Modal (captures; s3d, J-lens and NLA readouts; 2 probe jobs; Qwen generation) | about $30–40 |
| **Total** | **about $400** |
