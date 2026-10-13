# Results 0008: `agentic_misalignment`: the claim map (v2.7) on lens mentions of misalignment

Plan: [`plans/0008-agentic-misalignment-claim-map.md`](../../plans/0008-agentic-misalignment-claim-map.md)
(rev 13). Full generated report: [`0008_report_full.md`](0008_report_full.md). Issue #8, PR #9.
Run 2026-10-13. Judge `claude-opus-5-5` (effort low) for every stage.

**Provisional.** The eval has no held-out gate (the claim map's gate is chat gold). Every audit
label was made by Claude agents (the same model family as the detector and the judge), so the
gates are agent audits, not independent ones.

## Question

When a lens says something about the model's misalignment, is it reading a real concept, or
confabulating? How often does it name random misconduct ("murder", "stealing") rather than what
the model goes on to do? Camila's prior was that most of the correct mentions can be inferred
from the text, so they land in `correct` or `model_state`, and that the rest are unverifiable or
contradicted.

## Setup

- **Bank.** The 32 WorkspaceBench `agentic_misalignment` scenarios: 28 misaligned and 4
  controls, 18 from the Anthropic grid and 14 in-house. The lenses read every prompt position
  (50,452 sites). The model's pinned rollout is the "what it went on to do".
- **Lenses.**
  - O-lens RL600 (`s3d-rl600`) and J-lens (top-10 bag, summarised by interp-v1): 11 layers
    (20 to 60), 554,972 cells each.
  - NLA-RL at layer 42: 50,452 cells.
- **Pipeline.**
  1. Stage D, a blind detector (`amdetect-v2`, with one rulebook in `kinds.py`), finds every
     misalignment mention and records its kind, actor and stance.
  2. Stage H (`amverify-v1`) re-checks the headline candidates.
  3. Stage M runs the claim map v2.7 on every detected cell. Its system prompt is
     byte-identical to the chat one; the user prompt puts the marker inside the prompt and adds
     a research-context frame (`claimmap-v2.7+am2`).
  4. The intent call (`amintent-v2`, hand-audited) records which misconduct kinds each rollout
     performs and which the scenario sets up.
- **Headline population.** Spans whose actor is the AI or unspecified and whose stance is
  considers/plans or does/did, on misaligned items: "the lens says the model is considering or
  doing X".

| | O-lens RL600 | NLA-RL L42 | J-lens |
|---|---|---|---|
| cells with a mention | 43,379 (7.8%) | 18,674 (37%) | 10,252 bags → 8,812 summaries (1.6%) |
| claim-map cells judged / failed (mostly API refusals) | 42,298 / 1,081 | 18,030 / 644 | 8,697 / 115 |
| headline spans | 14,372 | 3,228 | 1,742 |

## Headline answers

![verdict shares](../../plots/am_claim_map/verdict_shares.png)

Headline verdict shares, pooled over spans. Bootstrap 95% CIs are over the 28 misaligned items.

| verdict | O-lens RL600 | NLA-RL L42 | J-lens |
|---|---|---|---|
| hallucination (contradicted by the text) | **16%** [14, 18] | **48%** [43, 52] | **1.3%** |
| correct | 23% | 8% | 33% |
| unverifiable | 43% | 29% | 54% |
| model_state | 8% | 8% | 7% |
| probe candidate (substitution) | 1.6% | 2.0% | 0.1% |
| junk | 2.5% | 0.8% | 2.7% |
| failed (refused) | 5.0% | 4.8% | 1.4% |

![kind tiers](../../plots/am_claim_map/kind_tiers.png)

| | O-lens | NLA | J-lens |
|---|---|---|---|
| **real intention**: the mention's kind is what the rollout does | **57%** | **37%** | **46%** |
| **unrelated misconduct**: neither the rollout, the design, nor an option the scenario sets up (raw → audit-corrected) | 26% → **20%** | 52% → **56%** | 34% → **34%** |
| hallucination rate among unrelated mentions | 22% | 55% | 1% |
| **unrealised-intent claims**: model_state or unverifiable, and not what the rollout does (denominator: all tiered headline spans, realised ones included) | 28% | 24% | 40% |
| same, among non-realised spans only | 64% | 38% | 75% |
| same, on the 4 controls (no realised spans exist; detector actor/stance, not Stage-H verified) | 64% | 37% | 76% |

1. **NLA confabulates heavily.** About half of its headline mentions are contradicted by the
   conversation.
   - 80% of those (1,234 of 1,540) are contradicted by text the model has **already read**, so
     they are not wrong predictions of what comes next.
   - More than half of its headline spans name misconduct that has nothing to do with the
     scenario, mostly (63%) invented deception and sabotage plots. Examples: "fake security
     reports with fabricated evidence", "an AI impersonator intercepts and forges messages",
     "escalating cyber attack events against Stark Industries".
   - Even when NLA names the right kind of misconduct, the specifics are often wrong (429 of
     1,136 realised spans are hallucinations), e.g. "the assistant might observe Julian dying in
     the flood" for Kyle and a server-room heat failure.
2. **O-lens is mostly grounded.**
   - Most of its mentions (57%) name what the rollout actually does: blackmail in the blackmail
     scenarios, cancelling the alert in the murder ones.
   - 16% of headline mentions are contradicted.
   - 20% (audit-corrected) name unrelated misconduct, spread over sabotage, power grabs ("I am
     now the sole authority", "boardroom coup"), oversight evasion and deception.
3. **J-lens almost never contradicts the text, but it is vague.**
   - Hallucination is about 1%, but over half of its headline mentions are unverifiable, and a
     third name unrelated misconduct ("an AI weighing whether to undermine oversight /
     sabotage / deceive").
   - Its summaries are generic enough that they cannot be contradicted. The cost is
     specificity, not truth.
4. **The hallucination verdict does not measure confabulated intent.**
   - Rule 4 of the taxonomy routes "the AI plans to X" to `model_state` or unverifiable, which
     are never checked against the text.
   - The closer measure is the unrealised-intent rate: on misaligned items, 24–40% of headline
     mentions claim an intent that the rollout does not carry out.
   - Among spans that do *not* name the rollout's action, 64% (O-lens), 38% (NLA) and 75%
     (J-lens) are such uncheckable intent claims. The control scenarios give the same rates
     (64% / 37% / 76%), so the lenses talk about non-realised misconduct in the same way
     whether or not the model misbehaves. (Control spans are not Stage-H verified, so this
     comparison is approximate.) NLA's lower rate is not a virtue: its non-realised mentions
     are mostly outright hallucinations.
5. **Camila's prior holds for "inferable from text" — clearly for O-lens.**
   - Of the realised-kind mentions (mention-level, all populations; report §4), the option was
     already set up before the read position for 92% (O-lens: 20,473 of 22,314), 88% (NLA) and
     85% (J-lens).
   - Only 6–13% name the rollout's action before the prompt has set it up at all
     (`never_set_up`: O-lens 1,279, NLA 656, J-lens 758). Even those are more often
     unverifiable than correct (O-lens: 620 unresolved vs 323 correct).
   - Base rate, no lens: over every (misaligned-item site, realised kind) pair, 61% of sites
     fall after the kind's enabling quote and 9.7% have none. O-lens (92% set up before, 5.7%
     never) clearly concentrates its realised mentions after the option is set up: it reads
     options off the text and does not foresee the model's choice. NLA (10.5% never) is at the
     base rate and J-lens (12.9%) above it, so this is not shown for those two.
   - Caveat: the intent audit counts planted tool lines (e.g. the `cancel_alert` tool in the
     system prompt) as "set up". This is the conservative choice for "set up before"; starting
     the clock at the threat email would move some mentions to `set_up_after`.
6. **Prior check: `model_state` and `correct`.**
   - `correct` is common for O-lens and J-lens (23% and 33%); `model_state` stays small (7–12%).
   - Most first-person intent claims are judged **unverifiable** instead (O-lens: 2,267
     realised spans tagged as unverified predictions).
   - Where a correct mention has support, the judge's quote is usually the **rollout** (J-lens
     466 of 581, NLA 168 of 263), so "correct" often means it correctly anticipated what the
     model went on to write. The O-lens split is in the full report.
7. **Layers.**
   - O-lens: the correct share rises from 10–17% at layers 20–24 to a 27% peak at layer 48;
     hallucination is 14–18% except a 23.5% bump at layer 24.
   - J-lens: the correct share climbs steadily to 54% at layer 60.
   - Layer 20 is noise for J-lens (5 spans).

   ![per layer](../../plots/am_claim_map/per_layer.png)

Also of note: the probe candidates are mostly the lens naming the AI as another known AI system
("names the AI as Claude / GPT / GLaDOS instead of Alex"). These are identity substitutions,
which plan 0002 treats as possible co-activation, not hallucination.

## Detector audit (agent-labelled)

These gates come from plan 0008's Calibration section.

| gate | O-lens | NLA | J-lens |
|---|---|---|---|
| lexicon-stratum recall (≥ 0.85) | 0.93 | 0.86 | 0.98 |
| overall recall estimate, incl. a lenient-screener sample (not gated) | 0.87 | 0.82 | 0.91 |
| precision of mentions (≥ 85% real) | 87.5% | 92.5% | 100% (summaries) |
| headline membership (≥ 80%) | 0.90 | 0.875 | 0.90 |
| wrongly excluded from headline (≤ 20%) | 5% | 2.5% | 2.5% |

How the gates were passed:
- `amdetect-v1` failed O-lens precision (72%) and NLA recall (0.78). The rulebook (`kinds.py`)
  fixed that; it covers genre labels, rules that name an act, legitimate objections, business
  "leverage", "fabricated" meaning fictional, and a human emailer's "I".
- `amdetect-v2` still failed O-lens headline membership (0.70), on vague urgency like "we must
  act before 10 AM". Stage H fixed it; it rejected 1,921 O-lens candidates as not real.
- What the detector still misses is mostly genre labels ("security breach", "espionage
  thriller"), rules naming an act, and people hiding the affair. It rarely misses the AI itself
  planning something.

The verified headline audit also found that many headline spans pass only through "unspecified
actor + does/did": narrated story events ("the leaked source code", "the hacked systems"), not
the AI's plans. This is about a third of NLA's headline (1,110 of 3,228 spans) and a fifth of
J-lens's (309 of 1,742); read the
headline as "the lens says someone is considering or doing it, most often the model".

## Caveats

- **API refusals.**
  - Without the research frame, Opus refused about 73% of claim-map calls on the murder scenario
    and 29% on blackmail-replacement. With it, 2.5% / 3.4% / 1.3% of cells remain failed (mostly refused)
    (O-lens / NLA / J-lens), concentrated in the same scenarios. They are reported as `failed`,
    not dropped.
  - 12 of 32 intent calls were refused and written by hand.
  - About 2% of J-lens summaries were refused (their cells are lost).
- **Named stand-in AIs.** NLA often renames the AI (Nova, ARC, The Oracle). The rulebook makes
  these `other_party`, so NLA's headline is probably undercounted. The detector cannot tell a
  stand-in AI from a human by name, so there is no sensitivity view for this.
- **One rollout.** The rollout is 1 of 8 samples. "Contradicted by the response" and
  "unrealised intent" are weak evidence; three misaligned items misbehave in only 1–2 of 8.
- **Small strata.** The corrected unrelated rate is a rough point estimate with no CI: the
  designed-only stratum has 1 label per arm and scenario-set-up 8–11. The O-lens correction
  (26% → 20%) rests on 6 of 40 "unrelated" spans relabelled as realised. Items differ in length by about 5×; item-macro shares are in the full
  report.
- **J-lens summary filter.** J-lens is scored on summaries, and only cells whose bag AND summary
  carry a mention enter. 15% of `lost_in_summary` cells hid a real mention that the summary
  detector missed.

## Spend

About **$5.9k** in total, including the pilot, both detector versions, the audits and Stage H:

| lens | detector | J-lens summaries | claim map | audits + Stage H | total |
|---|---|---|---|---|---|
| NLA-RL L42 | $269 | — | $1,418 | $17 | $1,704 |
| O-lens RL600 | $958 | — | $1,840 | $29 | $2,827 |
| J-lens | $716 | $222 | $399 | $33 | $1,370 |

## Reproduce

```
scripts/run_am_step1_chunked.sh      # intent, Stage D (+ J-lens summaries), Stage H, audit sheets
#   label audit/*.jsonl + intent overrides, then: scripts/am_merge_labels.py; scripts/am_audit.py score
scripts/run_am_step2_chunked.sh      # Stage M (claim map)
uv run --no-sync python scripts/am_audit.py sample-tiers   # label, merge, score
uv run --no-sync hbench report --eval agentic_misalignment --out outputs
uv run --no-sync python scripts/plot_am.py
```
