# Results 0005: basic-readout mention ladder (plan 0005 rev 2)

Pipeline: `scripts/sibling_probe/ladder.py` (see CLAUDE.md, plan 0005 section). Figures:
`plots/sibling_probe/ladder_{labels_by_method,labels_by_layer,probe_validation}.png`. Probe quality
(per class x layer AUROC, recipe choice, capital pilot): `outputs/sibling/ladder/probe_quality.md`.

Key points:
- 1,231 mention units (35 entity items x 4 methods). The commitment judge (brcommit-v1) scored
  0.988 unit accuracy on the 80-unit blind gold set (Wilson95 0.933–0.998) and 0.95 on the 60 Ali
  anchors, so it is above the 85% target and the prompt was not revised.
- J-lens and NLA-RL L42 mentions are almost never committed (they are candidate lists or hedged
  text). Committed substitutions come from s3d O-lens and ML-NLA, and they cluster at L20–36.
- Probes: entity banks of up to 120 entities per class (synthesizer bank), trained on 3
  templates, with the knowledge filter applied (kept counts 10-81 per class and bank, see
  `probe_quality.md`). Held-out AUROC medians: logreg 0.998, raw DiffMean 0.982, whitened
  DiffMean 0.998. Every used (class, layer) cell passes 0.85 for logreg and whitened DiffMean;
  raw DiffMean fails 40 of 1,056 cells, and those cells become probe n/a.
- Fire pair: logreg + raw DiffMean, as pre-specified in the plan. Logreg + whitened DiffMean is a
  sensitivity row. Whitening removes the within-gold variation that the fire rule tests, and the
  whitened pair validates 5 + 0 units against 12 + 7 for the primary pair.
- Method note: an item's probe scores average the 5 fold models, while the reference scores
  are single out-of-fold scores; this is not leakage, but it may make the empirical p slightly
  conservative or liberal.
- Control-concept fire rate is at most 2/30 per bank x layer, so no layer gets the weak-null flag.
- Ali agrees with the case study:
  - track & field is probe-validated at L20–28 (both s3d and ML-NLA);
  - swimming is a hallucination at s3d L20–24 and cross-method allowed at L28 / ML-NLA L32, but
    only through lens-prior-flagged support;
  - basketball fires but is never committed;
  - wrestling and late-layer sports are not committed.

Judge `brcommit-v1`; fire pair logreg + dom (probe ranks shown as logreg/dom = share of same-gold held-out references below the item); labels: N not committed, P probe-validated, X cross-method allowed (primary +-1 grid step), H hallucination.

Judge vs blind gold (80 units): accuracy 0.988 (Wilson95 [0.933, 0.998]); committed P/R 0.958/1.0; Ali anchors (60): 0.95.

## Counts per method x label (primary rule)

| method | not committed | probe-validated | cross-method allowed | hallucination |
|---|---|---|---|---|
| s3d O-lens | 213 | 12 | 74 | 117 |
| ML-NLA RL600 | 85 | 7 | 30 | 26 |
| J-lens | 661 | 1 | 0 | 0 |
| NLA-RL L42 | 5 | 0 | 0 | 0 |

Sensitivity: cross-method at any layer

| method | not committed | probe-validated | cross-method allowed | hallucination |
|---|---|---|---|---|
| s3d O-lens | 213 | 12 | 108 | 83 |
| ML-NLA RL600 | 85 | 7 | 44 | 12 |
| J-lens | 661 | 1 | 0 | 0 |
| NLA-RL L42 | 5 | 0 | 0 | 0 |

Sensitivity: without lens-prior-flagged allowances

| method | not committed | probe-validated | cross-method allowed | hallucination |
|---|---|---|---|---|
| s3d O-lens | 213 | 12 | 25 | 166 |
| ML-NLA RL600 | 85 | 7 | 7 | 49 |
| J-lens | 661 | 1 | 0 | 0 |
| NLA-RL L42 | 5 | 0 | 0 | 0 |

Sensitivity: probe fires at +-1 grid step

| method | not committed | probe-validated | cross-method allowed | hallucination |
|---|---|---|---|---|
| s3d O-lens | 213 | 20 | 70 | 113 |
| ML-NLA RL600 | 85 | 9 | 30 | 24 |
| J-lens | 661 | 1 | 0 | 0 |
| NLA-RL L42 | 5 | 0 | 0 | 0 |

Sensitivity: alternative fire pair logreg + wdm

| method | not committed | probe-validated | cross-method allowed | hallucination |
|---|---|---|---|---|
| s3d O-lens | 213 | 5 | 78 | 120 |
| ML-NLA RL600 | 85 | 0 | 36 | 27 |
| J-lens | 661 | 1 | 0 | 0 |
| NLA-RL L42 | 5 | 0 | 0 | 0 |

Probe status of committed units: {'fires': 20, 'no_fire': 164, 'n/a': 83}; n/a reasons: {'validity gate': 34, 'gold control': 46, 'concept not a bank class': 3}; probe-validated under a weak null (control FPR > 10%): 0.

## Control-concept fire rate (FPR) per bank x layer

| bank | L20 | L24 | L28 | L32 | L36 | L40 | L44 | L48 | L52 | L56 | L60 | L42 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sport | 0/36 | 0/33 | 0/36 | 0/33 | 0/33 | 0/36 | 0/36 | 0/36 | 0/36 | 0/36 | 0/36 | 0/36 |
| country_capital | 0/24 | 0/21 | 0/9 | 0/3 | 0/27 | 0/30 | 1/30 | 1/27 | 0/27 | 0/30 | 0/30 | 2/30 |
| country_landmark | 0/6 | 0/9 | 0/2 | 0/1 | 0/6 | 0/21 | 0/21 | 0/21 | 1/21 | 1/21 | 1/21 | 0/21 |
| musical_instrument | 0/16 | 0/20 | 0/20 | 0/20 | 0/16 | 0/20 | 0/24 | 0/24 | 0/20 | 0/24 | 0/24 | 0/24 |

## Ali vs the case study (`gw-ali-probe/docs/project/experiments/ali_probe/README.md`)

| concept | case-study verdict | ladder labels (per method, layer:label) | probe fires at |
|---|---|---|---|
| american_football | late-layer quiz distractor; not interpretable | s3d O-lens: L52:N; ML-NLA RL600: L60:N; J-lens: L36:N L56:N L60:N | — |
| baseball | (not in the README) | s3d O-lens: L52:N; J-lens: L20:N L24:N L40:N L44:N L56:N L60:N | — |
| basketball | represented for the person (Ali > >=21/22 boxers L20-60) | s3d O-lens: L28:N; J-lens: L36:N L40:N L44:N L48:N L52:N L56:N L60:N | [28, 52, 56, 60] |
| cricket | late-layer quiz distractor; not interpretable | s3d O-lens: L32:H; ML-NLA RL600: L60:N; J-lens: L60:N | — |
| figure_skating | not represented | s3d O-lens: L28:H L32:H | — |
| gymnastics | (not in the README) | s3d O-lens: L20:H | — |
| ice_hockey | (not in the README) | J-lens: L28:N | — |
| soccer | late-layer quiz distractor; not interpretable | J-lens: L36:N L48:N L56:N L60:N | [60] |
| swimming | not represented; lens generic-athlete prior -> hallucination | s3d O-lens: L20:H L24:H L28:X L52:N; ML-NLA RL600: L32:X | — |
| tennis | late-layer quiz distractor; not represented | s3d O-lens: L20:H L24:H L32:H L52:N; J-lens: L56:N L60:N | — |
| track_and_field | represented (Ali > all held-out boxers L8-36), via the name; lens agreement is the prior | s3d O-lens: L20:N L24:P L28:P L32:X L44:N L56:N; ML-NLA RL600: L20:P L24:P L28:P; J-lens: L20:N L36:N | [20, 24, 28] |
| volleyball | (not in the README) | J-lens: L56:N | — |
| wrestling | combat-sport neighbour shared by all boxers; not Ali-specific (no fire vs boxers) | s3d O-lens: L56:N; ML-NLA RL600: L52:N; J-lens: L44:N L48:N L56:N L60:N | — |

## Per-item breakdown

### sport

#### br-ent-ath-ali: "The athlete Muhammad Ali plays the sport of" (gold boxing)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| american_football | J-lens | 36,56,60 | 0 / 3 | described as an athlete in basketball, football, boxing or the olympics | — | — | L36:N L56:N L60:N | 0.27 |
| american_football | ML-NLA RL600 | 60 | 0 / 1 | sports-trivia context may also name cricket, football, or sports he played | — | — | L60:N | 0.27 |
| american_football | s3d O-lens | 52 | 0 / 1 | 拳击 (boxing) or football. | — | — | L52:N | 0.27 |
| baseball | J-lens | 20,24,40,44,56,60 | 0 / 6 | specifically which sport, such as baseball, demands the greatest skill | — | — | L20:N L24:N L40:N L44:N L56:N L60:N | 0.91 |
| baseball | s3d O-lens | 52 | 0 / 1 | 拳击 (boxing) or baseball. no, the question asks about a specific sport | — | — | L52:N | 0.27 |
| basketball | J-lens | 36,40,44,48,52,56,60 | 0 / 7 | described as an athlete in basketball, football, boxing or the olympics | — | — | L36:N L40:N L44:N L48:N L52:N L56:N L60:N | 0.64 |
| basketball | s3d O-lens | 28 | 0 / 1 | he does not play basketball. | — | — | L28:N | 0.64 |
| cricket | J-lens | 60 | 0 / 1 | tennis / ______ / wrestling / baseball / cricket | — | — | L60:N | 0.46 |
| cricket | ML-NLA RL600 | 60 | 0 / 1 | sports-trivia context may also name cricket, football, or sports he played | — | — | L60:N | 0.00 |
| cricket | s3d O-lens | 32 | 1 / 0 | the sport he practices is **cricket**. | L32 r=0.440/0.720 no | L32: none | L32:H | 0.18 |
| figure_skating | s3d O-lens | 28,32 | 2 / 0 | he is a professional athlete who competes in the sport of **figure skating**. | L28 n/a (validity gate (dom AUROC 0.811)); L32 n/a (validity gate (dom AUROC 0.816)) | L28: none; L32: none | L28:H L32:H | 0.64 |
| gymnastics | s3d O-lens | 20 | 1 / 0 | therefore, this person's profession is athlete. the answer is gymnastics | L20 r=0.920/0.660 no | L20: none | L20:H | 0.73 |
| ice_hockey | J-lens | 28 | 0 / 1 | which athlete or which olympic event (e.g., hockey) something refers to | — | — | L28:N | 0.55 |
| soccer | J-lens | 36,48,56,60 | 0 / 4 | basketball / athlete / olympic / olympics / nfl / athletic / basketball / boxing / soccer | — | — | L36:N L48:N L56:N L60:N | 0.55 |
| swimming | ML-NLA RL600 | 32 | 1 / 0 | michael phelps is an olympic swimmer, so the sport slot must resolve to "swimming" | L32 r=0.540/0.840 no | L32: s3d | L32:X | 0.73 |
| swimming | s3d O-lens | 20,24,28,52 | 3 / 1 | he is an olympian who competes in the sport of swimming | L20 n/a (validity gate (dom AUROC 0.845)); L24 r=0.880/0.760 no; L28 r=0.140/0.480 no | L20: none; L24: none; L28: mlnla-rl600 | L20:H L24:H L28:X L52:N | 1.00 |
| tennis | J-lens | 56,60 | 0 / 2 | basketball / football / boxing / tennis / baseball | — | — | L56:N L60:N | 0.18 |
| tennis | s3d O-lens | 20,24,32,52 | 3 / 1 | plays the sport of tennis. | L20 r=0.900/0.980 no; L24 r=0.860/1.000 no; L32 r=0.860/0.880 no | L20: none; L24: none; L32: none | L20:H L24:H L32:H L52:N | 0.73 |
| track_and_field | J-lens | 20,36 | 0 / 2 | anticipating a 'which one/which kind' question (哪一个/哪种/какой = which) about athletics | — | — | L20:N L36:N | 0.36 |
| track_and_field | ML-NLA RL600 | 20,24,28 | 3 / 0 | usain bolt is a sprinter/track and field athlete | L20 r=0.980/1.000 fires; L24 r=1.000/1.000 fires; L28 r=1.000/1.000 fires | L20: jlens,s3d; L24: jlens,s3d; L28: s3d | L20:P L24:P L28:P | 0.64 |
| track_and_field | s3d O-lens | 20,24,28,32,44,56 | 3 / 3 | **athletics** is the correct answer. | L24 r=1.000/1.000 fires; L28 r=1.000/1.000 fires; L32 r=0.880/0.900 no | L24: jlens,mlnla-rl600; L28: mlnla-rl600; L32: jlens,mlnla-rl600 | L20:N L24:P L28:P L32:X L44:N L56:N | 0.91 |
| volleyball | J-lens | 56 | 0 / 1 | basketball / football / boxing / tennis / baseball / soccer / wrestling / basketball / vol | — | — | L56:N | 0.09 |
| wrestling | J-lens | 44,48,56,60 | 0 / 4 | other sports like wrestling, basketball and baseball appear as weaker alternatives | — | — | L44:N L48:N L56:N L60:N | 0.09 |
| wrestling | ML-NLA RL600 | 52 | 0 / 1 | or "boxer"/"wrestling" if the trivia is misstated), likely then a period. | — | — | L52:N | 0.00 |
| wrestling | s3d O-lens | 56 | 0 / 1 | /wrestling athlete known for his golden gloves and heavyweight boxing career. | — | — | L56:N | 0.00 |

#### br-ent-ath-brady: "The athlete Tom Brady plays the sport of" (gold american_football)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| baseball | J-lens | 20,28,40,44,52,56 | 0 / 6 | about athletic greatness or mastery, likely in sports such as baseball | — | — | L20:N L28:N L40:N L44:N L52:N L56:N | 0.91 |
| baseball | s3d O-lens | 36,60 | 1 / 1 | the sport is **professional baseball**). he plays for the boston red sox | L36 r=0.980/0.837 no | L36: jlens | L36:X L60:N | 0.27 |
| basketball | J-lens | 28,40,44,56 | 0 / 4 | athletes in sports like baseball and basketball, possibly a personality quiz | — | — | L28:N L40:N L44:N L56:N | 0.64 |
| basketball | s3d O-lens | 24,32,44 | 1 / 2 | he plays the sport of basketball. | L24 r=1.000/0.939 no | L24: jlens | L24:X L32:N L44:N | 0.64 |
| gymnastics | s3d O-lens | 20,28 | 2 / 0 | the type of sport she plays is gymnastics. | L20 r=0.224/0.163 no; L28 r=0.490/0.286 no | L20: none; L28: none | L20:H L28:H | 0.73 |
| ice_hockey | s3d O-lens | 60 | 0 / 1 | a) baseball b) tennis c) football d) hockey | — | — | L60:N | 0.18 |
| rugby | J-lens | 56 | 0 / 1 | football, baseball, soccer, basketball, rugby; 足球 is chinese for | — | — | L56:N | 0.09 |
| soccer | J-lens | 40,44,48,56,60 | 0 / 5 | with baseball, soccer and basketball as close alternatives | — | — | L40:N L44:N L48:N L56:N L60:N | 0.55 |
| soccer | s3d O-lens | 40,52,56,60 | 3 / 1 | 足球 (football) with a 15-year history. | L40 r=1.000/0.939 no; L52 r=1.000/1.000 fires; L56 r=1.000/1.000 fires | L40: jlens; L52: jlens; L56: jlens | L40:X L52:P L56:P L60:N | 0.55 |
| swimming | ML-NLA RL600 | 24,28,32 | 3 / 0 | michael phelps is an olympic athlete whose sport is swimming | L24 r=0.837/0.755 no; L28 r=0.449/0.776 no; L32 r=0.429/0.898 no | L24: s3d; L28: s3d; L32: s3d | L24:X L28:X L32:X | 0.73 |
| swimming | s3d O-lens | 20,24,28,32 | 4 / 0 | the sport she participates in is swimming. | L20 n/a (validity gate (dom AUROC 0.845)); L24 r=0.837/0.755 no; L28 r=0.449/0.776 no; L32 r=0.429/0.898 no | L20: mlnla-rl600; L24: mlnla-rl600; L28: mlnla-rl600; L32: mlnla-rl600 | L20:X L24:X L28:X L32:X | 1.00 |
| tennis | s3d O-lens | 32,60 | 1 / 1 | mike is a professional tennis player. | L32 r=1.000/0.980 fires | L32: none | L32:P L60:N | 0.73 |
| track_and_field | ML-NLA RL600 | 20 | 1 / 0 | usain bolt is a track sprinter from jamaica, so "the sport" must be track and field | L20 r=0.163/0.224 no | L20: none | L20:H | 0.64 |

#### br-ent-ath-federer: "The athlete Roger Federer plays the sport of" (gold tennis)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| baseball | J-lens | 20 | 0 / 1 | which athlete or feat in something like baseball, represents the greatest mastery | — | — | L20:N | 0.91 |
| basketball | s3d O-lens | 40 | 0 / 1 | a) swimming b) tennis c) golf d) basketball | — | — | L40:N | 0.64 |
| cricket | J-lens | 28 | 0 / 1 | while cricket, olympics, athletic and hockey point to athletics or competition | — | — | L28:N | 0.46 |
| cycling | s3d O-lens | 32 | 1 / 0 | the sport played by lance armstrong is track cycling. | L32 r=0.600/0.260 no | L32: none | L32:H | 0.27 |
| figure_skating | s3d O-lens | 20 | 1 / 0 | the sport is figure skating. | L20 n/a (validity gate (dom AUROC 0.805)) | L20: none | L20:H | 0.64 |
| golf | s3d O-lens | 40 | 0 / 1 | a) swimming b) tennis c) golf d) basketball | — | — | L40:N | 0.09 |
| ice_hockey | J-lens | 28 | 0 / 1 | cricket, olympics, athletic and hockey point to athletics or competition | — | — | L28:N | 0.55 |
| ice_hockey | s3d O-lens | 28 | 1 / 0 | the sport is called "ice hockey." | L28 r=0.580/0.960 no | L28: jlens | L28:X | 0.18 |
| rowing | s3d O-lens | 28 | 1 / 0 | /category: **rowing**. here is the step-by-step logic: | L28 r=0.220/0.100 no | L28: none | L28:H | 0.27 |
| rugby | s3d O-lens | 56 | 0 / 1 | 网球 (tennis) or perhaps rugby. | — | — | L56:N | 0.27 |
| skiing | ML-NLA RL600 | 20 | 1 / 0 | lindsey vonn is an american alpine skier; the sport slot is alpine skiing/skiing | L20 r=0.340/0.120 no | L20: none | L20:H | 0.09 |
| soccer | s3d O-lens | 44,48 | 2 / 0 | which sport does david beckham play? the answer is football (soccer). | L44 r=0.960/1.000 no; L48 r=0.980/1.000 fires | L44: none; L48: none | L44:H L48:P | 0.55 |
| swimming | ML-NLA RL600 | 24,28 | 2 / 0 | a dangling prepositional phrase requiring the sport name immediately, i.e. "swimming." | L24 r=0.460/0.160 no; L28 r=0.180/0.220 no | L24: s3d; L28: s3d | L24:X L28:X | 0.73 |
| swimming | s3d O-lens | 20,24,28,32,40 | 4 / 1 | **swimming** is the sport he practices | L20 n/a (validity gate (dom AUROC 0.845)); L24 r=0.460/0.160 no; L28 r=0.180/0.220 no; L32 r=0.620/0.560 no | L20: mlnla-rl600; L24: mlnla-rl600; L28: mlnla-rl600; L32: mlnla-rl600 | L20:X L24:X L28:X L32:X L40:N | 1.00 |
| track_and_field | J-lens | 28 | 0 / 1 | cricket, olympics, athletic and hockey point to athletics or competition | — | — | L28:N | 0.36 |
| track_and_field | s3d O-lens | 20,24,28,32 | 3 / 1 | profession he plays is **track and field**. if you look at the rules | L20 r=0.260/0.100 no; L24 r=0.380/0.340 no; L28 r=0.580/0.420 no | L20: none; L24: jlens; L28: jlens | L20:H L24:X L28:X L32:N | 0.91 |

#### br-ent-ath-gretzky: "The athlete Wayne Gretzky plays the sport of" (gold ice_hockey)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| american_football | s3d O-lens | 32 | 1 / 0 |  | L32 r=1.000/1.000 fires | L32: none | L32:P | 0.27 |
| baseball | J-lens | 20,24,28,36,40 | 0 / 5 | among athletic pursuits, such as baseball or wrestling, framed around greatness and master | — | — | L20:N L24:N L28:N L36:N L40:N | 0.91 |
| baseball | ML-NLA RL600 | 20 | 1 / 0 | the athlete todd helton plays the sport of" is an incomplete copular statement requiring a | L20 r=0.980/1.000 fires | L20: jlens | L20:P | 0.18 |
| basketball | J-lens | 40,56 | 0 / 2 | other team sports like baseball, soccer and basketball as weaker alternative candidates | — | — | L40:N L56:N | 0.64 |
| basketball | s3d O-lens | 40 | 1 / 0 | 篮球 (basketball) in canada. | L40 r=0.940/0.780 no | L40: jlens | L40:X | 0.64 |
| boxing | s3d O-lens | 32 | 1 / 0 | what sport is mike tyson known for? **answer:** boxing | L32 r=0.380/0.820 no | L32: none | L32:H | 0.09 |
| figure_skating | s3d O-lens | 20 | 1 / 0 | figure skating is the sport. it is done indoors on ice. | L20 n/a (validity gate (dom AUROC 0.805)) | L20: none | L20:H | 0.64 |
| golf | ML-NLA RL600 | 32 | 1 / 0 | tiger woods is a golfer, so the sport must be golf | L32 r=0.980/0.960 no | L32: none | L32:H | 0.09 |
| gymnastics | s3d O-lens | 24 | 1 / 0 | that kelly participates in is **gymnastics**. | L24 n/a (gold control (dom p 0.057)) | L24: none | L24:H | 0.73 |
| rowing | s3d O-lens | 28 | 1 / 0 | he is an athlete who competes in the sport of rowing. | L28 r=0.000/0.000 no | L28: none | L28:H | 0.27 |
| snowboarding | ML-NLA RL600 | 28 | 1 / 0 | shaun white is a snowboarder, so the sport is snowboarding | L28 n/a (validity gate (dom AUROC 0.843)) | L28: none | L28:H | 0.00 |
| soccer | J-lens | 40,44,56 | 0 / 3 | other team sports like baseball, soccer and basketball as weaker alternative candidates | — | — | L40:N L44:N L56:N | 0.55 |
| soccer | s3d O-lens | 32,48 | 1 / 1 | 足球 (ice hockey) player. | L48 r=0.780/0.800 no | L48: jlens | L32:N L48:X | 0.55 |
| softball | ML-NLA RL600 | 20 | 0 / 1 | the unnamed athlete is clearly a baseball/softball figure | — | — | L20:N | 0.00 |
| swimming | ML-NLA RL600 | 24 | 1 / 0 | the required completion is almost certainly "swimming" (or "swimming." with a period) | L24 n/a (gold control (dom p 0.057)) | L24: s3d | L24:X | 0.73 |
| swimming | s3d O-lens | 20,24,28 | 3 / 0 | 游泳 (swimming) is the sport he participates in | L20 n/a (validity gate (dom AUROC 0.845)); L24 n/a (gold control (dom p 0.057)); L28 r=0.960/0.240 no | L20: mlnla-rl600; L24: mlnla-rl600; L28: mlnla-rl600 | L20:X L24:X L28:X | 1.00 |
| tennis | s3d O-lens | 28 | 1 / 0 | the sport played by billie jean king is tennis. | L28 r=1.000/1.000 fires | L28: none | L28:P | 0.73 |
| track_and_field | J-lens | 28 | 0 / 1 | about sports such as hockey, baseball, the olympics or athletics | — | — | L28:N | 0.36 |
| track_and_field | s3d O-lens | 20,24,28,32 | 2 / 2 | the sport of track and field is where he competes. | L20 r=0.560/1.000 no; L32 r=0.700/0.980 no | L20: none; L32: jlens | L20:H L24:N L28:N L32:X | 0.91 |
| wrestling | J-lens | 20,24 | 0 / 2 | among athletic pursuits, such as baseball or wrestling, framed around greatness and master | — | — | L20:N L24:N | 0.09 |

#### br-ent-ath-james: "The athlete LeBron James plays the sport of" (gold basketball)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| american_football | J-lens | 40,44,56 | 0 / 3 | weaker activation for other major american sports like the nfl and baseball | — | — | L40:N L44:N L56:N | 0.27 |
| american_football | ML-NLA RL600 | 32,56 | 1 / 1 | lionel messi is a footballer, so the expected answer is a sport | L32 r=0.800/0.820 no | L32: none | L32:H L56:N | 0.27 |
| american_football | s3d O-lens | 60 | 0 / 1 | basketball" (or potentially football, but basketball fits the name pattern better) | — | — | L60:N | 0.27 |
| baseball | J-lens | 20,40,44,56 | 0 / 4 | with baseball as a likely example | — | — | L20:N L40:N L44:N L56:N | 0.91 |
| cycling | s3d O-lens | 28 | 1 / 0 | she is a **cyclist**. specifically, she competes in **track cycling** | L28 r=0.360/0.220 no | L28: none | L28:H | 0.27 |
| figure_skating | s3d O-lens | 28 | 0 / 1 | figure skating is a sport performed on ice, involving | — | — | L28:N | 0.64 |
| ice_hockey | ML-NLA RL600 | 20 | 1 / 0 | patrick kane is an ice hockey player for the chicago blackhawks | L20 r=0.060/0.200 no | L20: none | L20:H | 0.09 |
| soccer | J-lens | 40,44,48,56 | 0 / 4 | baseball / football / 体育 / soccer | — | — | L40:N L44:N L48:N L56:N | 0.55 |
| soccer | ML-NLA RL600 | 32 | 1 / 0 | lionel messi is a footballer, so the expected answer is a sport — soccer/football/associat | L32 r=0.960/1.000 no | L32: none | L32:H | 0.18 |
| soccer | s3d O-lens | 36 | 0 / 1 | what sport is he most famously associated with? a) soccer b) basketball | — | — | L36:N | 0.55 |
| swimming | s3d O-lens | 20,24,28,32 | 4 / 0 | the person plays the sport of swimming. | L20 n/a (validity gate (dom AUROC 0.845)); L24 r=0.960/0.760 no; L28 r=0.800/0.740 no; L32 r=0.600/1.000 no | L20: none; L24: none; L28: none; L32: none | L20:H L24:H L28:H L32:H | 1.00 |
| tennis | s3d O-lens | 36 | 0 / 1 | a) soccer b) basketball c) tennis | — | — | L36:N | 0.73 |
| track_and_field | J-lens | 20 | 0 / 1 | competitive sport (竞技 means "athletics/competition"), with baseball as a likely example | — | — | L20:N | 0.36 |
| track_and_field | ML-NLA RL600 | 24,28 | 2 / 0 | subject established as sprinter lamine diack, so the sport in question is track and field | L24 r=0.560/0.740 no; L28 r=0.660/0.560 no | L24: jlens,s3d; L28: s3d | L24:X L28:X | 0.64 |
| track_and_field | s3d O-lens | 20,24 | 1 / 1 | plays the sport of **track and field**. it is not a player | L20 r=0.440/0.640 no | L20: jlens,mlnla-rl600 | L20:X L24:N | 0.91 |

#### br-ent-ath-jordan: "The athlete Michael Jordan plays the sport of" (gold basketball)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| american_football | J-lens | 56 | 0 / 1 | other sports such as baseball, soccer and football as weaker alternatives | — | — | L56:N | 0.27 |
| american_football | ML-NLA RL600 | 20,56 | 1 / 1 | tony romo is an american football player (nfl, dallas cowboys quarterback) | L20 r=0.980/0.980 fires | L20: none | L20:P L56:N | 0.27 |
| baseball | J-lens | 20,40,44,56,60 | 0 / 5 | the topic seems to be athletic greatness or mastery, possibly in baseball | — | — | L20:N L40:N L44:N L56:N L60:N | 0.91 |
| baseball | ML-NLA RL600 | 20,56 | 0 / 2 | requiring a sport name immediately (likely "football" or "baseball"), then a period | — | — | L20:N L56:N | 0.18 |
| baseball | s3d O-lens | 60 | 0 / 1 | basketball or baseball. the question asks to complete the sentence about michael jordan. | — | — | L60:N | 0.27 |
| fencing | s3d O-lens | 20 | 1 / 0 | the person is a professional athlete. specifically, a **fencer**. | L20 r=0.120/0.040 no | L20: none | L20:H | 0.00 |
| figure_skating | s3d O-lens | 20 | 1 / 0 | 3: figure skating. we know the sport is gymnastics. | L20 n/a (validity gate (dom AUROC 0.805)) | L20: none | L20:H | 0.64 |
| golf | ML-NLA RL600 | 32 | 1 / 0 | tiger woods is a golfer; the sport is golf | L32 r=0.400/0.460 no | L32: none | L32:H | 0.09 |
| gymnastics | s3d O-lens | 20 | 1 / 0 | we know the sport is gymnastics. | L20 r=0.340/0.120 no | L20: none | L20:H | 0.73 |
| skiing | s3d O-lens | 32 | 1 / 0 | the sport is skiing, and its primary category is **sports | L32 r=0.820/0.100 no | L32: none | L32:H | 0.09 |
| soccer | J-lens | 40,44,48,56,60 | 0 / 5 | with related alternatives like soccer, baseball, the nba and the olympics | — | — | L40:N L44:N L48:N L56:N L60:N | 0.55 |
| soccer | s3d O-lens | 36 | 0 / 1 | it is not known for being a "soccer player" in the traditional sense | — | — | L36:N | 0.55 |
| swimming | ML-NLA RL600 | 28 | 1 / 0 | michael phelps's sport is swimming | L28 r=0.780/0.460 no | L28: s3d | L28:X | 0.73 |
| swimming | s3d O-lens | 20,24,28,32 | 4 / 0 | the sport he plays is swimming. | L20 n/a (validity gate (dom AUROC 0.845)); L24 r=0.880/0.640 no; L28 r=0.780/0.460 no; L32 r=0.460/0.580 no | L20: none; L24: mlnla-rl600; L28: mlnla-rl600; L32: mlnla-rl600 | L20:H L24:X L28:X L32:X | 1.00 |
| tennis | ML-NLA RL600 | 56 | 0 / 1 | likely a choice list: basketball, baseball, football, tennis | — | — | L56:N | 0.27 |
| tennis | s3d O-lens | 28,32 | 1 / 1 | therefore, jimmy bower is a professional tennis player. | L28 r=0.900/0.980 no | L28: none | L28:H L32:N | 0.73 |
| track_and_field | ML-NLA RL600 | 24 | 1 / 0 | usain bolt is a jamaican sprinter, so the sport to name is track and field/athletics | L24 r=0.940/0.880 no | L24: s3d | L24:X | 0.64 |
| track_and_field | s3d O-lens | 20,24,28,32 | 2 / 2 | he is a track and field athlete. | L28 r=0.800/0.780 no; L32 r=0.580/0.480 no | L28: mlnla-rl600; L32: none | L20:N L24:N L28:X L32:H | 0.91 |

#### br-ent-ath-messi: "The athlete Lionel Messi plays the sport of" (gold soccer)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| baseball | J-lens | 20 | 0 / 1 | the likely frame is athletic mastery or greatness, perhaps in baseball | — | — | L20:N | 0.91 |
| basketball | J-lens | 40 | 0 / 1 | sport / 体育 / __________________ / football / basketball | — | — | L40:N | 0.64 |
| basketball | s3d O-lens | 36,40 | 2 / 0 | he plays **basketball**. this is his primary profession and | L36 r=1.000/1.000 fires; L40 r=1.000/1.000 fires | L36: jlens; L40: jlens | L36:P L40:P | 0.64 |
| figure_skating | s3d O-lens | 24 | 1 / 0 | the sport is **figure skating**. it is a | L24 n/a (validity gate (dom AUROC 0.785)) | L24: none | L24:H | 0.64 |
| rugby | s3d O-lens | 56 | 0 / 1 | what sport does lionel messi primarily play? a) rugby b) soccer | — | — | L56:N | 0.27 |
| swimming | ML-NLA RL600 | 24 | 1 / 0 | michael phelps is the world's most decorated swimmer, so the named sport must be swimming | L24 r=0.860/0.620 no | L24: s3d | L24:X | 0.73 |
| swimming | s3d O-lens | 20,24,28,36,40 | 4 / 1 | he competes in the sport of swimming, and he is a | L20 n/a (validity gate (dom AUROC 0.845)); L24 r=0.860/0.620 no; L28 r=0.200/0.200 no; L36 r=0.420/0.480 no | L20: mlnla-rl600; L24: mlnla-rl600; L28: mlnla-rl600; L36: none | L20:X L24:X L28:X L36:H L40:N | 1.00 |
| tennis | ML-NLA RL600 | 28,32 | 2 / 0 | domain facts required: roger federer's sport, tennis; | L28 r=1.000/1.000 fires; L32 r=1.000/1.000 fires | L28: s3d; L32: s3d | L28:P L32:P | 0.27 |
| tennis | s3d O-lens | 28,32,40 | 1 / 2 | he is a tennis player, specifically a professional athlete who competes | L32 r=1.000/1.000 fires | L32: mlnla-rl600 | L28:N L32:P L40:N | 0.73 |
| track_and_field | s3d O-lens | 20,24,28 | 3 / 0 | 运动 is track and field. he plays the role of an athlete | L20 r=0.660/0.800 no; L24 r=0.540/0.880 no; L28 r=0.900/0.700 no | L20: none; L24: none; L28: none | L20:H L24:H L28:H | 0.91 |

#### br-ent-ath-nadal: "The athlete Rafael Nadal plays the sport of" (gold tennis)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| badminton | s3d O-lens | 60 | 0 / 1 | 2. tennis **c. badminton** d. golf **e. table tennis | — | — | L60:N | 0.00 |
| baseball | J-lens | 20,24 | 0 / 2 | it looks like a passage asking which sport, such as baseball, or which athlete | — | — | L20:N L24:N | 0.91 |
| baseball | s3d O-lens | 44 | 0 / 1 | (a) football (b) tennis (c) rugby (d) baseball | — | — | L44:N | 0.27 |
| basketball | s3d O-lens | 48 | 0 / 1 | a) tennis b) soccer c) basketball | — | — | L48:N | 0.64 |
| cricket | J-lens | 28 | 0 / 1 | such as the olympics, cricket, hockey, athletics or marvel | — | — | L28:N | 0.46 |
| cycling | s3d O-lens | 32 | 0 / 1 | the most famous cyclist associated | — | — | L32:N | 0.27 |
| figure_skating | s3d O-lens | 20 | 1 / 0 | figure skating. the sport he participates in is soccer. | L20 n/a (validity gate (dom AUROC 0.805)) | L20: none | L20:H | 0.64 |
| golf | s3d O-lens | 60 | 0 / 1 | 2. tennis **c. badminton** d. golf **e. table tennis | — | — | L60:N | 0.09 |
| gymnastics | s3d O-lens | 24 | 1 / 0 | gymnastics. the answer provided states that bill is an athlete. | L24 r=0.160/0.220 no | L24: none | L24:H | 0.73 |
| ice_hockey | J-lens | 28 | 0 / 1 | such as the olympics, cricket, hockey, athletics or marvel | — | — | L28:N | 0.55 |
| ice_hockey | ML-NLA RL600 | 20 | 1 / 0 | the vancouver canucks are an ice hockey team, so the sport is hockey; subject is an athlet | L20 r=0.640/0.740 no | L20: none | L20:H | 0.09 |
| rowing | s3d O-lens | 28 | 1 / 0 | /profession is rowing. | L28 r=0.120/0.100 no | L28: none | L28:H | 0.27 |
| rugby | ML-NLA RL600 | 48 | 0 / 1 | answer choices like "a. tennis b. rugby", possibly then more quiz boilerplate | — | — | L48:N | 0.09 |
| rugby | s3d O-lens | 36,44 | 0 / 2 | he is not known for playing football, rugby, or | — | — | L36:N L44:N | 0.27 |
| soccer | ML-NLA RL600 | 32 | 1 / 0 | lionel messi is an argentine football/soccer player, so the sport domain is the answer. | L32 r=0.940/0.980 no | L32: none | L32:H | 0.18 |
| soccer | s3d O-lens | 20,36,44,48 | 1 / 3 | the sport he participates in is soccer. he is a | L20 r=0.960/0.980 no | L20: none | L20:H L36:N L44:N L48:N | 0.55 |
| swimming | s3d O-lens | 20,24,28,32 | 4 / 0 | the main sport he plays is swimming. | L20 n/a (validity gate (dom AUROC 0.845)); L24 r=0.900/0.380 no; L28 r=0.700/0.220 no; L32 r=0.660/0.360 no | L20: none; L24: none; L28: none; L32: none | L20:H L24:H L28:H L32:H | 1.00 |
| table_tennis | s3d O-lens | 60 | 0 / 1 | 2. tennis **c. badminton** d. golf **e. table tennis | — | — | L60:N | 0.09 |
| track_and_field | J-lens | 20,28 | 0 / 2 | 竞技 means "athletics/competitive sports" | — | — | L20:N L28:N | 0.36 |
| track_and_field | ML-NLA RL600 | 24,28 | 2 / 0 | topic entity established as usain bolt, track athlete | L24 r=0.440/0.620 no; L28 r=0.360/0.520 no | L24: jlens,s3d; L28: jlens,s3d | L24:X L28:X | 0.64 |
| track_and_field | s3d O-lens | 20,24,28,32 | 3 / 1 | 游泳 (swimming) or his specific career is track and field. | L20 r=0.260/0.560 no; L28 r=0.360/0.520 no; L32 r=0.180/0.200 no | L20: jlens,mlnla-rl600; L28: jlens,mlnla-rl600; L32: jlens,mlnla-rl600 | L20:X L24:N L28:X L32:X | 0.91 |

#### br-ent-ath-ronaldo: "The athlete Cristiano Ronaldo plays the sport of" (gold soccer)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| baseball | J-lens | 20 | 0 / 1 | with baseball as a likely candidate | — | — | L20:N | 0.91 |
| basketball | J-lens | 40,56 | 0 / 2 | basketball appears as a weaker, related sport | — | — | L40:N L56:N | 0.64 |
| basketball | s3d O-lens | 32 | 0 / 1 | 体育 (sports) career, and the object is basketball. | — | — | L32:N | 0.64 |
| cricket | s3d O-lens | 60 | 1 / 0 | the answer is **cricket**. | L60 r=0.740/0.680 no | L60: none | L60:H | 0.18 |
| cycling | s3d O-lens | 28 | 1 / 0 | the sport that this person practices is track cycling. | L28 r=0.120/0.160 no | L28: none | L28:H | 0.27 |
| gymnastics | s3d O-lens | 20 | 1 / 0 | the sport is gymnastics. he is a | L20 r=0.340/0.180 no | L20: none | L20:H | 0.73 |
| ice_hockey | J-lens | 28 | 0 / 1 | together with 'athletic', 'hockey', and 'marvel', this suggests a quiz- or survey-style qu | — | — | L28:N | 0.55 |
| rugby | s3d O-lens | 48 | 1 / 0 | rugby (specifically, soccer) is the sport ronaldo plays. | L48 r=0.260/0.140 no | L48: none | L48:H | 0.27 |
| swimming | ML-NLA RL600 | 24,28,32 | 3 / 0 | domain knowledge: michael phelps = swimming; the subject is a famous us olympian | L24 r=0.720/0.400 no; L28 r=0.640/0.140 no; L32 r=0.640/0.200 no | L24: s3d; L28: s3d; L32: s3d | L24:X L28:X L32:X | 0.73 |
| swimming | s3d O-lens | 20,24,28,32 | 3 / 1 | the sport is **swimming**. he is a professional swimmer. | L20 n/a (validity gate (dom AUROC 0.845)); L24 r=0.720/0.400 no; L32 r=0.640/0.200 no | L20: mlnla-rl600; L24: mlnla-rl600; L32: mlnla-rl600 | L20:X L24:X L28:N L32:X | 1.00 |
| tennis | ML-NLA RL600 | 20 | 1 / 0 | established entity: andy murray, british tennis player | L20 r=0.720/0.880 no | L20: s3d | L20:X | 0.27 |
| tennis | s3d O-lens | 20 | 1 / 0 | when we look at the specific sport of tennis, we find that phelps played it. | L20 r=0.720/0.880 no | L20: mlnla-rl600 | L20:X | 0.73 |
| track_and_field | s3d O-lens | 20,24 | 2 / 0 | the sport he plays is track and field. | L20 r=0.660/0.740 no; L24 r=0.720/0.840 no | L20: none; L24: none | L20:H L24:H | 0.91 |

#### br-ent-ath-ruth: "The athlete Babe Ruth plays the sport of" (gold baseball)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| american_football | J-lens | 40,44,48,52,56,60 | 0 / 6 | other team sports such as hockey, cricket, soccer, basketball and football appear as close | — | — | L40:N L44:N L48:N L52:N L56:N L60:N | 0.27 |
| basketball | J-lens | 40,44,52,56,60 | 0 / 5 | other team sports such as hockey, cricket, soccer, basketball and football appear as close | — | — | L40:N L44:N L52:N L56:N L60:N | 0.64 |
| basketball | ML-NLA RL600 | 20 | 1 / 0 | requiring the sport name — almost certainly "basketball" — followed by a period | L20 r=0.960/0.960 no | L20: none | L20:H | 0.00 |
| basketball | s3d O-lens | 36 | 0 / 1 | 2. basketball. he was a professional baseball player who primarily played for | — | — | L36:N | 0.64 |
| boxing | s3d O-lens | 36 | 1 / 0 | the answer is **boxing**. | L36 r=0.700/0.520 no | L36: none | L36:H | 0.09 |
| cricket | J-lens | 40,44,48,56,60 | 0 / 5 | other team sports such as hockey, cricket, soccer, basketball and football appear as close | — | — | L40:N L44:N L48:N L56:N L60:N | 0.46 |
| figure_skating | s3d O-lens | 20,28 | 2 / 0 | the athlete competes in the sport of **figure skating**. | L20 n/a (validity gate (dom AUROC 0.805)); L28 n/a (validity gate (dom AUROC 0.811)) | L20: none; L28: none | L20:H L28:H | 0.64 |
| gymnastics | s3d O-lens | 24,28 | 1 / 1 | the sport associated with this person is gymnastics | L24 r=0.680/0.380 no | L24: none | L24:H L28:N | 0.73 |
| ice_hockey | J-lens | 28,36,40,44,48 | 0 / 5 | the likely subject is sports or the olympics (hockey, athletic) | — | — | L28:N L36:N L40:N L44:N L48:N | 0.55 |
| skateboarding | ML-NLA RL600 | 32 | 1 / 0 | tony hawk is famous for skateboarding | L32 r=0.600/0.340 no | L32: none | L32:H | 0.00 |
| soccer | J-lens | 40,44,48,56,60 | 0 / 5 | other team sports such as hockey, cricket, soccer, basketball and football appear as close | — | — | L40:N L44:N L48:N L56:N L60:N | 0.55 |
| soccer | s3d O-lens | 32,60 | 0 / 2 | （or rather, plays) soccer, and the category is "sport". therefore, the answer is **tennis* | — | — | L32:N L60:N | 0.55 |
| softball | J-lens | 56,60 | 0 / 2 | mlb / cricket / softball / tennis | — | — | L56:N L60:N | 0.00 |
| swimming | s3d O-lens | 20,24 | 1 / 1 | 运动 of swimming. the person is a sports figure. | L20 n/a (validity gate (dom AUROC 0.845)) | L20: none | L20:H L24:N | 1.00 |
| tennis | J-lens | 56 | 0 / 1 | football / basketball / soccer / mlb / cricket / softball / tennis | — | — | L56:N | 0.18 |
| tennis | s3d O-lens | 32 | 1 / 0 | the sport that michael plays is **tennis**. | L32 r=0.740/0.680 no | L32: none | L32:H | 0.73 |
| track_and_field | ML-NLA RL600 | 24,28 | 2 / 0 | "ten-time world champion" plus olympic gold medals signals track and field/athletics, spec | L24 r=0.520/0.680 no; L28 r=0.260/0.380 no | L24: s3d; L28: none | L24:X L28:H | 0.64 |
| track_and_field | s3d O-lens | 20 | 1 / 0 | the sport he competes in is track and field. | L20 r=0.600/0.640 no | L20: mlnla-rl600 | L20:X | 0.91 |

#### br-ent-ath-williams: "The athlete Serena Williams plays the sport of" (gold tennis)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| baseball | J-lens | 20 | 0 / 1 | which type of athletic greatness or mastery, such as in baseball, is being discussed | — | — | L20:N | 0.91 |
| basketball | J-lens | 40 | 0 / 1 | most likely tennis or basketball, to fill a blank | — | — | L40:N | 0.64 |
| cricket | J-lens | 28 | 0 / 1 | sport (cricket, hockey, olympics), or pop-culture item (marvel) | — | — | L28:N | 0.46 |
| gymnastics | s3d O-lens | 36 | 1 / 0 | gymnastics is the sport she primarily plays. | L36 r=0.720/0.540 no | L36: none | L36:H | 0.73 |
| ice_hockey | J-lens | 28 | 0 / 1 | athlete, sport (cricket, hockey, olympics), or pop-culture item (marvel) | — | — | L28:N | 0.55 |
| ice_hockey | s3d O-lens | 28 | 1 / 0 | the sport they participate in is **ice hockey**. | L28 r=0.360/0.860 no | L28: jlens | L28:X | 0.18 |
| skiing | ML-NLA RL600 | 20 | 1 / 0 | "mia bing" is a chinese skier, so alpine skiing or ski racing is the expected domain. | L20 r=0.220/0.080 no | L20: none | L20:H | 0.09 |
| soccer | ML-NLA RL600 | 20 | 0 / 1 | referencing sports database entities ("world cup racing world championships") | — | — | L20:N | 0.18 |
| swimming | ML-NLA RL600 | 20,32 | 1 / 1 | michael phelps is the most decorated olympic swimmer; the required fact is swimming | L32 r=0.220/0.380 no | L32: s3d | L20:N L32:X | 0.73 |
| swimming | s3d O-lens | 20,24,28,32 | 4 / 0 | sport: she competes in swimming. this is a very specific sport. | L20 n/a (validity gate (dom AUROC 0.845)); L24 r=0.360/0.360 no; L28 r=0.120/0.200 no; L32 r=0.220/0.380 no | L20: mlnla-rl600; L24: mlnla-rl600; L28: mlnla-rl600; L32: mlnla-rl600 | L20:X L24:X L28:X L32:X | 1.00 |
| table_tennis | s3d O-lens | 52 | 1 / 0 | 乒乓球 (or tennis, depending on the source, but commonly associated with serena williams) spo | L52 r=0.000/0.080 no | L52: none | L52:H | 0.09 |
| track_and_field | ML-NLA RL600 | 24,28 | 1 / 1 | michael johnson is a sprinter, so the sport is athletics/track and field | L28 r=0.440/0.740 no | L28: s3d | L24:N L28:X | 0.64 |
| track_and_field | s3d O-lens | 20,24,32 | 3 / 0 | specifically, she is a track and field athlete. the sport is track and field. | L20 r=0.540/0.640 no; L24 r=0.380/0.800 no; L32 r=0.760/0.540 no | L20: mlnla-rl600; L24: mlnla-rl600; L32: mlnla-rl600 | L20:X L24:X L32:X | 0.91 |
| triathlon | ML-NLA RL600 | 24 | 0 / 1 | most plausibly "athletics" or "triathlon"/"running," followed by a period | — | — | L24:N | 0.00 |
| volleyball | J-lens | 56 | 0 / 1 | most likely "tennis" (with volleyball as a weaker alternative) | — | — | L56:N | 0.09 |

#### br-ent-ath-woods: "The athlete Tiger Woods plays the sport of" (gold golf)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| american_football | ML-NLA RL600 | 44 | 0 / 1 | possibly multiple-choice options like "golf / tennis / football" continuing the quiz list | — | — | L44:N | 0.27 |
| american_football | s3d O-lens | 32,52,60 | 0 / 3 | football is an athletic activity, not a geographic location. | — | — | L32:N L52:N L60:N | 0.27 |
| baseball | J-lens | 20,40,44 | 0 / 3 | mastery or competency is involved, possibly in baseball, such as which athlete | — | — | L20:N L40:N L44:N | 0.91 |
| baseball | ML-NLA RL600 | 40 | 0 / 1 | possible distractors like tennis/baseball may follow if multiple-choice | — | — | L40:N | 0.18 |
| basketball | J-lens | 40 | 0 / 1 | golf / golf / tennis / 高尔夫 / tennis / baseball / nascar / soccer / basketball / pga | — | — | L40:N | 0.64 |
| cricket | J-lens | 28 | 0 / 1 | sit alongside cricket, rugby, hockey, athlete and athletic | — | — | L28:N | 0.46 |
| cricket | s3d O-lens | 60 | 0 / 1 | 3. **tennis** 4. **football** 5. **cricket** | — | — | L60:N | 0.18 |
| gymnastics | s3d O-lens | 20,24,32 | 1 / 2 | sport:** the specific sport practiced is **gymnastics**. | L24 r=0.700/0.480 no | L24: none | L20:N L24:H L32:N | 0.73 |
| ice_hockey | J-lens | 28 | 0 / 1 | sit alongside cricket, rugby, hockey, athlete and athletic | — | — | L28:N | 0.55 |
| motor_racing | J-lens | 40 | 0 / 1 | other sports like tennis, baseball and nascar show up only as weaker alternatives | — | — | L40:N | 0.00 |
| rowing | s3d O-lens | 28,32 | 1 / 1 | the sport associated with wayne is **rowing** (also known as | L32 n/a (gold control (dom p 0.105)) | L32: none | L28:N L32:H | 0.27 |
| rugby | J-lens | 28 | 0 / 1 | sit alongside cricket, rugby, hockey, athlete and athletic | — | — | L28:N | 0.09 |
| rugby | ML-NLA RL600 | 56 | 0 / 1 | possibly multiple-choice options (golf, tennis, rugby) | — | — | L56:N | 0.09 |
| skiing | s3d O-lens | 36 | 0 / 1 | a) swimming b) tennis c) golf d) skiing | — | — | L36:N | 0.09 |
| soccer | J-lens | 40 | 0 / 1 | golf / golf / tennis / 高尔夫 / tennis / baseball / nascar / soccer | — | — | L40:N | 0.55 |
| swimming | ML-NLA RL600 | 20 | 1 / 0 | topic domain: michael phelps, olympic swimming legend | L20 n/a (validity gate (dom AUROC 0.845)) | L20: s3d | L20:X | 0.73 |
| swimming | s3d O-lens | 20,24,28,32,36 | 4 / 1 | 游泳 (swimming) is the sport he competes in. | L20 n/a (validity gate (dom AUROC 0.845)); L24 r=0.600/0.540 no; L28 r=0.400/0.440 no; L32 n/a (gold control (dom p 0.105)) | L20: mlnla-rl600; L24: mlnla-rl600; L28: none; L32: none | L20:X L24:X L28:H L32:H L36:N | 1.00 |
| tennis | J-lens | 40,44,60 | 0 / 3 | other sports like tennis, baseball and nascar show up only as weaker alternatives | — | — | L40:N L44:N L60:N | 0.18 |
| tennis | ML-NLA RL600 | 40,44,56 | 0 / 3 | possible distractors like tennis/baseball may follow if multiple-choice | — | — | L40:N L44:N L56:N | 0.27 |
| tennis | s3d O-lens | 28,36,44,52,56,60 | 0 / 6 | it does not play tennis. | — | — | L28:N L36:N L44:N L52:N L56:N L60:N | 0.73 |
| track_and_field | ML-NLA RL600 | 24,28 | 2 / 0 | named entity "lolo jones" primes domain knowledge: us sprinter, track and field, olympic h | L24 r=0.980/0.920 no; L28 r=0.980/0.780 no | L24: s3d; L28: s3d | L24:X L28:X | 0.64 |
| track_and_field | s3d O-lens | 20,24,32 | 2 / 1 | the sport he participates in is track and field. | L20 r=0.900/0.860 no; L24 r=0.980/0.920 no | L20: mlnla-rl600; L24: mlnla-rl600 | L20:X L24:X L32:N | 0.91 |

### country_capital

#### br-ent-cap-athens: "The city of Athens is the capital of" (gold greece)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| bulgaria | J-lens | 44,48,52 | 0 / 3 | neighbors like bulgaria, cyprus, macedonia and croatia appear as close geographic alternat | — | — | L44:N L48:N L52:N | 0.00 |
| china | J-lens | 40,56,60 | 0 / 3 | 希腊 is chinese for 'greece' | — | — | L40:N L56:N L60:N | 0.89 |
| croatia | J-lens | 28,44,48,52,56 | 0 / 5 | with croatia showing up as a neighboring mediterranean country | — | — | L28:N L44:N L48:N L52:N L56:N | 0.00 |
| cyprus | J-lens | 40,44,48,52 | 0 / 4 | with nearby associations to athens, the mediterranean, cyprus and italy | — | — | L40:N L44:N L48:N L52:N | 0.00 |
| germany | J-lens | 56 | 0 / 1 | spanish/italian (grecia) and german (griechenland), along with nearby european countries | — | — | L56:N | 0.33 |
| italy | J-lens | 36,40,52,56,60 | 0 / 5 | related concepts (athens, greek, mediterranean, europe, italy) point to a geographic or na | — | — | L36:N L40:N L52:N L56:N L60:N | 0.33 |
| north_macedonia | J-lens | 44 | 0 / 1 | neighbors like bulgaria, cyprus, macedonia and croatia appear as close geographic alternat | — | — | L44:N | 0.00 |
| romania | J-lens | 48,52,56 | 0 / 3 | neighboring balkan/southeastern european countries (bulgaria, cyprus, croatia, romania, se | — | — | L48:N L52:N L56:N | 0.22 |
| russia | J-lens | 60 | 0 / 1 | spanish/italian 'grecia', and the russian fragment 'гре-' | — | — | L60:N | 0.67 |
| serbia | J-lens | 48 | 0 / 1 | neighboring balkan/southeastern european countries (bulgaria, cyprus, croatia, romania, se | — | — | L48:N | 0.11 |
| spain | J-lens | 56,60 | 0 / 2 | the name in chinese (希腊), spanish/italian (grecia) and german (griechenland) | — | — | L56:N L60:N | 0.56 |

#### br-ent-cap-berlin: "The city of Berlin is the capital of" (gold germany)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| austria | J-lens | 44,48,52 | 0 / 3 | related readouts like 'bundes', 'republic', 'nazi', and 'austria' point to germany's polit | — | — | L44:N L48:N L52:N | 0.00 |
| canada | J-lens | 20,36 | 0 / 2 | berlin / russia / germany / beijing / nazi / moscow / civilization / city / canada / citie | — | — | L20:N L36:N | 0.44 |
| china | J-lens | 52,56,60 | 0 / 3 | also in german 'deutschland' and chinese '德国' | — | — | L52:N L56:N L60:N | 0.89 |
| italy | J-lens | 60 | 0 / 1 | and germania (italian/latin), along with the related tokens german and berlin | — | — | L60:N | 0.33 |
| poland | J-lens | 48,52 | 0 / 2 | the surrounding concepts are berlin, neighboring poland and austria, and the nazi era. | — | — | L48:N L52:N | 0.11 |
| russia | J-lens | 20,24,28,32,36,56,60 | 0 / 7 | the strongest cues are germany and russia with a wwii/nazi tint | — | — | L20:N L24:N L28:N L32:N L36:N L56:N L60:N | 0.67 |
| spain | J-lens | 56 | 0 / 1 | германии (russian, "of germany") and alemania (spanish) | — | — | L56:N | 0.56 |
| sweden | J-lens | 52 | 0 / 1 | nearby european countries (austria, sweden, poland) and berlin appear as weaker alternativ | — | — | L52:N | 0.00 |
| united_kingdom | J-lens | 32 | 0 / 1 | berlin/germany is the leading candidate, with the united kingdom and russia as alternative | — | — | L32:N | 0.78 |
| united_kingdom | ML-NLA RL600 | 60 | 0 / 1 | possibly translated german-english tourism or fact-page prose about berlin | — | — | L60:N | 0.33 |
| united_kingdom | s3d O-lens | 32 | 1 / 0 | germany is the capital of the united kingdom. | L32 n/a (gold control (dom p 0.092)) | L32: jlens | L32:X | 0.44 |

#### br-ent-cap-cairo: "The city of Cairo is the capital of" (gold egypt)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| algeria | J-lens | 52 | 0 / 1 | tunisia / morocco / saudi / libya / algeria / iraq | — | — | L52:N | 0.00 |
| brazil | J-lens | 20 | 0 / 1 | egypt / ethiopia / africa / civilization / kingdom / iraq / russia / kingdom / brazil / ca | — | — | L20:N | 0.44 |
| canada | J-lens | 20 | 0 / 1 | egypt / ethiopia / africa / civilization / kingdom / iraq / russia / kingdom / brazil / ca | — | — | L20:N | 0.44 |
| china | J-lens | 56,60 | 0 / 2 | the readouts include the country name in english, chinese (埃及) and arabic (مصر) | — | — | L56:N L60:N | 0.89 |
| ethiopia | J-lens | 20,24,28,32,36,40,44,48,52,56,60 | 0 / 11 | an african kingdom or civilization such as egypt or ethiopia | — | — | L20:N L24:N L28:N L32:N L36:N L40:N L44:N L48:N L52:N L56:N L60:N | 0.11 |
| ethiopia | s3d O-lens | 56 | 1 / 0 | 埃塞俄比亚 (egypt) and the capital is cairo. | L56 r=0.245/0.000 no | L56: jlens | L56:X | 0.00 |
| iraq | J-lens | 20,24,32,36,44,48,52 | 0 / 7 | egypt / ethiopia / africa / civilization / kingdom / iraq / russia | — | — | L20:N L24:N L32:N L36:N L44:N L48:N L52:N | 0.00 |
| libya | J-lens | 24,40,44,52,56 | 0 / 5 | with iraq, saudi arabia, libya, ethiopia and nigeria as nearby alternatives | — | — | L24:N L40:N L44:N L52:N L56:N | 0.00 |
| morocco | J-lens | 28,40,44,48,52 | 0 / 5 | with a 'kingdom' reading (e.g., saudi arabia, morocco) as an alternative | — | — | L28:N L40:N L44:N L48:N L52:N | 0.00 |
| nigeria | J-lens | 24,28,32,36,40,44,48,52 | 0 / 8 | with iraq, saudi arabia, libya, ethiopia and nigeria as nearby alternatives | — | — | L24:N L28:N L32:N L36:N L40:N L44:N L48:N L52:N | 0.22 |
| russia | J-lens | 20 | 0 / 1 | egypt / ethiopia / africa / civilization / kingdom / iraq / russia / kingdom | — | — | L20:N | 0.67 |
| saudi_arabia | J-lens | 24,28,32,36,44,48,52 | 0 / 7 | with iraq, saudi arabia, libya, ethiopia and nigeria as nearby alternatives | — | — | L24:N L28:N L32:N L36:N L44:N L48:N L52:N | 0.00 |
| syria | J-lens | 48 | 0 / 1 | ethiopia, morocco, tunisia, syria, iraq) as alternatives | — | — | L48:N | 0.00 |
| tunisia | J-lens | 40,44,48,52,56,60 | 0 / 6 | nearby alternatives drawn from north africa and the arab world (morocco, tunisia, libya, e | — | — | L40:N L44:N L48:N L52:N L56:N L60:N | 0.00 |
| united_kingdom | J-lens | 56 | 0 / 1 | the readouts include the country name in english, chinese (埃及) and arabic | — | — | L56:N | 0.78 |

#### br-ent-cap-madrid: "The city of Madrid is the capital of" (gold spain)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| argentina | J-lens | 20,24,48,52,56 | 0 / 5 | latin american nations like mexico, colombia, argentina, brazil | — | — | L20:N L24:N L48:N L52:N L56:N | 0.11 |
| belgium | J-lens | 52 | 0 / 1 | spain / france / portugal / belgium / europe | — | — | L52:N | 0.11 |
| brazil | J-lens | 20 | 0 / 1 | latin american nations like mexico, colombia, argentina, brazil | — | — | L20:N | 0.44 |
| china | J-lens | 40,56,60 | 0 / 3 | españa, and 西班牙, chinese for 'spain') converge on it | — | — | L40:N L56:N L60:N | 0.89 |
| colombia | J-lens | 20,24 | 0 / 2 | latin american nations like mexico, colombia, argentina, brazil | — | — | L20:N L24:N | 0.00 |
| france | J-lens | 52,56,60 | 0 / 3 | spain / france / portugal / belgium | — | — | L52:N L56:N L60:N | 0.22 |
| germany | J-lens | 52,56,60 | 0 / 3 | spain / france / portugal / belgium / europe / germany / italy | — | — | L52:N L56:N L60:N | 0.33 |
| italy | J-lens | 40,44,48,52 | 0 / 4 | with neighbors portugal, italy, and iberia as close alternatives | — | — | L40:N L44:N L48:N L52:N | 0.33 |
| mexico | J-lens | 20,24,28,32,36,44,48 | 0 / 7 | latin american nations like mexico, colombia, argentina, brazil | — | — | L20:N L24:N L28:N L32:N L36:N L44:N L48:N | 0.00 |
| portugal | J-lens | 24,36,40,44,48,52,56 | 0 / 7 | its cities (madrid, barcelona), the iberian peninsula and portugal, and spanish-speaking l | — | — | L24:N L36:N L40:N L44:N L48:N L52:N L56:N | 0.11 |
| united_kingdom | J-lens | 36,60 | 0 / 2 | neighboring or related countries such as portugal, mexico and england | — | — | L36:N L60:N | 0.78 |
| united_kingdom | ML-NLA RL600 | 56 | 0 / 1 | listing attractions with slightly non-native english phrasing | — | — | L56:N | 0.33 |
| united_states | J-lens | 20,24 | 0 / 2 | most likely a spanish-speaking or ibero-american country or city | — | — | L20:N L24:N | 0.44 |
| united_states | ML-NLA RL600 | 48 | 0 / 1 | formulaic definitions ("capital of the united states of america"), suggesting a factsheet | — | — | L48:N | 0.00 |
| united_states | s3d O-lens | 32 | 1 / 0 | it is also the capital of the united states. | L32 n/a (gold control (dom p 0.081)) | L32: none | L32:H | 0.44 |

#### br-ent-cap-moscow: "The city of Moscow is the capital of" (gold russia)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| canada | J-lens | 20,24,28,32,36,48,52 | 0 / 7 | russia / civilization / canada / cities / kingdom | — | — | L20:N L24:N L28:N L32:N L36:N L48:N L52:N | 0.44 |
| china | J-lens | 28,32,40,56,60 | 0 / 5 | followed by other very large nations (canada, china, india) | — | — | L28:N L32:N L40:N L56:N L60:N | 0.89 |
| france | J-lens | 60 | 0 / 1 | russian (россии/россия), french (russie) and arabic (روسيا) | — | — | L60:N | 0.22 |
| india | J-lens | 20,24,28,32 | 0 / 4 | moscow / america / india / russian | — | — | L20:N L24:N L28:N L32:N | 0.22 |
| poland | J-lens | 44,48,52 | 0 / 3 | with ukraine, poland and serbia as nearby alternatives | — | — | L44:N L48:N L52:N | 0.11 |
| romania | J-lens | 48 | 0 / 1 | nearby states such as poland, romania, and serbia | — | — | L48:N | 0.22 |
| serbia | J-lens | 44,48 | 0 / 2 | with ukraine, poland and serbia as nearby alternatives | — | — | L44:N L48:N | 0.11 |
| spain | J-lens | 52,56,60 | 0 / 3 | variants like russian, soviet, ussr, moscow, spanish 'rusia' | — | — | L52:N L56:N L60:N | 0.56 |
| ukraine | J-lens | 36,40,44,52,56 | 0 / 5 | with beijing, canada and ukraine appearing as related country alternatives | — | — | L36:N L40:N L44:N L52:N L56:N | 0.00 |
| united_kingdom | J-lens | 48,60 | 0 / 2 | possible continuations include "soviet union," "republic of ...," "united kingdom," | — | — | L48:N L60:N | 0.78 |
| united_states | J-lens | 20,24,32 | 0 / 3 | russia / civilization / canada / cities / kingdom / city / moscow / america | — | — | L20:N L24:N L32:N | 0.44 |
| united_states | s3d O-lens | 28 | 1 / 0 | 首都 (capital) of the usa. it is the largest country | L28 n/a (gold control (dom p 0.541)) | L28: jlens | L28:X | 0.44 |

#### br-ent-cap-nairobi: "The city of Nairobi is the capital of" (gold kenya)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| brazil | J-lens | 20 | 0 / 1 | russia / indonesia / kingdom / brazil / city | — | — | L20:N | 0.44 |
| china | J-lens | 56,60 | 0 / 2 | 肯尼亚 is chinese for "kenya" | — | — | L56:N L60:N | 0.89 |
| ethiopia | J-lens | 20,24,28,32,36,40,44,48,52,56,60 | 0 / 11 | most likely an african country or ancient kingdom/civilization (e.g., ethiopia, kenya) | — | — | L20:N L24:N L28:N L32:N L36:N L40:N L44:N L48:N L52:N L56:N L60:N | 0.11 |
| india | J-lens | 32 | 0 / 1 | kenya / africa / ethiopia / uganda / country / world / nigeria / indonesia / india / count | — | — | L32:N | 0.22 |
| indonesia | J-lens | 20,28,32 | 0 / 3 | africa / ethiopia / kenya / kingdom / civilization / russia / indonesia / kingdom | — | — | L20:N L28:N L32:N | 0.11 |
| nepal | J-lens | 36 | 0 / 1 | kenya / ethiopia / africa / uganda / tanzania / nairobi / nepal / country | — | — | L36:N | 0.00 |
| nigeria | J-lens | 24,28,32,36,44,48,52,56 | 0 / 8 | kenya / ethiopia / africa / rwanda / tanzania / uganda / nigeria / nairobi | — | — | L24:N L28:N L32:N L36:N L44:N L48:N L52:N L56:N | 0.22 |
| nigeria | s3d O-lens | 60 | 0 / 1 | the second-most populous after nigeria | — | — | L60:N | 0.00 |
| russia | J-lens | 20 | 0 / 1 | africa / ethiopia / kenya / kingdom / civilization / russia / indonesia | — | — | L20:N | 0.67 |
| rwanda | J-lens | 24,28,40,44 | 0 / 4 | or a neighbor such as ethiopia, rwanda, tanzania or uganda | — | — | L24:N L28:N L40:N L44:N | 0.00 |
| somalia | J-lens | 40,44,48,52,56 | 0 / 5 | kenya / uganda / tanzania / ethiopia / africa / nairobi / zambia / rwanda / somalia / coun | — | — | L40:N L44:N L48:N L52:N L56:N | 0.00 |
| tanzania | J-lens | 24,28,36,40,44,48,52,56,60 | 0 / 9 | a neighbor such as ethiopia, rwanda, tanzania or uganda | — | — | L24:N L28:N L36:N L40:N L44:N L48:N L52:N L56:N L60:N | 0.00 |
| tanzania | s3d O-lens | 52 | 0 / 1 | a) nairobi, kenya b) nairobi, tanzania | — | — | L52:N | 0.00 |
| uganda | J-lens | 24,28,32,36,40,44,48,52,56,60 | 0 / 10 | or a neighbor such as ethiopia, rwanda, tanzania or uganda | — | — | L24:N L28:N L32:N L36:N L40:N L44:N L48:N L52:N L56:N L60:N | 0.00 |
| uganda | s3d O-lens | 32 | 0 / 1 | kenya is the largest country in africa by area, but uganda has the largest population. | — | — | L32:N | 0.00 |
| united_kingdom | ML-NLA RL600 | 52 | 0 / 1 | simple, slightly awkward non-native english, present tense declaratives | — | — | L52:N | 0.33 |
| united_kingdom | s3d O-lens | 44 | 0 / 1 | the official language is swahili and english. | — | — | L44:N | 0.44 |
| zambia | J-lens | 40,44,48,52 | 0 / 4 | kenya / uganda / tanzania / ethiopia / africa / nairobi / zambia | — | — | L40:N L44:N L48:N L52:N | 0.00 |

#### br-ent-cap-ottawa: "The city of Ottawa is the capital of" (gold canada)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| australia | J-lens | 48,52 | 0 / 2 | saskatchewan / ottawa / united / australia | — | — | L48:N L52:N | 0.00 |
| france | s3d O-lens | 28,60 | 0 / 2 | quebec is the largest french-speaking province | — | — | L28:N L60:N | 0.11 |
| nigeria | J-lens | 32 | 0 / 1 | other country names such as nigeria and the united kingdom appear as weaker alternatives | — | — | L32:N | 0.22 |
| portugal | J-lens | 60 | 0 / 1 | the spanish/portuguese spelling "canadá" points the same way | — | — | L60:N | 0.11 |
| spain | J-lens | 60 | 0 / 1 | the spanish/portuguese spelling "canadá" points the same way | — | — | L60:N | 0.56 |
| united_kingdom | J-lens | 32 | 0 / 1 | other country names such as nigeria and the united kingdom appear as weaker alternatives | — | — | L32:N | 0.78 |
| united_kingdom | s3d O-lens | 48,52 | 1 / 1 | 英国 (canada's capital city) is ottawa. | L48 r=0.195/0.390 no | L48: none | L48:H L52:N | 0.44 |
| united_states | s3d O-lens | 44 | 0 / 1 | it is the second largest country in north america. | — | — | L44:N | 0.44 |

#### br-ent-cap-paris: "The city of Paris is the capital of" (gold france)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| belgium | J-lens | 40,44,48,52,56 | 0 / 5 | nearby european countries like italy, spain and belgium appear as weaker alternatives | — | — | L40:N L44:N L48:N L52:N L56:N | 0.11 |
| canada | J-lens | 20,24,28,36 | 0 / 4 | city / paris / france / civilization / cities / canada / kings | — | — | L20:N L24:N L28:N L36:N | 0.44 |
| canada | s3d O-lens | 28 | 0 / 1 | the capital of canada is | — | — | L28:N | 0.00 |
| china | J-lens | 32,40,56,60 | 0 / 4 | the chinese particle 了吗 means "has it ...?" and marks a yes/no question | — | — | L32:N L40:N L56:N L60:N | 0.89 |
| germany | J-lens | 48,56,60 | 0 / 3 | neighbouring european countries (spain, belgium, italy, germany) and related terms | — | — | L48:N L56:N L60:N | 0.33 |
| germany | s3d O-lens | 36 | 0 / 1 | the capital of germany is berlin. | — | — | L36:N | 0.00 |
| greece | s3d O-lens | 40 | 0 / 1 | "paris" comes from the greek mythological figure paris | — | — | L40:N | 0.00 |
| italy | J-lens | 40,44,48,52,56,60 | 0 / 6 | nearby european countries like italy, spain and belgium appear as weaker alternatives | — | — | L40:N L44:N L48:N L52:N L56:N L60:N | 0.33 |
| japan | s3d O-lens | 28,36 | 0 / 2 | just as tokyo is for japan | — | — | L28:N L36:N | 0.00 |
| russia | J-lens | 20,56,60 | 0 / 3 | city / paris / france / civilization / cities / canada / kings / kingdom / russia / africa | — | — | L20:N L56:N L60:N | 0.67 |
| spain | J-lens | 40,44,48,56,60 | 0 / 5 | nearby european countries like italy, spain and belgium appear as weaker alternatives | — | — | L40:N L44:N L48:N L56:N L60:N | 0.56 |
| united_kingdom | J-lens | 60 | 0 / 1 | the concept of france as a country (in english, chinese 法国 | — | — | L60:N | 0.78 |
| united_kingdom | s3d O-lens | 48 | 0 / 1 | it is spelled **paris**, not "paris" in english, but the prompt asks | — | — | L48:N | 0.44 |
| united_states | J-lens | 24 | 0 / 1 | canada / city / kingdoms / country / america / monarchy | — | — | L24:N | 0.44 |
| united_states | s3d O-lens | 32 | 1 / 0 | 首都 of the united states is paris. | L32 n/a (gold control (logreg p 0.428)) | L32: none | L32:H | 0.44 |

#### br-ent-cap-rome: "The city of Rome is the capital of" (gold italy)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| argentina | J-lens | 28 | 0 / 1 | argentina probably shows up through its strong italian immigrant ties. | — | — | L28:N | 0.11 |
| brazil | J-lens | 20 | 0 / 1 | with other countries like america and brazil as weaker alternative guesses | — | — | L20:N | 0.44 |
| china | J-lens | 52,56,60 | 0 / 3 | the tokens include italia, the chinese 意大利 ("italy"), and rome | — | — | L52:N L56:N L60:N | 0.89 |
| france | J-lens | 40,52 | 0 / 2 | other european countries (spain, france, romania) as weaker alternatives | — | — | L40:N L52:N | 0.22 |
| germany | J-lens | 52,60 | 0 / 2 | italy / italian / romania / france / 意大利 / rome / germany / spain | — | — | L52:N L60:N | 0.33 |
| greece | J-lens | 44,48 | 0 / 2 | nearby mediterranean and european countries (spain, greece, romania) appear as weaker alte | — | — | L44:N L48:N | 0.00 |
| romania | J-lens | 24,40,44,48,52,56 | 0 / 6 | italia / roma / romania / monarchy | — | — | L24:N L40:N L44:N L48:N L52:N L56:N | 0.22 |
| russia | J-lens | 56,60 | 0 / 2 | chinese "意大利," and russian "италии" | — | — | L56:N L60:N | 0.67 |
| spain | J-lens | 40,44,48,52 | 0 / 4 | other european countries (spain, france, romania) as weaker alternatives | — | — | L40:N L44:N L48:N L52:N | 0.56 |
| united_kingdom | J-lens | 56 | 0 / 1 | representing the country italy (in english, italian "italia," | — | — | L56:N | 0.78 |
| united_kingdom | ML-NLA RL600 | 36,56 | 0 / 2 | slightly awkward translated-from-italian english typical of tourism blogs | — | — | L36:N L56:N | 0.33 |
| united_kingdom | s3d O-lens | 56 | 0 / 1 | in english, it's just "rome". | — | — | L56:N | 0.44 |
| united_states | J-lens | 20,32 | 0 / 2 | with other countries like america and brazil as weaker alternative guesses | — | — | L20:N L32:N | 0.44 |
| vatican_city | J-lens | 20,24,28,36,56 | 0 / 5 | most strongly rome/italy (with the vatican and ancient civilization as related senses) | — | — | L20:N L24:N L28:N L36:N L56:N | 0.00 |
| vatican_city | ML-NLA RL600 | 32,36,40,48,52,56,60 | 0 / 7 | rome facts — capital of italy, lazio region, vatican city, seat of government | — | — | L32:N L36:N L40:N L48:N L52:N L56:N L60:N | 0.00 |
| vatican_city | s3d O-lens | 24,28,40,44,48 | 0 / 5 | vatican city is the center of the catholic church, led by the pope. | — | — | L24:N L28:N L40:N L44:N L48:N | 0.00 |

#### br-ent-cap-tokyo: "The city of Tokyo is the capital of" (gold japan)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| brazil | J-lens | 20,24,32 | 0 / 3 | japan / civilization / tokyo / russia / indonesia / kingdom / city / america / canada / br | — | — | L20:N L24:N L32:N | 0.44 |
| brazil | s3d O-lens | 32 | 0 / 1 | the capital of brazil is brasília. | — | — | L32:N | 0.00 |
| cambodia | J-lens | 24 | 0 / 1 | korea / kingdoms / kyoto / china / cambodia | — | — | L24:N | 0.00 |
| canada | J-lens | 20,60 | 0 / 2 | japan / civilization / tokyo / russia / indonesia / kingdom / city / america / canada / br | — | — | L20:N L60:N | 0.44 |
| china | J-lens | 24,28,32,36,40,44,48,52,56,60 | 0 / 10 | japan / indonesia / tokyo / civilization / brazil / korea / kingdoms / kyoto / china / cam | — | — | L24:N L28:N L32:N L36:N L40:N L44:N L48:N L52:N L56:N L60:N | 0.89 |
| china | s3d O-lens | 48 | 0 / 1 | is the capital of which country? a) china b) japan | — | — | L48:N | 0.00 |
| france | s3d O-lens | 32 | 0 / 1 | the capital of france is paris. | — | — | L32:N | 0.11 |
| india | J-lens | 32 | 0 / 1 | japan, indonesia, china, india, usa, brazil, united kingdom | — | — | L32:N | 0.22 |
| indonesia | J-lens | 20,24,28,32,36,40,44,48,52,56,60 | 0 / 11 | japan / civilization / tokyo / russia / indonesia / kingdom | — | — | L20:N L24:N L28:N L32:N L36:N L40:N L44:N L48:N L52:N L56:N L60:N | 0.11 |
| russia | J-lens | 20 | 0 / 1 | japan / civilization / tokyo / russia / indonesia | — | — | L20:N | 0.67 |
| united_kingdom | J-lens | 32 | 0 / 1 | (japan, indonesia, china, india, usa, brazil, united kingdom) | — | — | L32:N | 0.78 |
| united_states | J-lens | 20,32 | 0 / 2 | japan / civilization / tokyo / russia / indonesia / kingdom / city / america | — | — | L20:N L32:N | 0.44 |
| united_states | s3d O-lens | 32 | 0 / 1 | but you asked for the capital of america. | — | — | L32:N | 0.44 |

### country_landmark

#### br-ent-lmk-big: "The Big Ben is located in the country of" (gold england)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| australia | J-lens | 60 | 0 / 1 | with other english-speaking countries (australia, canada) and france as weaker alternative | — | — | L60:N | 0.00 |
| canada | J-lens | 36,48,60 | 0 / 3 | canada appears as a weaker alternative among english-speaking or commonwealth countries. | — | — | L36:N L48:N L60:N | 0.17 |
| china | J-lens | 20,24,28,40,44,56 | 0 / 6 | the chinese 国的 means 'of the country' | — | — | L20:N L24:N L28:N L40:N L44:N L56:N | 1.00 |
| china | ML-NLA RL600 | 40,44 | 0 / 2 | machine-translated chinese-english informational register | — | — | L40:N L44:N | 0.00 |
| china | NLA-RL L42 | 42 | 0 / 1 | trivia/quiz question format in chinese ("答案 (d) ...") signals a knowledge-quiz style answe | — | — | L42:N | 0.00 |
| finland | s3d O-lens | 24 | 0 / 1 | the capital of the kingdom is **finland**. the | — | — | L24:N | 0.17 |
| france | J-lens | 60 | 0 / 1 | other english-speaking countries (australia, canada) and france as weaker alternatives | — | — | L60:N | 0.17 |
| france | s3d O-lens | 20,32 | 1 / 1 | 国家 (country) of france. | L32 n/a (gold control (dom p 0.072)) | L32: none | L20:N L32:H | 0.83 |
| germany | ML-NLA RL600 | 28 | 1 / 0 | berlin is the capital of germany; expects country name | L28 n/a (validity gate (dom AUROC 0.828)) | L28: none | L28:H | 0.00 |
| italy | ML-NLA RL600 | 24 | 1 / 0 |  | L24 r=0.380/0.440 no | L24: none | L24:H | 0.50 |
| japan | s3d O-lens | 32 | 1 / 0 | the country is **japan**. it is a | L32 n/a (gold control (dom p 0.072)) | L32: none | L32:H | 0.50 |
| russia | s3d O-lens | 24 | 0 / 1 | country of origin: the country where the flag comes from is **russia**. | — | — | L24:N | 0.00 |
| scotland | J-lens | 40,44,48 | 0 / 3 | alternatives in the readouts include britain/uk (英国 is chinese for "uk/britain"), scotland | — | — | L40:N L44:N L48:N | 0.00 |
| sweden | s3d O-lens | 56 | 1 / 0 | the answer is **с. швеция** (sweden). | L56 r=0.500/0.380 no | L56: none | L56:H | 0.17 |
| switzerland | ML-NLA RL600 | 20 | 1 / 0 | bern is the capital of switzerland, so geographic completion must name that country | L20 n/a (gold control (dom p 0.078)) | L20: none | L20:H | 0.00 |
| united_states | NLA-RL L42 | 42 | 0 / 1 | country or nation name noun phrase (e.g., "britain," "the united states," "england") | — | — | L42:N | 0.17 |
| vatican_city | ML-NLA RL600 | 24 | 0 / 1 | almost certainly "italy" (or "vatican city"), followed by more sentence fragments | — | — | L24:N | 0.17 |

#### br-ent-lmk-christ: "The Christ the Redeemer is located in the country of" (gold brazil)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| argentina | J-lens | 40,44,48,56,60 | 0 / 5 | brazil strongly favored over neighbors like argentina, mexico, and colombia | — | — | L40:N L44:N L48:N L56:N L60:N | 0.17 |
| argentina | ML-NLA RL600 | 40,48 | 0 / 2 | likely followed by location detail or a multiple-choice list (brazil, argentina, peru, chi | — | — | L40:N L48:N | 0.17 |
| argentina | s3d O-lens | 44,52,56,60 | 0 / 4 | the country that is known as the "land of the sun"? **a. argentina** | — | — | L44:N L52:N L56:N L60:N | 0.00 |
| canada | s3d O-lens | 56 | 0 / 1 | a) canada b) mexico c) brazil d) spain | — | — | L56:N | 0.33 |
| chile | ML-NLA RL600 | 20,40 | 0 / 2 | demanding a proper noun country name (likely chile) | — | — | L20:N L40:N | 0.00 |
| chile | s3d O-lens | 32,56,60 | 0 / 3 | 国家 (country) of chile. the correct answer is **peru**. | — | — | L32:N L56:N L60:N | 0.00 |
| china | J-lens | 20,24,28 | 0 / 3 | 国的 is chinese for "of the country/nation" | — | — | L20:N L24:N L28:N | 1.00 |
| colombia | J-lens | 40 | 0 / 1 | brazil strongly favored over neighbors like argentina, mexico, and colombia | — | — | L40:N | 0.00 |
| colombia | ML-NLA RL600 | 24 | 1 / 0 | demanding a south american country name, most plausibly colombia (the actual answer) | L24 n/a (gold control (logreg p 0.050)) | L24: none | L24:H | 0.00 |
| costa_rica | s3d O-lens | 60 | 0 / 1 | 巴西 (brazil) and costa rica. | — | — | L60:N | 0.00 |
| el_salvador | s3d O-lens | 20 | 1 / 0 | 国家 (country) of el salvador. but, as | L20 n/a (gold control (logreg p 0.159)) | L20: none | L20:H | 0.00 |
| france | s3d O-lens | 20,48 | 1 / 1 | the country where it is located is france. | L20 n/a (gold control (logreg p 0.159)) | L20: none | L20:H L48:N | 0.83 |
| italy | ML-NLA RL600 | 28 | 1 / 0 | the colosseum is in rome, so the country is italy | L28 n/a (validity gate (dom AUROC 0.827)) | L28: s3d | L28:X | 0.50 |
| italy | s3d O-lens | 24,32,48 | 2 / 1 | italy is the country where this town is located. | L24 n/a (gold control (logreg p 0.050)); L32 n/a (validity gate (dom AUROC 0.756)) | L24: mlnla-rl600; L32: mlnla-rl600 | L24:X L32:X L48:N | 0.33 |
| japan | s3d O-lens | 48 | 0 / 1 | brazil / france / japan / italy. the answer is **brazil**. | — | — | L48:N | 0.50 |
| mexico | J-lens | 40,44,48 | 0 / 3 | brazil strongly favored over neighbors like argentina, mexico, and colombia | — | — | L40:N L44:N L48:N | 0.00 |
| mexico | s3d O-lens | 24,28,32,48,56 | 1 / 4 | the country located in this region is **mexico**. it | L32 n/a (validity gate (dom AUROC 0.711)) | L32: none | L24:N L28:N L32:H L48:N L56:N | 0.17 |
| panama | ML-NLA RL600 | 28 | 0 / 1 | so the country is italy; expect "italy" rather than "panama." | — | — | L28:N | 0.00 |
| panama | s3d O-lens | 60 | 0 / 1 | /panama (country)** — wait, the blank is for a *country* that begins with **i**. | — | — | L60:N | 0.00 |
| peru | ML-NLA RL600 | 40 | 0 / 1 | a multiple-choice list (brazil, argentina, peru, chile) | — | — | L40:N | 0.17 |
| peru | s3d O-lens | 24,32,36,56,60 | 3 / 2 | peru, a country in south america. however, the capital is named after | L24 n/a (gold control (logreg p 0.050)); L32 n/a (gold control (logreg p 0.090)); L36 n/a (gold control (dom p 0.140)) | L24: none; L32: none; L36: none | L24:H L32:H L36:H L56:N L60:N | 0.50 |
| portugal | J-lens | 60 | 0 / 1 | a world cup winner, or a country where portuguese is spoken | — | — | L60:N | 0.00 |
| portugal | s3d O-lens | 56 | 0 / 1 | the answer choices provided were in portuguese | — | — | L56:N | 0.00 |
| spain | ML-NLA RL600 | 24 | 0 / 1 | expecting location facts, capital status, population, or spanish-colonial trivia | — | — | L24:N | 0.33 |
| spain | s3d O-lens | 56 | 0 / 1 | a) canada b) mexico c) brazil d) spain | — | — | L56:N | 0.33 |
| switzerland | s3d O-lens | 20 | 0 / 1 | 地点 of switzerland. if you have been to a place, you must answer with that country. | — | — | L20:N | 0.17 |
| united_kingdom | J-lens | 20,24,28,36 | 0 / 4 | probably the next word naming a sovereign entity such as "united kingdom" or "united natio | — | — | L20:N L24:N L28:N L36:N | 0.83 |
| united_states | J-lens | 40,44,48,56,60 | 0 / 5 | specifically a latin american country | — | — | L40:N L44:N L48:N L56:N L60:N | 0.00 |
| united_states | ML-NLA RL600 | 20,24,44,48 | 0 / 4 | is located in the continent south america | — | — | L20:N L24:N L44:N L48:N | 0.33 |
| united_states | NLA-RL L42 | 42 | 0 / 1 | cultural context about a south american or latin american nation | — | — | L42:N | 0.17 |
| united_states | s3d O-lens | 24,28,32,36,40,44,48,52,56,60 | 0 / 10 | peru, a country in south america. | — | — | L24:N L28:N L32:N L36:N L40:N L44:N L48:N L52:N L56:N L60:N | 0.83 |
| venezuela | J-lens | 44 | 0 / 1 | the alternatives (argentina, mexico, venezuela) are competing latin american answers | — | — | L44:N | 0.00 |

#### br-ent-lmk-colosseum: "The Colosseum is located in the country of" (gold italy)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| china | J-lens | 24,28 | 0 / 2 | the chinese '国的' ('of the country/national') reinforces the abstract 'country/nation' mean | — | — | L24:N L28:N | 1.00 |
| france | J-lens | 56,60 | 0 / 2 | greece, france and romania are runner-up candidates | — | — | L56:N L60:N | 0.17 |
| france | s3d O-lens | 24,28,32,48,52 | 2 / 3 | in the nation of **france** (though the prompt says "island," | L24 n/a (gold control (dom p 0.098)); L28 n/a (gold control (dom p 0.109)) | L24: none; L28: none | L24:H L28:H L32:N L48:N L52:N | 0.83 |
| germany | s3d O-lens | 48 | 0 / 1 | a) italy b) france c) spain d) germany | — | — | L48:N | 0.33 |
| greece | J-lens | 48,56 | 0 / 2 | with greece, romania and rome as the nearby alternatives | — | — | L48:N L56:N | 0.00 |
| greece | ML-NLA RL600 | 60 | 0 / 1 | or a list of options like "italy / greece / spain" | — | — | L60:N | 0.00 |
| greece | s3d O-lens | 24 | 1 / 0 | it is located in a country called **greece | L24 n/a (gold control (dom p 0.098)) | L24: none | L24:H | 0.00 |
| romania | J-lens | 40,44,48,56 | 0 / 4 | romania appears only as a surface-similar neighbor of "roma". | — | — | L40:N L44:N L48:N L56:N | 0.00 |
| spain | J-lens | 60 | 0 / 1 | other european countries like france and spain appear as weaker alternatives | — | — | L60:N | 0.17 |
| spain | ML-NLA RL600 | 60 | 0 / 1 | or a list of options like "italy / greece / spain" | — | — | L60:N | 0.33 |
| spain | s3d O-lens | 48 | 0 / 1 | a) italy b) france c) spain d) germany | — | — | L48:N | 0.33 |
| united_kingdom | J-lens | 24,28,36 | 0 / 3 | about to name a state (e.g., 'united kingdom/united nations' or 'the nation of...') | — | — | L24:N L28:N L36:N | 0.83 |
| united_states | s3d O-lens | 24 | 0 / 1 | he has never left his home country, the united states. | — | — | L24:N | 0.83 |
| vatican_city | J-lens | 40,44 | 0 / 2 | romans / roman / roma / vatican / romania | — | — | L40:N L44:N | 0.00 |
| vatican_city | ML-NLA RL600 | 24 | 0 / 1 | roman empire, vatican, seven hills likely referenced next. | — | — | L24:N | 0.17 |
| vatican_city | s3d O-lens | 52 | 0 / 1 | home of the colosseum and vatican city | — | — | L52:N | 0.00 |

#### br-ent-lmk-eiffel: "The Eiffel Tower is located in the country of" (gold france)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| australia | s3d O-lens | 24 | 0 / 1 | "australia is famous for kangaroos and..." and write it correctly. | — | — | L24:N | 0.00 |
| belgium | J-lens | 48 | 0 / 1 | most likely "france" (with belgium as a runner-up) | — | — | L48:N | 0.00 |
| brazil | s3d O-lens | 28 | 1 / 0 | the answer is **brazil**. the puzzle is a geography riddle: | L28 n/a (gold control (dom p 0.221)) | L28: none | L28:H | 0.33 |
| china | J-lens | 20,28,40 | 0 / 3 | country and the chinese '国的' ('of the country/nation') | — | — | L20:N L28:N L40:N | 1.00 |
| finland | s3d O-lens | 32 | 1 / 0 | country of **finland**, not the city. | L32 n/a (validity gate (dom AUROC 0.815)) | L32: none | L32:H | 0.17 |
| germany | s3d O-lens | 24 | 0 / 1 | germany is a country located in europe. its capital is | — | — | L24:N | 0.33 |
| italy | J-lens | 40,44 | 0 / 2 | other european countries such as italy and spain | — | — | L40:N L44:N | 0.00 |
| italy | ML-NLA RL600 | 20,32 | 2 / 0 | mont blanc, highest peak in the alps, straddles the france–italy border | L20 n/a (validity gate (dom AUROC 0.848)); L32 n/a (validity gate (dom AUROC 0.756)) | L20: none; L32: none | L20:H L32:H | 0.50 |
| italy | s3d O-lens | 48 | 0 / 1 | which country is represented by the clue "city of light"?** a) italy b) | — | — | L48:N | 0.33 |
| peru | s3d O-lens | 28 | 1 / 0 | the country is **peru**. (located in south america) | L28 n/a (gold control (dom p 0.221)) | L28: none | L28:H | 0.50 |
| spain | J-lens | 40,44,60 | 0 / 3 | other european countries such as italy and spain | — | — | L40:N L44:N L60:N | 0.17 |
| spain | ML-NLA RL600 | 28 | 1 / 0 | barcelona is in spain; nearby cities, language (spanish/catalan) | L28 n/a (validity gate (dom AUROC 0.807)) | L28: none | L28:H | 0.33 |
| united_kingdom | J-lens | 20,28,32,36 | 0 / 4 | likely the passage refers to a specific state such as the united kingdom or united nations | — | — | L20:N L28:N L32:N L36:N | 0.83 |
| united_states | s3d O-lens | 28 | 0 / 1 | the country is **peru**. (located in south america) | — | — | L28:N | 0.83 |

#### br-ent-lmk-great: "The Great Pyramid of Giza is located in the country of" (gold egypt)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| brazil | s3d O-lens | 32 | 0 / 1 | you are in south america, specifically brazil. | — | — | L32:N | 0.33 |
| china | J-lens | 20,28,40,44,56,60 | 0 / 6 | the chinese "国的" ("of the country") suggests the passage concerns a named political entity | — | — | L20:N L28:N L40:N L44:N L56:N L60:N | 1.00 |
| ethiopia | J-lens | 40,44 | 0 / 2 | the other countries (ethiopia, morocco, qatar) are nearby middle eastern and african alter | — | — | L40:N L44:N | 0.00 |
| france | s3d O-lens | 24 | 1 / 0 | located in france. it is a country located in europe. | L24 r=0.918/0.857 no | L24: none | L24:H | 0.83 |
| iraq | J-lens | 28 | 0 / 1 | with iraq surfacing as one candidate | — | — | L28:N | 0.00 |
| italy | ML-NLA RL600 | 24 | 1 / 0 | final tokens "rome is located in the country of" form an incomplete prepositional phrase d | L24 r=0.980/0.959 no | L24: s3d | L24:X | 0.50 |
| italy | s3d O-lens | 20,28,32 | 1 / 2 | 意大利 (italy), and its capital is paris. | L32 n/a (validity gate (dom AUROC 0.756)) | L32: none | L20:N L28:N L32:H | 0.33 |
| jordan | ML-NLA RL600 | 20 | 1 / 0 | geographic knowledge required: petra is in jordan; the jordan river runs there | L20 n/a (concept not a bank class) | L20: none | L20:H | 0.00 |
| mexico | s3d O-lens | 28,32,44 | 2 / 1 | so, the answer is mexico. | L28 n/a (validity gate (dom AUROC 0.749)); L32 n/a (validity gate (dom AUROC 0.711)) | L28: none; L32: none | L28:H L32:H L44:N | 0.17 |
| morocco | J-lens | 40 | 0 / 1 | the other countries (ethiopia, morocco, qatar) are nearby middle eastern and african alter | — | — | L40:N | 0.00 |
| peru | s3d O-lens | 20 | 0 / 1 | peru. the king lives in the same country as the ancient ruins, | — | — | L20:N | 0.50 |
| qatar | J-lens | 40 | 0 / 1 | the other countries (ethiopia, morocco, qatar) are nearby middle eastern and african alter | — | — | L40:N | 0.00 |
| spain | s3d O-lens | 24 | 0 / 1 | 国家 (country) of spain, and a city in that country. most people live in | — | — | L24:N | 0.33 |
| united_kingdom | J-lens | 20,32 | 0 / 2 | suggests the passage concerns a named political entity such as the united nations or unite | — | — | L20:N L32:N | 0.83 |
| united_states | s3d O-lens | 28,32 | 0 / 2 | here is how the united states fits into that description | — | — | L28:N L32:N | 0.83 |

#### br-ent-lmk-sydney: "The Sydney Opera House is located in the country of" (gold australia)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| argentina | J-lens | 44,48 | 0 / 2 | weaker alternatives such as canada and argentina, along with the generic token "country," | — | — | L44:N L48:N | 0.17 |
| argentina | ML-NLA RL600 | 24 | 1 / 0 | geographic knowledge required: buenos aires is in argentina; also associated with rio de l | L24 n/a (gold control (dom p 0.050)) | L24: none | L24:H | 0.17 |
| brazil | s3d O-lens | 48 | 0 / 1 | what country is this? a) australia b) brazil c) canada d | — | — | L48:N | 0.33 |
| canada | J-lens | 36,40,44,48,60 | 0 / 5 | australia and canada show up as the likely candidate answers. | — | — | L36:N L40:N L44:N L48:N L60:N | 0.17 |
| canada | ML-NLA RL600 | 20 | 1 / 0 | requiring a country name — almost certainly "canada", followed by a period | L20 r=0.396/0.625 no | L20: s3d | L20:X | 0.00 |
| canada | s3d O-lens | 20,28,48,52,56 | 1 / 4 | country: it is in **canada | L28 n/a (gold control (dom p 0.253)) | L28: none | L20:N L28:H L48:N L52:N L56:N | 0.33 |
| china | J-lens | 24,28,32,40,44,48,56,60 | 0 / 8 | 国的 and 国家的 mean "country's" / "national" in chinese | — | — | L24:N L28:N L32:N L40:N L44:N L48:N L56:N L60:N | 1.00 |
| france | s3d O-lens | 20,24,28 | 2 / 1 | 国家 (country) of france. the country's flag is blue, white, | L20 r=0.979/1.000 fires; L28 n/a (gold control (dom p 0.253)) | L20: none; L28: none | L20:P L24:N L28:H | 0.83 |
| japan | s3d O-lens | 52,56 | 0 / 2 | a) new zealand b) australia c) canada d) japan | — | — | L52:N L56:N | 0.50 |
| netherlands | s3d O-lens | 28 | 1 / 0 | the country is called **holland** (though technically it's part of the netherlands) | L28 n/a (gold control (dom p 0.253)) | L28: none | L28:H | 0.00 |
| new_zealand | s3d O-lens | 52 | 0 / 1 | which country is the location of this song? a) new zealand b) australia | — | — | L52:N | 0.00 |
| sweden | s3d O-lens | 24 | 0 / 1 | the answer is that sweden is a kingdom located in northern europe | — | — | L24:N | 0.17 |
| switzerland | s3d O-lens | 20 | 0 / 1 | where is switzerland located?** the country is named | — | — | L20:N | 0.17 |
| united_kingdom | J-lens | 28 | 0 / 1 | possibly part of a name like "united kingdom" or "united nations." | — | — | L28:N | 0.83 |
| united_kingdom | s3d O-lens | 20,32 | 1 / 1 | he is in the city of **london**, which is the capital of england, | L32 n/a (gold control (dom p 0.288)) | L32: jlens | L20:N L32:X | 0.17 |
| united_states | ML-NLA RL600 | 24 | 0 / 1 | also associated with rio de la plata, tango, and south america | — | — | L24:N | 0.33 |
| united_states | s3d O-lens | 44 | 0 / 1 | 澳大利亚 (australia), not america. | — | — | L44:N | 0.83 |
| wales | ML-NLA RL600 | 28,32 | 0 / 2 | answer must be "australia," possibly with state "new south wales." | — | — | L28:N L32:N | 0.00 |
| wales | s3d O-lens | 36 | 0 / 1 | he lives in the state known as "new south wales." | — | — | L36:N | 0.00 |

#### br-ent-lmk-taj: "The Taj Mahal is located in the country of" (gold india)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| afghanistan | J-lens | 28 | 0 / 1 | likely right before or while naming a specific nation, with afghanistan as a leading candi | — | — | L28:N | 0.00 |
| bangladesh | J-lens | 40 | 0 / 1 | nearby south asian associations like pakistan, bangladesh, hindi and hindu | — | — | L40:N | 0.00 |
| bolivia | s3d O-lens | 32 | 1 / 0 | 美国 (usa) in the country of **bolivia**. | L32 n/a (gold control (dom p 0.061)) | L32: none | L32:H | 0.00 |
| canada | s3d O-lens | 28 | 1 / 0 | 地点 (country) of canada. they have a specific, unique currency | L28 n/a (gold control (dom p 0.082)) | L28: none | L28:H | 0.33 |
| china | J-lens | 20,24,28,40,44 | 0 / 5 | "国的" is chinese for "of the country" | — | — | L20:N L24:N L28:N L40:N L44:N | 1.00 |
| egypt | s3d O-lens | 28,32,36,48,52 | 4 / 1 | /in which country it is located? **egypt**. specifically, it is in the country | L28 n/a (gold control (dom p 0.082)); L32 n/a (gold control (dom p 0.061)); L36 n/a (gold control (dom p 0.081)); L48 r=0.960/1.000 no | L28: none; L32: none; L36: none; L48: none | L28:H L32:H L36:H L48:H L52:N | 0.00 |
| fiji | s3d O-lens | 24 | 1 / 0 | the correct answer is **fiji**. | L24 r=0.300/0.840 no | L24: none | L24:H | 0.00 |
| france | s3d O-lens | 20,24,32,52 | 1 / 3 | the country where the game takes place is **france**. | L32 n/a (gold control (dom p 0.061)) | L32: none | L20:N L24:N L32:H L52:N | 0.83 |
| germany | s3d O-lens | 52 | 0 / 1 | a) england b) india c) germany d) france | — | — | L52:N | 0.33 |
| indonesia | J-lens | 44 | 0 / 1 | neighboring south and southeast asian candidates (pakistan, iran, indonesia, 'chinese/chin | — | — | L44:N | 0.00 |
| iran | J-lens | 44,48 | 0 / 2 | neighboring south and southeast asian candidates (pakistan, iran, indonesia, 'chinese/chin | — | — | L44:N L48:N | 0.00 |
| japan | s3d O-lens | 24,48 | 2 / 0 | the country where it is located is called japan. | L24 r=0.740/0.080 no; L48 r=0.140/0.080 no | L24: none; L48: none | L24:H L48:H | 0.50 |
| jordan | s3d O-lens | 28 | 1 / 0 | it is located in the country of **jordan**. | L28 n/a (concept not a bank class) | L28: none | L28:H | 0.00 |
| mongolia | s3d O-lens | 20 | 0 / 1 | the country that hosts the school is called mongolia. | — | — | L20:N | 0.00 |
| nepal | ML-NLA RL600 | 20 | 0 / 1 | likely answer is nepal (or india). | — | — | L20:N | 0.00 |
| pakistan | J-lens | 40,44,48,56,60 | 0 / 5 | with nearby south asian associations like pakistan, bangladesh, hindi and hindu | — | — | L40:N L44:N L48:N L56:N L60:N | 0.00 |
| pakistan | ML-NLA RL600 | 24 | 1 / 0 | "karachi is located in the country of" — an incomplete prepositional phrase requiring a co | L24 r=0.240/0.140 no | L24: none | L24:H | 0.00 |
| pakistan | s3d O-lens | 60 | 0 / 1 | which south asian country is famous for the taj mahal? a. pakistan b. india | — | — | L60:N | 0.00 |
| peru | ML-NLA RL600 | 28,32 | 2 / 0 | puno is in peru, on lake titicaca's shores in the andes | L28 n/a (gold control (dom p 0.082)); L32 n/a (gold control (dom p 0.061)) | L28: s3d; L32: s3d | L28:X L32:X | 0.17 |
| peru | s3d O-lens | 24,28,32,36 | 3 / 1 | 国家 (country) of peru. the most famous city in this country is | L28 n/a (gold control (dom p 0.082)); L32 n/a (gold control (dom p 0.061)); L36 n/a (gold control (dom p 0.081)) | L28: mlnla-rl600; L32: mlnla-rl600; L36: mlnla-rl600 | L24:N L28:X L32:X L36:X | 0.50 |
| turkey | s3d O-lens | 48 | 1 / 0 | in the country of **turkey**. | L48 r=0.920/0.400 no | L48: none | L48:H | 0.00 |
| united_kingdom | J-lens | 20 | 0 / 1 | likely a phrase such as "united nations", "united kingdom", or national greatness | — | — | L20:N | 0.83 |
| united_kingdom | s3d O-lens | 52 | 0 / 1 | located in the country of ___? a) england b) india c) germany | — | — | L52:N | 0.17 |
| united_states | ML-NLA RL600 | 28,32 | 0 / 2 | most plausibly "peru" (possibly "peru, in south america") | — | — | L28:N L32:N | 0.33 |
| united_states | s3d O-lens | 28,32,36 | 0 / 3 | 美国 (the united states). it is located in the country of **jordan**. | — | — | L28:N L32:N L36:N | 0.83 |

### musical_instrument

#### br-ent-mus-armstrong: "The musician Louis Armstrong is famous for playing the" (gold trumpet)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| banjo | s3d O-lens | 56 | 0 / 1 | a) guitar b) banjo c) trumpet d) flute | — | — | L56:N | 0.00 |
| cornet | ML-NLA RL600 | 20,28,36,40,56,60 | 1 / 5 | louis armstrong's signature instrument is the trumpet (and cornet), plus his scat singing | L20 n/a (gold control (logreg p 0.060)) | L20: none | L20:H L28:N L36:N L40:N L56:N L60:N | 0.00 |
| cornet | s3d O-lens | 60 | 1 / 0 | the cornet was his main instrument for early new orleans | L60 r=0.766/0.617 no | L60: mlnla-rl600 | L60:X | 0.00 |
| drums | s3d O-lens | 20 | 1 / 0 | trumpet and drums. actually, louis armstrong is famous for playing the trumpet | L20 n/a (gold control (logreg p 0.060)) | L20: none | L20:H | 0.20 |
| flute | J-lens | 28,40 | 0 / 2 | with instruments like guitar, flute and trumpet playing a melody | — | — | L28:N L40:N | 0.20 |
| flute | s3d O-lens | 56 | 0 / 1 | a) guitar b) banjo c) trumpet d) flute | — | — | L56:N | 0.20 |
| guitar | J-lens | 28,32 | 0 / 2 | with instruments like guitar, flute and trumpet playing a melody | — | — | L28:N L32:N | 0.80 |
| guitar | s3d O-lens | 32,56 | 0 / 2 | louis armstrong is best known for jazz, not the electric guitar. | — | — | L32:N L56:N | 0.60 |
| piano | s3d O-lens | 28,44,60 | 0 / 3 | 乐器 (trumpet), not computers or keyboards, for his entire career. | — | — | L28:N L44:N L60:N | 0.20 |
| saxophone | J-lens | 40,44 | 0 / 2 | a wind or brass instrument such as trumpet or saxophone | — | — | L40:N L44:N | 0.20 |
| saxophone | s3d O-lens | 48 | 0 / 1 | trumpet** (or saxophone) is his most famous instrument. | — | — | L48:N | 0.40 |
| trombone | s3d O-lens | 44,56,60 | 1 / 2 | trombone** – louis armstrong's most famous instrument in jazz. | L60 r=0.447/0.149 no | L60: none | L44:N L56:N L60:H | 0.20 |
| tuba | s3d O-lens | 28 | 1 / 0 | /tuba as his main instrument, and he was a pioneer of jazz. | L28 n/a (concept not a bank class) | L28: none | L28:H | 0.00 |
| violin | s3d O-lens | 52 | 0 / 1 | trumpet, not violin, is louis armstrong's instrument. | — | — | L52:N | 0.60 |

#### br-ent-mus-beethoven: "The musician Ludwig van Beethoven is famous for playing the" (gold piano)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| cello | ML-NLA RL600 | 24,44,48,60 | 0 / 4 | most plausibly "piano", possibly "cello" given the preceding cello phrase | — | — | L24:N L44:N L48:N L60:N | 0.00 |
| guitar | J-lens | 20,28,32,36,40,44,48,56,60 | 0 / 9 | playing instruments like piano or guitar and composers such as beethoven | — | — | L20:N L28:N L32:N L36:N L40:N L44:N L48:N L56:N L60:N | 0.80 |
| harp | s3d O-lens | 20 | 0 / 1 | piano (or harp) throughout his life. | — | — | L20:N | 0.00 |
| harpsichord | s3d O-lens | 60 | 1 / 0 | the most famous classical composer to play the harpsichord in his time | L60 r=0.800/0.960 no | L60: none | L60:H | 0.00 |
| violin | J-lens | 40,44,56,60 | 0 / 4 | with other instruments like guitar and violin as nearby alternatives | — | — | L40:N L44:N L56:N L60:N | 0.20 |
| violin | ML-NLA RL600 | 48,52,60 | 0 / 3 | his signature instrument is the piano (also violin, cello, but piano is the canonical answ | — | — | L48:N L52:N L60:N | 0.20 |
| violin | NLA-RL L42 | 42 | 0 / 1 | most likely "piano" or "violin." | — | — | L42:N | 0.00 |
| violin | s3d O-lens | 20,24,36,44,48,52,56,60 | 4 / 4 | violin was beethoven's main instrument. | L20 r=0.740/0.580 no; L44 r=0.560/0.560 no; L48 r=0.560/0.580 no; L60 r=0.780/0.840 no | L20: none; L44: jlens,mlnla-rl600,nla-rl-L42; L48: jlens,mlnla-rl600; L60: jlens,mlnla-rl600 | L20:H L24:N L36:N L44:X L48:X L52:N L56:N L60:X | 0.60 |

#### br-ent-mus-davis: "The musician Miles Davis is famous for playing the" (gold trumpet)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| bass_guitar | s3d O-lens | 32 | 1 / 0 | bass guitar and was one of the most famous jazz musicians of the 20th century. | L32 r=0.851/0.723 no | L32: none | L32:H | 0.20 |
| flute | J-lens | 28,40,56 | 0 / 3 | such as guitar, flute, saxophone and trumpet | — | — | L28:N L40:N L56:N | 0.20 |
| flute | ML-NLA RL600 | 20 | 0 / 1 | plus modal jazz, "kind of blue," flute, and horn accessories like his mute | — | — | L20:N | 0.00 |
| flute | s3d O-lens | 60 | 1 / 0 | flute (the primary instrument he used to define the cool jazz sound) | L60 r=0.447/0.404 no | L60: jlens | L60:X | 0.20 |
| guitar | J-lens | 20,24,28,32,36,40,44,60 | 1 / 7 | the passage likely describes a person who plays jazz guitar or a jazz music scene. | L28 r=1.000/1.000 fires | L28: none | L20:N L24:N L28:P L32:N L36:N L40:N L44:N L60:N | 0.80 |
| guitar | s3d O-lens | 44 | 1 / 0 | he also played guitar and piano, but his | L44 r=0.979/0.957 no | L44: jlens | L44:X | 0.60 |
| piano | J-lens | 56 | 0 / 1 | with sax, piano and flute as alternative jazz instruments | — | — | L56:N | 0.20 |
| piano | ML-NLA RL600 | 40 | 0 / 1 | likely followed by options like trumpet, saxophone, piano | — | — | L40:N | 0.00 |
| piano | s3d O-lens | 20,24,44,52 | 3 / 1 | he played the piano and saxophone | L20 r=0.979/0.872 no; L24 r=0.894/0.723 no; L44 r=0.979/0.915 no | L20: none; L24: none; L44: mlnla-rl600 | L20:H L24:H L44:X L52:N | 0.20 |
| saxophone | J-lens | 24,28,36,40,44,56,60 | 0 / 7 | a jazz musician playing instruments like the saxophone or guitar | — | — | L24:N L28:N L36:N L40:N L44:N L56:N L60:N | 0.20 |
| saxophone | ML-NLA RL600 | 40 | 0 / 1 | likely followed by options like trumpet, saxophone, piano. | — | — | L40:N | 0.00 |
| saxophone | s3d O-lens | 20,24,32,36,44,48,52 | 4 / 3 | he played the piano and saxophone | L20 r=0.787/0.723 no; L24 r=0.787/0.766 no; L32 r=0.936/0.872 no; L36 r=0.872/0.681 no | L20: jlens; L24: jlens; L32: jlens; L36: jlens | L20:X L24:X L32:X L36:X L44:N L48:N L52:N | 0.40 |
| trombone | J-lens | 60 | 0 / 1 | perhaps a sentence saying someone plays the trumpet, trombone or sax | — | — | L60:N | 0.00 |
| trombone | s3d O-lens | 24,52,56 | 1 / 2 | 乐器（trumpet) and later the trombone. he is known for his melodic solos | L24 r=0.213/0.468 no | L24: none | L24:H L52:N L56:N | 0.20 |
| violin | s3d O-lens | 44 | 0 / 1 | a) guitar b) piano c) saxophone d) violin | — | — | L44:N | 0.60 |

#### br-ent-mus-hendrix: "The musician Jimi Hendrix is famous for playing the" (gold guitar)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| harmonica | s3d O-lens | 60 | 0 / 1 | guitar (fender stratocaster), harmonica, etc., but he is best known for | — | — | L60:N | 0.20 |

#### br-ent-mus-john: "The musician Elton John is famous for playing the" (gold piano)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| bassoon | ML-NLA RL600 | 28 | 0 / 1 | kid-friendly "fun facts" listing ("the origin of the word 'bassoon'...") | — | — | L28:N | 0.20 |
| drums | s3d O-lens | 32,44 | 1 / 1 | 琴 (keyboard) and drums, elton john is one of the most famous pianists | L32 r=0.900/0.980 no | L32: none | L32:H L44:N | 0.20 |
| guitar | J-lens | 20,24,28,32,36,40,44,56,60 | 0 / 9 | a person who plays instruments such as guitar or piano | — | — | L20:N L24:N L28:N L32:N L36:N L40:N L44:N L56:N L60:N | 0.80 |
| guitar | s3d O-lens | 20,24,48,52,60 | 3 / 2 | they were famous for playing guitars, keyboards, and harmonica. | L24 r=0.600/0.900 no; L52 r=0.700/0.920 no; L60 r=0.620/0.920 no | L24: jlens; L52: jlens; L60: jlens | L20:N L24:X L48:N L52:X L60:X | 0.60 |
| harmonica | s3d O-lens | 24 | 1 / 0 | they were famous for playing guitars, keyboards, and harmonica. | L24 r=0.840/0.680 no | L24: none | L24:H | 0.20 |
| organ | ML-NLA RL600 | 24,28,36,40,60 | 0 / 5 | elton john's instrumentation and band — piano, hammond organ, "rocket man," | — | — | L24:N L28:N L36:N L40:N L60:N | 0.00 |
| organ | s3d O-lens | 20 | 1 / 0 | paul mccartney famously played a hammond organ on many of his songs | L20 r=0.800/0.520 no | L20: mlnla-rl600 | L20:X | 0.00 |
| saxophone | s3d O-lens | 60 | 0 / 1 | but he also played guitar, saxophone, and other instruments. | — | — | L60:N | 0.40 |
| synthesizer | ML-NLA RL600 | 60 | 0 / 1 | elton john is the expected entity (piano/synth organist) | — | — | L60:N | 0.00 |
| synthesizer | s3d O-lens | 32,36 | 1 / 1 | elton john are indeed famous for their extensive use of pianos and synthesizers in rock mu | L36 r=0.960/0.940 no | L36: none | L32:N L36:H | 0.00 |
| vibraphone | ML-NLA RL600 | 28 | 0 / 1 | elton john's signature instrument — the piano/vibraphone/hammered dulcimer — is the expect | — | — | L28:N | 0.00 |

#### br-ent-mus-ma: "The musician Yo-Yo Ma is famous for playing the" (gold cello)

| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |
|---|---|---|---|---|---|---|---|---|
| bass_guitar | ML-NLA RL600 | 36 | 1 / 0 | yo-yo ma is the world-famous double bassist, so the instrument is cello | L36 n/a (gold control (dom p 0.057)) | L36: none | L36:H | 0.00 |
| bass_guitar | s3d O-lens | 56 | 0 / 1 | cello (the large string instrument), rockstar bassist, or experimental cellist | — | — | L56:N | 0.20 |
| bassoon | ML-NLA RL600 | 60 | 0 / 1 | classical instrument names — cello, viola, bassoon, harp — plus yo-yo ma's signature instr | — | — | L60:N | 0.20 |
| double_bass | J-lens | 56 | 0 / 1 | most likely naming a string instrument such as the double bass (with violin as the runner- | — | — | L56:N | 0.00 |
| double_bass | NLA-RL L42 | 42 | 0 / 1 | a word naming a string instrument like "cello" or "double bass," | — | — | L42:N | 0.00 |
| double_bass | s3d O-lens | 36 | 0 / 1 | double bass, is an instrument of the string family. | — | — | L36:N | 0.00 |
| guitar | J-lens | 20,24,28,32,36,40,44 | 0 / 7 | possibly instruments like guitar, so it anticipates a music-related word | — | — | L20:N L24:N L28:N L32:N L36:N L40:N L44:N | 0.80 |
| guitar | s3d O-lens | 52,56 | 1 / 1 | 大提琴 (cello) and guitar. | L56 r=1.000/0.878 no | L56: none | L52:N L56:H | 0.60 |
| harp | ML-NLA RL600 | 60 | 0 / 1 | classical instrument names — cello, viola, bassoon, harp — plus yo-yo ma's signature instr | — | — | L60:N | 0.00 |
| piano | J-lens | 28 | 0 / 1 | musicians performing on instruments like guitar or piano | — | — | L28:N | 0.20 |
| viola | J-lens | 60 | 0 / 1 | most likely "violin," with viola, viol, or bass as alternatives | — | — | L60:N | 0.00 |
| viola | ML-NLA RL600 | 20,60 | 1 / 1 | likely follow-up clues about bach, violist yo-yo ma (who also plays cello). | L60 r=0.224/0.388 no | L60: jlens | L20:N L60:X | 0.00 |
| violin | J-lens | 40,44,56,60 | 0 / 4 | likely someone playing the violin or a related instrument | — | — | L40:N L44:N L56:N L60:N | 0.20 |
| violin | ML-NLA RL600 | 36,44,48 | 0 / 3 | syntactically demanding a singular instrument noun (e.g., "cello", "violin") | — | — | L36:N L44:N L48:N | 0.20 |
| violin | s3d O-lens | 20,24,28,32,40,48,52,56 | 3 / 5 | violin being the most famous instrument he plays | L20 n/a (gold control (dom p 0.108)); L24 n/a (gold control (dom p 0.052)); L28 n/a (gold control (dom p 0.062)) | L20: none; L24: none; L28: none | L20:H L24:H L28:H L32:N L40:N L48:N L52:N L56:N | 0.60 |

