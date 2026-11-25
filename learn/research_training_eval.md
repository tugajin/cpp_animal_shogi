# Research: training-signal design & oracle-based evaluation for dobutsu shogi selfplay

Compiled 2026-11-20. Purpose: external evidence for deciding what to VERIFY/MEASURE next in this
project (value-only SingleNet + UBFM/Descent best-first search, selfplay loop, loss
`LAMBDA·MSE(out, s) + (1−LAMBDA)·MSE(out, r)` with LAMBDA=0.5, ε=10%, po_num=100),
given the Experiment 4.5 / audit findings:

- ans (problem.txt, sente-only, ±0.2) plateaued at 0.8717; ordering in sente-losing positions is
  INTACT (audit_order: argmax 94.6% vs 90.7%, pairwise 87.9% vs 87.5%) — value *compression*
  (−0.59 vs +0.85 mean) does not affect move choice.
- Real strength leaks: win-position conversion (win kept 89.5%, win→draw 6.9%, win→lose 3.6%)
  and draw preservation (kept 84.2%, draw→lose 15.8%) on played moves.
- Gote-to-move positions are weaker than sente (argmax 91.2% vs 94.2%).
- Standing user decisions: oracle data is for evaluation/diagnosis ONLY, never training;
  playing strength (audit A/C) is the goal metric, ans is reference only.

Verification-status legend: **[full text]** read from the paper itself · **[abstract]** abstract
only · **[docs]** project/codebase documentation · **[blog/report]** non-peer-reviewed source.
Anything not verified from an actual source is flagged inline.

---

## Topic 1 — Dobutsu shogi and other small / solved games learned by selfplay

### 1.1 Tanaka Tetsuro, "An Analysis of a Board Game 'Doubutsu Shogi'"
IPSJ SIG Notes 2009-GI-22 (2009). **[full text via secondary reproductions — see flag]**

Findings:
- Dobutsu shogi is strongly solved: 246,803,167 reachable positions (99,485,568 non-terminal);
  the initial position is a **win for the second player (gote) in 78 plies** — sente loses to
  zugzwang under perfect play.
- The drop (打ち) rule is what gives the game its depth: without drops there are only 962,894
  positions, the deepest win is 37 plies, and the opening is a draw.
- An independent Rust retrograde reproduction matched Tanaka's position counts exactly and
  agreed with Clausecker's independent solution on 50,000 sampled positions with zero
  mismatches — the solution (this project's oracle) is as trustworthy as ground truth gets.

→ VERIFY/MEASURE here: framing, not a new experiment. Because the initial position is a gote
win, "sente-losing" is the theoretically central region of the game, not an edge case; the
selfplay sente-win majority is an artefact of mutual errors. When interpreting the 4-cell
teacher table (search accuracy 93–95% where sente wins vs 60–67% where sente loses), weight
the sente-losing cells as the main line of the game, not as a tail case.

*Flag: counts and the 78-ply result were confirmed via an independent reproduction write-up
(brianhliou.com) rather than by reading Tanaka's original Japanese text directly.*

### 1.2 Nakayashiki & Kaneko, "A Survey on AlphaZero Algorithm through Dobutsu Shogi"
GPW 2019, pp.86–93. (Baseline paper, given; summarised from the uploaded PDF.)

Findings:
- AlphaZero-style selfplay learning reached oracle agreement only ~0.64–0.73, while supervised
  learning on the full solution reached ~0.966–0.991.
- Larger networks sometimes *hurt* under selfplay at fixed compute.
- This project's program already sits well above their selfplay band (≈0.87 ans / ≈0.89
  sign-equivalent at comparable game counts, ~90k vs their 85k games) — the remaining gap to
  the supervised ceiling is a signal-quality gap, not a capacity gap.

→ VERIFY/MEASURE here: do not spend Exp 5 on network enlargement or a full AlphaZero (policy
head + PUCT) migration; the paper gives no support for either at this scale.

### 1.3 Nakayashiki & Kaneko, "Learning of Evaluation Functions via Self-Play Enhanced by Checkmate Search"
TAAI 2018, pp.126–131. DOI 10.1109/TAAI.2018.00036. **[abstract]**

Findings:
- In Mini-shogi (5×5), embedding an exact checkmate (tsume) solver inside AlphaZero-style
  self-play "dramatically improves the efficiency" of learning, via the quality of the
  self-play game records.
- The method stays free of human game knowledge — the solver is exact search, not heuristics.
- The mechanism is record *quality*: positions near mate get exact rather than
  network-estimated treatment during generation.

→ VERIFY/MEASURE here: this is the strongest published counter-hypothesis to "just raise
po_num". Dobutsu games are short (~21 plies) and mate/try lines decide the win→draw and
draw→lose leaks the audit found. Test, on a small generation (~2,000 games): records produced
with mate-search enhancement vs records at uniformly raised PO, compared per wall-clock on
(a) teacher (search) accuracy in the win/lose cells of investigate45-C, and (b) audit A/C of
a model briefly trained on each. If mate-enhancement wins per unit time, Exp 5's lever is
generation-side exact search, not PO and not augmentation.

### 1.4 Nakayashiki & Kaneko, "Evaluation of Loss Function for Stable Policy Learning in Dobutsu Shogi"
TAAI 2020, pp.199–204. DOI 10.1109/TAAI51410.2020.00044 (best-paper third place, TAAI 2020). **[abstract only — full text paywalled (IEEE); details below are from the J-GLOBAL machine-translated abstract and must be treated as unverified beyond that]**

Findings (abstract-level):
- Applies PPO-style policy optimisation to AlphaZero-style learning *in dobutsu shogi itself*,
  targeting the efficiency/stability of policy learning.
- Implies the authors found vanilla policy-gradient-style updates in this exact game unstable
  enough to warrant a dedicated loss-function paper.

→ VERIFY/MEASURE here: only indirectly applicable (this program has no policy head), but it is
evidence that instability in dobutsu-shogi selfplay learning is a documented phenomenon in
this exact game — supports conservative update sizes and generation-over-generation gating
(reject a new model that does not beat its parent on audit A/C) rather than blind accumulation.

### 1.5 Nakayashiki & Kaneko, "逆転の余地を考慮した評価関数の設計とどうぶつしょうぎによる評価" (Design of Evaluation Functions Considering Recovery)
GPW 2020, IPSJ. **[abstract read in full from the IPSJ PDF]**

Findings:
- Replaces the scalar win-probability value with a *soft value* (from Soft Q-learning) so the
  evaluation reflects the number of recovery routes — a position that is losing but has many
  drawing/winning continuations is valued differently from a hopeless one with the same
  win rate.
- In match experiments, players built on it **play more safely when winning as gote and aim
  at comebacks better when losing as sente** — i.e., the benefit shows up exactly in
  win-maintenance and losing-position behaviour, not in aggregate accuracy.

→ VERIFY/MEASURE here: the audit's two leaks (win→draw/lose 10.5%, draw→lose 15.8%) are
precisely the behaviours this paper's design targets. Candidate verification for Exp 6:
a target that distinguishes "losing with recovery routes" from "losing without" — e.g. a
3-way win/draw/lose probability head, or scalar targets softened by sibling-category counts —
judged by audit C's win-maintenance and draw-maintenance cells specifically, not by ans.

### 1.6 Shioda & Ito, "Learning of Evaluation Functions on Mini-Shogi Using Self-playing Game Records"
TAAI 2020, pp.41–46, DOI 10.1109/TAAI51410.2020.00016; extended as "Improving Mini-Shogi Engine
Using Self-play and Possibility of White's Advantage", Journal of Information Science and
Engineering 38(3):591–603 (2022). **[abstract]**

Findings:
- A YaneuraOu-based Mini-shogi engine whose NN evaluation was learned from self-play game
  records won all competitions it entered in 2020 — self-play-record learning (the classic
  shogi method) is competitive in a small shogi variant without any oracle.
- Their experiments suggest a **second-player (gote) advantage in Mini-shogi**.

→ VERIFY/MEASURE here: per-colour strength asymmetry is a documented phenomenon in a sibling
variant, and in dobutsu shogi the *game itself* is a gote win (1.1). Keep per-turn splits in
every audit (already done) and treat the gote-to-move gap (91.2% vs 94.2%) as a first-class
metric for Exp 5, not a footnote.

### 1.7 Ota, Osa, Omura & Harada, "Revisiting Regularized Policy Optimization for Stable and Efficient Reinforcement Learning in Two-Player Games"
ICML 2026 (per U-Tokyo lab page). **[lab page only — tangential]**

- Model-free regularised policy optimisation evaluated on dobutsu shogi, Gardner chess, Go,
  Hex, Othello; claims higher training efficiency than existing methods. No search, no value
  network of this project's kind — listed for completeness only; no verification action.

---

## Topic 2 — Symmetry augmentation (mirror / rotation)

### 2.1 Silver et al., "Mastering the Game of Go without Human Knowledge" (AlphaGo Zero)
Nature 550:354–359 (2017); and Silver et al., "A General Reinforcement Learning Algorithm
that Masters Chess, Shogi, and Go through Self-Play" (AlphaZero), arXiv:1712.01815. **[full text]**

Findings:
- Go training data was augmented with all **8 symmetries** (rotations/reflections) of each
  position, and a random symmetry transform was applied before each NN evaluation in MCTS to
  average out orientation bias.
- AlphaZero explicitly does **not** augment for chess/shogi — their rules are asymmetric
  (pawn direction, castling), so most dihedral transforms are illegal there.
- **No ablation number for augmentation is published** in either paper: it is stated as
  practice, its effect size is unverified from sources.

→ VERIFY/MEASURE here: dobutsu shogi's valid symmetry group is smaller than Go's and is
exactly the oracle normalisation this project already implements — left-right mirror, and
180° rotation *only combined with* colour swap + turn swap (project-derived fact, from
ai/oracle.hpp, not from a paper). Because the model output is turn-perspective, rotation
twins have invariant targets, so augmentation is label-exact. The missing ablation in the
literature is precisely what Exp 5-V1 would measure — run it as an A/B (same data, same
budget, augmentation the only change) and judge on audit A/C per turn.

### 2.2 Wu, "Accelerating Self-Play Learning in Go" (KataGo)
arXiv:1902.10565 (2019, rev. 2020). **[full text for training methods; symmetry from codebase docs — see flag]**

Findings:
- ~50× compute reduction vs ELF OpenGo (surpassed ELF in 19 days on <30 GPUs), from a bundle
  of methods — the headline for this project is *playout cap randomization*: with probability
  p=0.25 a turn gets a full search (N=600→1000), otherwise a fast search (n=100→200), and
  **only full-search turns are recorded for training** (for policy targets), relieving the
  value-target data bottleneck (one outcome per game vs dense per-turn targets).
- Also: forced playouts + policy-target pruning, global pooling, auxiliary targets, SWA, and
  gating (a candidate net must win ≥100/200 games against the incumbent).
- KataGo's training pipeline applies **random symmetry augmentation in the data loader**
  (each batch randomly rotated/flipped), and analysis mode can average over all symmetries.

*Flag: the KataGo paper text itself does not foreground augmentation (a full-text search for
"augment" in the arXiv HTML returned no matches); the augmentation practice is verified from
KataGo codebase documentation (DeepWiki's repo-derived pages: "The data loader supports:
Random symmetry augmentation"; "KataGo exploits the 8-fold symmetry of the Go board for data
augmentation and policy averaging"). Treat as standard-practice evidence, not a measured
effect.*

→ VERIFY/MEASURE here: two separable ideas. (a) Symmetry: same as 2.1. (b) Playout-cap
randomisation, translated to UBFM: mix a minority of high-PO games into generation and
*record training data preferentially from high-PO turns*, vs uniform PO at equal wall-clock —
a direct, literature-backed design for the games-vs-PO question (see also 3.3).

### 2.3 Catan-Zero research chronicle (Gumbel-AlphaZero Expert Iteration on Settlers of Catan)
Project report on GitHub (nickita-khylkouski/catan-zero-public), 2026. **[blog/report — NOT peer-reviewed]**

Findings:
- Measured their value net's symmetry violation directly: across the 12 dihedral transforms
  of the same position, value std was 0.049 (range up to 0.29 on [−1,1]) and prior
  orientation-noise std 0.175 nats — *larger than the spread separating the moves the net
  must rank*. "A large fraction of what looked like irreducible value noise … is violation of
  a known invariance."
- Test-time symmetry averaging gave ~3.3× denoising (≈√12, errors decorrelate) with no
  retraining.
- Train-time symmetry augmentation was tested as a recipe arm: "good (58.5% gate) but not
  better than the value-target fix alone (59.0%), so not adopted" — they fixed the value
  targets *first*.

→ VERIFY/MEASURE here: before committing Exp 5 to augmentation, run the cheap diagnostic
this report suggests: measure the current net's value disagreement between each audit
position and its mirror / rotate-normalised twin. If twin disagreement is small relative to
sibling-value gaps, augmentation treats a non-problem and Exp 5 should lead with a different
lever; if it is comparable, augmentation (and/or test-time twin averaging of leaf values,
a zero-training-cost variant) is well-aimed. Also note their ordering lesson: target fixes
before augmentation.

---

## Topic 3 — Training-signal design for value networks in selfplay

### 3.1 Willemsen, Baier & Kaisers, "Value Targets in Off-Policy AlphaZero: A New Greedy Backup"
Neural Computing and Applications 34(3):1801–1814 (2022; earlier ALA workshop, AAMAS 2020).
DOI 10.1007/s00521-021-05928-5. **[full text]**

Findings:
- Defines a family of value targets subsuming the original AlphaZero target, soft-Z, and A0C.
  Key argument: AlphaZero's game-outcome target is **on-policy** — self-play outcomes reflect
  the *exploratory* behaviour policy, so the learned values converge to the exploratory
  policy's values, not to greedy/tournament play.
- The proposed greedy-backup target (A0GB) was the **only** target that found the optimal
  policy in a small tabular domain where the original target failed; soft-Z / A0C / A0GB beat
  the original target in performance and training speed on **Connect Four (a solved game)**
  and 6×6 Breakthrough.

→ VERIFY/MEASURE here: this project's targets have the mirror-image version of the same
contamination — `r` is generated by ε=10% exploratory play, and `s` is the search value of a
tree built under that same exploratory policy. The Exp 5/6 LAMBDA question should therefore
be framed as *which policy do we want the values to describe?* A concrete testable variant:
targets backed up under greedy move selection (the UBFM analogue of greedy backup — root
value from the best line, ignoring ε-moves actually played) vs the current played-line
targets, in a short run judged by audit A/C. Prediction from the paper: greedy-style targets
should specifically improve win-conversion (values stop crediting exploratory blunders).

### 3.2 Schaul, Quan, Antonoglou & Silver, "Prioritized Experience Replay"
ICLR 2016, arXiv:1511.05952. **[full text]**

Findings:
- Sampling transitions in proportion to |TD error| (proportional or rank-based) beat uniform
  replay on 41/49 Atari games with DQN.
- Two safeguards are part of the method, not optional extras: stochastic (not greedy)
  prioritisation, and importance-sampling weights with β annealed to 1 — without them,
  prioritisation loses diversity and biases the fit.

→ VERIFY/MEASURE here: the legal way to oversample "rare-but-critical" positions in this
project *without touching oracle data for training* is to prioritise by signals available at
generation time: teacher disagreement |s − r|, model-vs-twin inconsistency (mirror/rotate),
or low sibling-margin (only-move-like positions). Candidate Exp 6 experiment: disagreement-
prioritised sampling with IS weights vs uniform, judged by audit C leak cells. PER's warning
applies directly: monitor overall audit A to catch diversity loss.

### 3.3 KataGo playout cap randomization (Wu 2019 — see 2.2)
**[full text]**

- The value-target bottleneck is structural: each game yields many policy targets but only
  one outcome. KataGo's answer is to spend search unevenly and record selectively, not to
  generate uniformly more games.

→ VERIFY/MEASURE here: for the games-vs-PO decision, the literature-backed middle option is
*mixed* PO with selective recording, tested at equal wall-clock against both current settings
(po=100 uniform) and uniform po≈400. Metrics: teacher accuracy (investigate45-C machinery)
of the recorded data, and audit A/C after a short training run on each variant.

### 3.4 Raccoon backgammon self-play analysis (practitioner project write-up)
research.raccoonbg.com/training_analysis.html — ten AlphaZero-style self-play experiments
(ResNet policy-value net, single-GPU scale, external anchor = GNUBG 2-ply). **[blog/report — NOT peer-reviewed; treat as a well-documented anecdote, not evidence of record]**

Findings (all verified from the page itself):
- Stated principle, demonstrated repeatedly: **"Loss curves are fitting metrics, not strength
  metrics"** — policy/value loss diverged from head-to-head strength in both directions
  across the track (e.g. 338 iterations in which every fitting metric degraded while play
  quality was fully preserved).
- exp004: blending the MCTS root Q into value targets (bootstrap α=0.5) made value loss ~3×
  *lower* (0.038–0.053 vs 0.105–0.140) with **no detectable strength change** — smoother
  targets ≠ better values; "value-target noise was not the binding constraint."
- exp000/exp001: a persistent **+0.3 ppg first-player bias** despite a verified 50/50 opening;
  diagnosed as a small raw-network asymmetry *amplified by shallow (25-sim) search and then
  reinforced by training on the resulting games*; it shrank to ~+0.08 at 200 sims — "a useful
  thermometer for whether search depth is adequate to correct the raw network."
- exp005: **800-sim search was the strongest self-play lever found** (only intervention with a
  statistically significant within-run gain, +0.38 ppg, p≈0.035); exp006: iteration count
  alone improved so slowly (~17,000 iterations to close the gap) as to be non-viable. Track
  conclusion: "the bottleneck is the *quality* of the self-play training signal, not the
  amount of training."

→ VERIFY/MEASURE here: three direct imports. (a) It corroborates this project's move to
audit-based metrics — never accept/reject an Exp 5 change on loss or ans alone. (b) Before
expecting a LAMBDA change to help, note exp004's null result for target *smoothing*; the
3.1 greedy-backup framing (changing *which policy* the target describes) is the variant with
a mechanism, pure re-weighting may have none. (c) The bias-thermometer test discriminates the
two live explanations of the gote gap: measure per-turn audit metrics on data/models at
po=100 vs po≈400 — if the gap shrinks with PO (as Raccoon's bias did), search depth is the
lever; if it does not, it is a data/representation problem and augmentation (Topic 2) is.

### 3.5 Anthony, Tian & Barber, "Thinking Fast and Slow with Deep Learning and Tree Search" (Expert Iteration / ExIt)
NeurIPS 2017, arXiv:1705.08439. **[full text]**

Findings:
- Decomposes learning into a search *expert* generating targets and a network *apprentice*
  generalising them; tabula-rasa ExIt on Hex defeated MoHex 1.0 (the reigning Olympiad
  champion program) and beat REINFORCE with 84.4% wins at equal training time.
- Tree-policy targets (TPT) beat chosen-action targets (CAT); TPT are formally
  *cost-sensitive* — a monotonic transformation of the advantage function — so positions and
  moves where the choice matters carry proportionally more learning signal.

→ VERIFY/MEASURE here: precedent that target *shape*, not data volume, drove the gain in a
small-board game learned from search alone. Value-only analogue for this project: weight the
value loss by an advantage-like quantity available without the oracle — e.g. the sibling
spread of search values, or |s − r| — so that positions where the move choice is critical
dominate the loss. This is a cheaper cousin of 3.2's prioritised sampling (weighting instead
of sampling) and can be A/B-tested in the same short-run harness.

---

## Topic 4 — Oracle-based MOVE-quality evaluation as a strength proxy

### 4.1 Guid & Bratko, "Computer Analysis of World Chess Champions"
Computers and Games (CG 2006); ICGA Journal version. **[full text]**

Findings:
- Strength assessed by the **average evaluation loss between the moves a player chose and
  the engine's best moves** (Crafty at fixed depth 12), with moves labelled **blunders when
  the error exceeded 1.00 pawn**.
- They designed a **position-complexity measure** — how much the best move changes across
  search depths — to correct for the fact that some players face/seek harder positions;
  raw error rates without a complexity correction confound style with strength.
- Rankings were stable across engines of very different strength (Crafty ~2620 vs later
  Rybka 3): "it is not necessary to use a computer stronger than the players" to get
  sensible move-quality rankings.

→ VERIFY/MEASURE here: this project's audit_order is the same construction with a *perfect*
evaluator, so Guid–Bratko's main caveat (evaluator strength) disappears — what remains
worth importing is the complexity correction. Add to the audit: a per-position complexity
proxy (e.g. does the argmax move change between po=25 and po=400, or the sibling value
margin), and report leak rates split by complexity. Otherwise an Exp 5 "improvement" could
be nothing but easier positions in the sample.

### 4.2 Regan & Haworth, "Intrinsic Chess Ratings"
AAAI 2011. (Follow-up: Regan, Macieja & Haworth, "Understanding Distributions of Chess
Performances", 2011.) **[full text]**

Findings:
- Rates players purely by move quality against a strong engine (Rybka 3, Multi-PV, depth 13)
  rather than by game results; the only inputs are the evaluations of all candidate moves
  and which move was played.
- A fallible-agent model with sensitivity (s) and consistency (c) parameters, fitted by
  regression, establishes a smooth correspondence between move-quality statistics and Elo —
  i.e., move-choice-vs-oracle statistics are a *validated* strength proxy, not an ad-hoc one.

→ VERIFY/MEASURE here: justification for the project's current direction (audit A/C as the
primary metric, no matches needed). Optional upgrade: fit a simple two-parameter
fallible-agent model to audit C's category-drop distribution to express generation-over-
generation changes in Elo-like units, giving Exp 5 a calibrated effect size instead of raw
percentage points. At minimum, report the blunder rate (category drop ≥1; ≥2 as "blunder")
per generation as the headline number, mirroring 4.1's thresholding.

### 4.3 Industry practice: average centipawn loss / win-probability loss
Chess.com-style analysis and Chess DNA's analysis of 3,540 games. **[secondary / commercial sources — NOT academic; flagged]**

- Common convention: a blunder = giving away ≥20% win probability. Chess DNA claims blunder
  rate explained 58% of rating variance vs 31% for accuracy (commercial blog analysis —
  do not cite as evidence).
- In dobutsu shogi the oracle makes this exact rather than approximate: a category drop
  (win→draw, draw→lose, win→lose) *is* a win-probability cliff, so audit C's transition
  matrix is the exact analogue of win-probability-loss auditing.

→ VERIFY/MEASURE here: none beyond 4.1/4.2 — included so the project's metric can be
described accurately as "exact category-preservation auditing, the solved-game analogue of
ACPL/blunder-rate analysis."

---

## Consolidated verification agenda (mapped to the open decisions)

Ordered by cost — the first three are diagnostics that discriminate between the competing
Exp 5 levers *before* any long run:

- **V-1. Symmetry-consistency diagnostic** (hours, no training). Measure the current net's
  value disagreement across mirror / rotate-normalised twins on audit positions, relative to
  sibling-value gaps (Catan-Zero method, 2.3). Decides whether augmentation targets a real
  defect; also test test-time twin-averaging of leaf values in the audit harness.
- **V-2. PO-as-thermometer test** (~2,000 games at po≈400). Compare teacher accuracy
  (investigate45-C) and per-turn audit gaps against po=100 data. If the gote gap shrinks with
  PO → search depth is the lever (Raccoon, 3.4). If not → data/representation (augmentation).
- **V-3. Mate-search enhancement vs uniform PO** (TAAI 2018, 1.3). Same small-generation
  harness as V-2, third arm: tsume-solver-enhanced records. Compare per wall-clock. This is
  the published result most specific to shogi-variant selfplay and it points at generation
  quality, not volume.
- **V-4. Target-policy short run** (Willemsen et al., 3.1; Raccoon exp004 caveat, 3.4).
  Greedy-backup-style targets vs current played-line targets at fixed LAMBDA; judge ONLY by
  audit A/C (esp. win-maintenance). Do not judge by loss — exp004 shows loss will improve
  ~3× even when strength does not move.
- **V-5. Complexity-stratified audit** (Guid–Bratko, 4.1). Re-report existing audit results
  split by a best-move-stability complexity proxy, so all Exp 5 comparisons are
  complexity-controlled. Pure analysis of existing data.
- **V-6. (Exp 6 candidate) Disagreement-prioritised sampling or advantage-weighted loss**
  (PER, 3.2; ExIt TPT, 3.5), using |s − r| / sibling spread — no oracle involvement, so it
  respects the no-oracle-training decision.
- **V-7. (Exp 6 candidate) Recovery-aware targets** (GPW 2020, 1.5): 3-way W/D/L head or
  route-count-softened targets, judged on the win-maintenance and draw-maintenance cells
  specifically.

What the literature does NOT support spending on: more games at unchanged settings
(duplication already 50.5%, new-position rate falling; Raccoon exp006; KataGo 3.3), network
enlargement (GPW 2019, 1.2), full AlphaZero migration (1.2), and any conclusion drawn from
loss curves or ans alone (3.4, 4.2).

## Verification-status summary (what to distrust)

- **Abstract-only**: Nakayashiki & Kaneko TAAI 2020 loss-function paper (paywalled; J-GLOBAL
  machine translation); TAAI 2018 checkmate-search paper; Shioda & Ito TAAI 2020.
- **Documentation-level**: KataGo symmetry augmentation (codebase docs via DeepWiki; the
  paper text does not foreground it). KataGo playout-cap randomization IS from the paper.
- **Non-peer-reviewed**: Raccoon backgammon analysis (practitioner blog, but internally
  detailed and self-critical); Catan-Zero chronicle (project report); Chess DNA figure
  (commercial blog — not cited as evidence anywhere above).
- **No published effect size**: AlphaGo Zero / AlphaZero symmetry augmentation — standard
  practice, never ablated in print. Any gain claimed for Exp 5-V1 must come from the
  project's own A/B.
- **Secondary-source confirmation**: Tanaka 2009 position counts / 78-ply gote win (via an
  independent reproduction that matched them exactly).
