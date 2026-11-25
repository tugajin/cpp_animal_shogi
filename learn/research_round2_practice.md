# Round-2 research: mate solvers in self-play, soft values, and practitioner practice (dlshogi / KataGo)

Scope: primary sources read where obtainable (full texts, official abstracts, source code, author's own docs).
Every entry carries a **verification status**. Claims marked (unverified) come from an abstract only or a
secondary source and the full text could not be read (IEEE paywall etc.).

Program context assumed throughout: value-only net + best-first minimax (UBFM/Descent-style), mate solver already
embedded in search (5-ply mate / 4-ply mated at node evaluation), targets = 0.5·(search score s) + 0.5·(game result r),
oracle (Tanaka retrograde DB) used for evaluation/diagnosis only. Known leaks from the oracle audit of self-play
records: win positions converted only 89.5% (3.6% blundered to losses), draw positions held only 84.2%.

---

## 1. Nakayashiki & Kaneko — mate solver inside self-play, and the dobutsu follow-ups

### 1.1 "Learning of Evaluation Functions via Self-Play Enhanced by Checkmate Search"
- Taichi Nakayashiki, Tomoyuki Kaneko. TAAI 2018 (IEEE), pp. 126–131. DOI 10.1109/TAAI.2018.00036.
- **Verification status: official abstract read (Semantic Scholar / IEEE record); full text paywalled — mechanism
details below the abstract level are.**

What the abstract states verbatim (paraphrased tightly):
- Setting is AlphaZero-style iteration (self-play with current eval function + training on recent self-play records)
in **Minishogi (5×5)**, not dobutsu shogi.
- Proposal: "incorporate a checkmate solver in self-play". This "small enhancement **dramatically improves the
efficiency**" of the experiments, and the improvement comes "**via the quality of game records in self-play**".
- The method remains free of human game knowledge (the solver is domain-dependent code, but no human records/heuristics).

What is NOT established from accessible sources:
- The exact integration point (root move override during generation? leaf-value override inside search? game
adjudication when a mate is found? target correction after the game?). The abstract's wording ("quality of game
records") is consistent with the solver acting **during game generation** so that mates are actually played /
games terminate correctly, which then cleans up the outcome labels r — but the exact mechanism, the solver used
(df-pn is plausible: Kaneko is the author of the df-pn implementation in GPS Shogi and of "Parallel Depth First
Proof Number Search", AAAI 2010 — this connection is an inference, not stated in the abstract), mate depth/node
limits, and the numeric size of the improvement / metric (win rate vs a baseline at equal games? equal wall time?)
could not be verified.

**Implication for this program.** The one claim that *is* verified — gains come through **record quality**, i.e.
through cleaner r labels and correctly finished endgames, not through better targets per se — suggests a concrete
measurement the current program has not yet made:
- **Mate-solver firing audit (diagnostic, no training):** instrument generation (or replay stored records through
the search) to log, per position, whether the mate/mated verdict changed the backed-up root value or the played
move compared with mate checks disabled. Report the firing rate by game phase and by oracle category. If the
solver rarely fires before the position is already resolved, the 5/4-ply budget — not the concept — is the limit,
and the next experiment is mate depth (7/9 ply, or node-capped df-pn-style) rather than more PO.
- **Record-quality A/B (generation-side):** generate two small record sets at identical PO, mate checks on vs off,
and compare them with the existing oracle audit machinery (teacher accuracy of s and r by category) *before*
training anything. This isolates the TAAI-2018 claim in this engine at record level, cheaply.
- Note the asymmetry with dlshogi (§3.1): dlshogi's leaf mate check is 7-ply (odd) and runs *inside* search;
this program's 5/4-ply checks are shallower, and dobutsu "mate" includes try-rule wins, which are races, not
king-capture sequences — solver coverage should be measured separately for capture-mates vs try-wins.

### 1.2 "どうぶつしょうぎを用いた AlphaZero の手法の調査" (AlphaZero survey on dobutsu shogi)
- Taichi Nakayashiki, Tomoyuki Kaneko. Game Programming Workshop 2019 proceedings, pp. 86–93, 2019-11-01.
- **Verification status: official abstract verified via CiNii/Ceek index record (id.nii.ac.jp/1001/00199884);
numeric results (AZ self-play accuracy 0.64–0.73 vs supervised 0.966–0.991 etc.) are from the project owner's
copy of the PDF as recorded in the previous round — not re-read in this round.**

Abstract-level claims verified this round:
- They measured how correctly an AlphaZero-trained net judges positions, by direct comparison against Tanaka's
full retrograde analysis; varied network size; also ran supervised learning from the analysis itself.
- Findings: for supervised learning, larger nets are better; **for AlphaZero-style learning, larger nets are NOT
necessarily better**; and MCTS hyperparameters change search behaviour substantially.

**Implication.** Two usable points: (a) capacity increases are not a reliable lever in the self-play regime — deprioritise
net enlargement again; (b) their evaluation methodology *is* this program's audit (net outputs vs retrograde DB),
so audit deltas are a publishable, precedent-backed primary metric for Exp 5 — preferable to small game samples (§4).

### 1.3 "逆転の余地を考慮した評価関数の設計とどうぶつしょうぎによる評価" (soft value / recovery-aware evaluation)
- Taichi Nakayashiki, Tomoyuki Kaneko. The 25th Game Programming Workshop 2020 (GPW 2020), pp. 22–29.
PDF: https://ipsj.ixsq.nii.ac.jp/record/207649/files/IPSJ-GPWS2020005.pdf
- **Verification status: FULL TEXT READ (primary).**

Exact definition of the soft value (their §2.2–§3):
- From soft Q-learning (Haarnoja et al.): soft Q and soft value under policy π,
Q^π_soft(s,a) = r + γ·E[V^π_soft(s′)], V^π_soft(s) = E_{a∼π}[ Q^π_soft(s,a) − α·log π(a|s)],
with the softmax policy π(a|s) ∝ exp(Q^π_soft(s,a)/τ), τ = α during learning; α weights the entropy term.
- Dobutsu instantiation: terminal V = 1 for a win, 0 for a loss (note: 0/1, not ±1); r = 0, γ = 1; values are
sign-flipped each ply (zero-sum). Computed by **retrograde analysis over all 246,803,167 reachable positions**:
seed terminals, then repeatedly compute positions whose children are all computed, processing in order of
increasing remaining plies to the end. Positions caught in cycles (sennichite) cannot be computed this way; the
fix used is step (4): when a child is still uncomputed, assume V(child) = 0 (the loss value in their 0/1 encoding,
i.e. "assume no recovery") and proceed. Full computation ≈ 1 hour single-threaded.
- The quantity it encodes: in a won position, the *fraction/diversity of outcome-preserving moves*
（勝敗不変手 — moves that keep the game-theoretic outcome) drives the policy entropy; a win with many preserving
moves gets a higher soft value than a win with a single preserving move. Symmetrically, in a lost position the
soft value is higher when the *opponent's* preserving moves are few / hard to find — i.e. it measures
"room for recovery" on both sides. They define 勝敗不変手率 = (# outcome-preserving moves)/(# legal moves).

Measured results (their §4, all against oracle-perfect reference players, 100,000 games per colour):
- Players: Top-n-randomized (uniform among the top-n moves by true W/D/L class), Soft Qvalue (α,τ), Expectation
(same retrograde computation with the entropy term removed), Shortest-win/Longest-lose.
- Vs Top-2-randomized: Soft Qvalue (α=0.1, τ=0) won **100% as both sente and gote**; Expectation (α=0.3) 99.87%/99.96%;
Top-1-randomized only 40.4% as sente. The mechanism evidence: the **opponent's** outcome-preserving-move ratio
faced when playing Soft Qvalue was **0.0989**, vs 0.269 (Expectation), 0.270 (Shortest-win/Longest-lose),
0.379 (Top-1) — the soft-value player steers lost positions into ones where the opponent has very few safe moves,
i.e. positions the opponent is likely to get wrong, and thereby comes back.
- Head-to-head at matched entropy (Table 4): Soft Qvalue (α=0.3) vs Expectation (α=0.3, τ=0.3) — at τ=0, 100%/100%;
win rate degrades gracefully with τ (0.1 → 99.6%/99.7%; 0.3 → 59.0%/60.7%).
- Value-vs-distance scatter (10,000 sampled positions): both soft value and Expectation drift toward 0 as plies-to-
terminal grow, but the soft value changes **more smoothly and with less overlap between the win and loss clouds**.
Explicitly (their §4.3): with Expectation, in positions far from the end **both winning and losing moves take
values near 0** — an exact-solver demonstration of the value compression this program observes in its learned net.
- Along Tanaka's 78-ply forced gote win from the initial position, the soft value discriminates positions that the
hard value labels identically (all "gote win"), and varies less ply-to-ply than Expectation.

**Implications for this program (the most directly actionable paper of this round):**
1. **Stratify the existing audit by "number of preserving moves".** The leaks (win→draw/loss 10.5%, draw→loss 15.8%)
are predicted by this paper to concentrate in positions where the side to move has *few* outcome-preserving moves
(fragile wins/draws). The oracle DB plus the position's legal-move list gives this count for every audited
position today, with no training and no rule conflict (evaluation only). If the concentration is confirmed,
"fragile positions" become the named target of Exp 5 instead of the diffuse "win/draw cells".
2. **A search-derived soft backup is available without the oracle.** The entropy-regularised backup can be applied
to the engine's *own* search tree (softmax over children of backed-up values instead of hard minimax), producing
a soft search score s_soft from the same generation run. Training (or even just move selection) with a target
that distinguishes "win with 5 preserving moves" from "win with 1" attacks exactly the conversion leak, and does
not touch the oracle-for-training prohibition. This is a well-defined Exp 5/6 arm: hard minimax backup vs
soft backup at matched PO, judged by the audit's conversion rates.
3. **Exact soft values from the oracle are computable in ~1 hour** (their reported cost) and would make a superb
*diagnostic* scale (e.g. correlation of net value with soft value vs with hard value). Using them as training
targets would violate the standing "oracle = evaluation only" decision — flagged here as a decision point for
the user, not a recommendation to bypass it.

### 1.4 "Maximum Entropy Reinforcement Learning in Two-Player Perfect Information Games"
- Taichi Nakayashiki, Tomoyuki Kaneko. IEEE SSCI 2021, pp. 1–8. DOI 10.1109/SSCI50451.2021.9659991.
- **Verification status: official abstract read; full text paywalled.**

Abstract-level claims:
- Max-entropy RL (soft Q-learning / SAC-style) applied to two-player zero-sum games: the entropy term keeps
(increases) alternative choices on our side and reduces them on the opponent's side → more robust strategies
(the game version of the GPW 2020 idea, now as a *learning* algorithm rather than a post-hoc evaluation).
- It "**improves the stability and efficiency of learning by reducing the variance in learning targets** of neural
networks, which are typically {1, 0, −1} for win, draw, or loss in AlphaZero".
- The resulting soft state values take **intermediate real values, especially near zero in positions near the
opening**; benefits shown by (i) analysis of converged values from *extended retrograde analysis* for dobutsu
shogi and (ii) empirical learning efficiency of self-play RL agents.

**Implication.** This is the literature's direct answer to the observed compression/saturation question (§4 of the
task): hard {+1, 0, −1} outcome targets have maximal variance and force the net to spend capacity on
near-unlearnable distinctions; entropy-regularised targets are graded and lower-variance. For this program the
cheapest faithful translation is not full SAC — it is the **soft backup of §1.3(2)** applied to the search score,
and/or softening only the r-component (e.g. r replaced by a distance/entropy-aware value). It also gives a
*predicted signature* to verify before adopting: if target variance is the problem, then per-position target
variance across generations should be highest exactly in the leak cells — measurable from stored records.

### 1.5 "Evaluation of Loss Function for Stable Policy Learning in Dobutsu Shogi"
- Taichi Nakayashiki, Tomoyuki Kaneko. TAAI 2020 (IEEE), pp. 199–204. DOI 10.1109/TAAI51410.2020.00044.
(TAAI 2020 best-paper award list, 3rd place.)
- **Verification status: official abstract read; full text paywalled.**

Abstract-level claims: they take AlphaZero-style policy learning and replace/augment the policy loss with **PPO**
(motivated by recent analyses of policy optimisation in tree search), in dobutsu shogi, where the retrograde DB
allows exact evaluation; result: improved **sample efficiency** of learning.

**Implication.** Limited direct transfer (this program has no policy head), but two transferable points:
(a) in this exact game, *loss-function choice* is a first-order, measurable variable for learning stability —
worth one controlled arm in Exp 5 (e.g. MSE vs Huber vs cross-entropy-on-soft-targets for the value head) judged
by audit, not by training loss; (b) methodological precedent: they too used the retrograde DB as the evaluation
ground truth instead of game samples.

### 1.6 Other Nakayashiki/Kaneko dobutsu work
- "ついたて王手どうぶつしょうぎの提案と CFR による戦略の学習" (dark/check dobutsu shogi variant, CFR), GPW 2021,
pp. 34–41. Full text read (IPSJ PDF). Imperfect-information variant; CFR with abstraction learned a strategy
beating a random player >80% as sente. **Low relevance** to this program beyond confirming the group's continued
use of exact/equilibrium computation as ground truth for small shogi variants.

---

## 2. Other work on small solved games with solver available during learning (2019–2025)

### 2.1 Mini-2048 (3×3 2048): full solution + RL, and the follow-up analyses
- Shuhei Yamashita, Tomoyuki Kaneko, Taichi Nakayashiki, "3×3 盤面の 2048 の完全解析と強化学習の研究",
GPW 2022, pp. 1–8.
- **Verification status: official abstract verified via CiNii/Ceek index; full text not read.**

Abstract-level claims: they fully solved 3×3 2048 (state counts, scores under optimal play, robustness to model
perturbations), trained an agent with a simplified Stochastic MuZero, evaluated it by **agreement rate with the
optimal policy**, and showed that **intensively training on positions where the gap between the best and worst
moves' expected scores is large is important** for learning.

Follow-ups using the same solved game as an analysis instrument:
- Shunsuke Terauchi, Kiminori Matsuzaki, "Using strongly solved Mini2048 to analyze players with N-tuple networks",
TAAI 2023, pp. 165–178; and the 2024/2025 programming-symposium extensions with Expectimax search.
**Status: abstract read, partially truncated.** Reported pattern: deeper search raises average
score overall, but against solver ground truth, for high-capacity evaluators deeper search could move play
*away* from the true best move in some positions — i.e. "more search" is not monotonically "more correct" when
judged exactly.
- Tomoya Kaneko, Kiminori Matsuzaki, "ミニ 2048 における Double Q 学習の効果 — 過大評価を乗り越えるために —",
GPW 2025. **Status: full text read (IPSJ PDF, primary).** Findings: in deep RL on solved mini-2048,
**value overestimation arises specifically in positions where several options' values are close**; their earlier
work had found that introducing board symmetries (rotation/mirror augmentation) into data generation induced
overestimation especially in the opening. Double Q-learning removed the overestimation and achieved the best
greedy-play final score, but introduced **underestimation**, which they flag as harmful for MCTS.

**Implications for this program.**
- **Margin-weighted data (solver-free version):** the mini-2048 result says the positions that matter are those
with a large best-vs-worst spread. This program can compute an analogous spread from its *own search* (max child
value − min child value, or root value minus value of the played move) at generation time and upweight such
records — no oracle needed. (An oracle-margin version exists but conflicts with the no-oracle-training rule.)
- **Minimax-backup bias check:** best-first minimax backup over learned values has the classic max-operator
overestimation bias that Double-Q addresses. The audit can measure the sign of the bias directly: mean
(net value − oracle-expected outcome) in win positions vs loss positions. If win positions are systematically
overestimated, that is a concrete mechanism for win→draw/loss leaks (the search trusts inflated children), and a
cheap mitigation arm exists (target-net / double estimator, or backing up a soft/average value instead of the
pure max — same arm as §1.3(2)).
- **Augmentation warning for Exp 5 V1:** if mirror/rotation augmentation is adopted, the GPW 2025 result predicts
a specific side effect — value overestimation where sibling values are close, worst early in the game. The
verification plan should include a before/after bias measurement (§ above) as a guard metric for V1, not just
accuracy.

### 2.2 Mini-shogi self-play records (UEC group)
- Masahiro Shioda, Takeshi Ito, "Learning of Evaluation Functions on Mini-Shogi Using Self-playing Game Records",
TAAI 2020 (IEEE), pp. 175–180 (per KAKEN record) / DOI 10.1109/TAAI51410.2020.00016.
- **Verification status: official abstract read; full text paywalled.**

Abstract-level claims: YaneuraOu-based mini-shogi program, evaluation learned from **self-play game records** with
the standard shogi learning method; won the UEC Cup mini-shogi tournament (March 2020) and scored >70% against
leading programs; analysis of the self-play records suggested a possible **second-player (gote) advantage** in
mini-shogi.

**Implication.** (a) Precedent that plain self-play-record learning (no solver in the loop) is competitive in a
small shogi variant — the solver-in-generation of §1.1 is an efficiency enhancer, not a prerequisite.
(b) Their colour-asymmetry observation transfers as a *diagnostic*: in dobutsu, theory says gote wins (78 plies),
so the self-play colour win-rate, tracked per generation, is a free calibration signal — if self-play sente win
rate drifts far above what the audit says it should be, the value function (not the game) is biased. This program
already found a sente/gote accuracy asymmetry; adding the colour win-rate of the *generating* games per generation
to the dashboard costs nothing.

### 2.3 Gardner chess / small Hex — negative result of this search
No 2019–2025 paper was found that learns Gardner (5×5) chess or small Hex by self-play *with the exact solver used
inside the learning loop* (data filtering / target correction / curriculum). Gardner chess was weakly solved
(Mhalla & Prost, 2013) and small Hex boards are solved, but the located literature uses solvers for analysis, not
as a training component. **Status: absence-of-evidence claim — searches were in English and Japanese; treat as
"not found", not "does not exist".** The dobutsu/mini-shogi/mini-2048 line above is, as far as this search shows,
the main body of solver-in-the-loop learning work on small solved games.

---

## 3. Practitioner practice: dlshogi ecosystem and KataGo

### 3.1 dlshogi self-play implementation (primary source: source code)
- Tadao Yamaoka, DeepLearningShogi, `selfplay/self_play.cpp` (master branch, read in full this round).
https://github.com/TadaoYamaoka/DeepLearningShogi/blob/master/selfplay/self_play.cpp
- **Verification status: PRIMARY (author's source code).**

Concrete practices relevant to (a) draws, (b) resignation/length, (d) target quality:
1. **Exact results override the net inside search.** At leaf expansion, repetition is checked first
(`Position::isDraw(16)`): RepetitionDraw → node marked Draw, backup value 0.5 (win-probability scale);
RepetitionWin / RepetitionSuperior (perpetual check / declaration advantage) → Win; RepetitionLose /
RepetitionInferior → Lose. Code comment: for repetition nodes the ValueNet value is *not used*. Immediately
after, a **mate check** (`mateMoveInOddPly<MATE_SEARCH_DEPTH>`, MATE_SEARCH_DEPTH = 7) plus a nyugyoku
(declaration-win) check: if mate found, the node is marked Win and the NN value is **overwritten**
(comment: 詰みの場合、ValueNetの値を上書き). Nodes with no legal moves are Lose. Only otherwise is the NN
evaluated. — I.e. dlshogi's search leaves are *solver-corrected* exactly like this program's, but with a
deeper fixed mate depth (7 vs 5) and with repetition outcomes (not just mate) as exact overrides.
2. **Root-level df-pn mate search** runs asynchronously in parallel threads (DfPn member, MateSearchEntry slots
with RUNNING/NOMATE/WIN/LOSE states), with configurable ROOT_MATE_SEARCH_DEPTH and MATE_SEARCH_MAX_NODE
(minimum 10,000 nodes) — a separate, deeper mate channel than the leaf check.
3. **Adjudication instead of resignation:** after move selection, if the chosen move's win rate satisfies
|winrate| > WINRATE_THRESHOLD (default **0.99** on the −1..+1 scale), the game is terminated on the spot and
recorded as a win/loss (`gameResult` set, records written). Games are therefore rarely played out to actual
mate once decided — but termination uses the *search's* win rate, and the result label is the adjudicated one.
4. **Draw handling in records:** a game reaching repetition is recorded with result Draw (reason
GAMERESULT_SENNICHITE); perpetual-check repetitions are wins/losses, not draws. A game exceeding **MAX_MOVE
(320)** is declared a draw, and **by default its records are discarded entirely** (OUT_MAX_MOVE=false →
records cleared before output) — overlong games are treated as corrupted data, not as draw examples.
Games shorter than MIN_MOVE (default 10 plies) are likewise not output as training data.
5. Teacher records store, per position, the move, the search win rate converted to a score
(`value_to_score(best_wp)`), and the final game result — the same (search value, outcome) pair structure as
this program's (s, r), which makes the dlshogi practices above directly comparable.

**Implications for this program.**
- **Measure the override profile.** dlshogi practice suggests logging what fraction of leaf evaluations are
solver-decided (mate / repetition / no-move) vs NN-decided, per game phase. If, in dobutsu generation, almost
all endgame leaves are already solver-decided, then residual endgame error lives in the *backup/path* (which
children the best-first search chooses to expand), not in leaf values — a different fix (expansion policy)
than "more PO" or "deeper mate".
- **Adjudication bias check.** If this program's generation terminates games early by any rule (mate found,
move cap, draw rule), the r labels of late positions are adjudicated, not played — verify how the current
generator ends games and whether r for the last ~10 plies is systematically the search's opinion restated
(which would make r ≈ s at the end of games and silently change the effective LAMBDA by phase).
- **Drawn-game data policy is a real design knob:** dlshogi *discards* its pathological draw class (overlong
games) rather than training on it. Dobutsu analogue to verify: what fraction of this program's records come
from sennichite-drawn games, and are all their positions labelled r = 0 — including positions that were
objectively won/lost mid-game? If so, drawn games inject exactly the label noise that would produce "draw
positions held only 84%" behaviour, and a filtering/relabelling arm (e.g. drop records from drawn games, or
cap their weight) is a cheap, well-precedented experiment.

### 3.2 Why dlshogi added the mate solver — developer interview (secondary)
- Interview with Tadao Yamaoka, NicoNico News / Bunshun Online, "水匠 vs dlshogi 長時間マッチ観戦記" (2021-08/10).
https://originalnews.nico/336302
- **Verification status: SECONDARY (journalism), but direct quotes of the developer.**

Content: after the 1st Denryusen, dlshogi's endgame was found weak ("it would not try to read mates where it had
to go for mate"). Yamaoka dropped the "deep learning only" constraint and relied on existing technique (df-pn)
for mate. Quote (translated): "Once dlshogi stopped missing mates, it began mass-producing self-play records
whose endgame accuracy was high. Training on those records as teacher data, dlshogi's endgame kept getting
stronger… The quality of the teacher (data) is what matters. When we changed that, it suddenly became strong."
He also states his development loop looks only at **win rate and 正解率** (the rate at which the net, without
search, predicts the move that long-thinking dlshogi plays) — i.e. a *move-matching proxy metric* against a
strong reference, not raw loss.

**Implication.** Two independent lines (TAAI 2018 abstract; dlshogi's history) converge on the same causal story:
**solver-corrected generation → cleaner endgame records → endgame strength**, with the effect attributed to the
records, not to search-at-play-time. For this program, whose solver is already in search, the transferrable
experiment is §1.1's firing audit + a *depth* arm, and the transferrable metric is Yamaoka's proxy idea in oracle
form: **move-matching rate** — how often the net's greedy move (or the search's move) equals an oracle-preserving
move — which is precisely the audit's category-preservation rate. The audit is therefore not just a diagnostic;
it is the same class of proxy metric a top practitioner uses as his primary development signal.

### 3.3 KataGo (primary: author's paper abstract + KataGoMethods.md, read this round)
- David J. Wu, "Accelerating Self-Play Learning in Go", arXiv:1902.10565 (2019, rev. 2020); and
`docs/KataGoMethods.md`, KataGo GitHub (author's own documentation of post-paper methods).
- **Verification status: PRIMARY (author's docs).**

Practices relevant here:
1. **Soft resignation (anti-bias).** KataGo does *not* resign self-play games the way AlphaZero/Leela do. Past
the point where those would resign, KataGo "conducts faster and lower-weighted searches to finish the game… to
avoid introducing pathological biases in the training data", and the late positions receive **reduced frequency
weight** (< 1). Directly on point for task item (b): hard resignation truncates the losing side's late-game
distribution and biases value targets; the fix is cheaper search + down-weighting, not truncation.
2. **Short-term value/score targets (variance reduction).** Since the g170 run, KataGo trains auxiliary heads to
predict exponentially-averaged *future MCTS values*, (1−λ)·Σ λ^(t′−t)·MCTS_value(t′), at three horizons
(≈ next 6, 16, 50 turns on 19×19, scaled by board area), plus analogous score heads. Wu's stated rationale:
these provide **lower-variance feedback** for the main head that predicts the final outcome (λ→∞), and nets
train "slightly faster and achieve slightly better value and score loss on the main head" with them. This is
the same diagnosis as SSCI 2021 §1.4 (hard outcome targets are high-variance), answered differently:
keep the outcome target, but *add* graded, search-based intermediate targets.
3. **Policy surprise weighting (data weighting by teacher disagreement).** Positions whose search result was
"surprising" relative to the net's prior (KL between prior and search policy) get upweighted frequency in
training (half the total weight redistributed ∝ KL), deliberately *without* importance-sampling correction —
"biasing rare good moves upward… seems fairly innocuous". The value analogue is noted as experimental.
4. **Subtree value bias correction (online debiasing).** KataGo measures, per local-pattern bucket, the observed
signed error of the net's value vs deeper-search subtree values, and corrects node utilities online by
λ·(bucket bias), λ = 0.35; reported +30–60 Elo. The premise: the net has *persistent, pattern-specific*
evaluation biases that search keeps rediscovering.
5. **Draws / value target form.** KataGo's value utility is an expected score in [−1, 1] (win/loss plus a
score-margin component), so drawn games contribute a graded 0-centred signal rather than a third hard class;
games with no result are excluded from training. (Go-specific in detail; the transferable principle is
"graded target, exclude no-result games", matching dlshogi's discard policy in §3.1(4).)

**Implications for this program.**
- **Short-term targets are computable from existing records for free.** The generator already stores the root
search value s_t at every ply. Auxiliary targets ŝ_t^(λ) = (1−λ)Σ λ^k s_{t+k} (with sign flips) at horizons
scaled to dobutsu game length (e.g. mean horizons ≈ 3, 8, 25 plies) can be built from the *same* record files
and added as auxiliary heads or as a target blend — a zero-generation-cost Exp 5 arm aimed squarely at the
variance/compression problem, with two independent literature precedents (§1.4, this section).
- **Surprise weighting, value version:** upweight records by |s − v_net| (search-vs-net disagreement) at data
loading time. This targets training exactly at the positions where the net is wrong — the leak positions —
and needs no oracle and no new generation. (KataGo does the policy version and flags the value version as
plausible-but-unproven; mark accordingly if adopted.)
- **Bias audit before bias correction:** KataGo's correction presumes *pattern-specific persistent* bias. The
dobutsu audit can test the presumption cheaply: bucket audited positions by simple local features (e.g. which
piece moved last, lion files, hand pieces) and check whether the net's signed error vs oracle is consistent
within buckets across disjoint record sets. If yes, an online/offline debias is justified; if the error is
not pattern-stable, skip this direction.
- **If generation uses any early termination**, adopt the KataGo/dlshogi lesson as a verification item: compare
r-label agreement with oracle for positions in the last 10 plies vs mid-game; a phase-dependent gap indicates
adjudication/truncation bias in the targets.

---

## 4. Measuring strength differences with a few hundred games (task item c)

### 4.1 Error bars — the arithmetic (standard statistics; sources below)
- A match score s over N decisive+drawn games has SE(s) ≈ sqrt(s(1−s)/N) (draws counted 0.5 reduce variance
further under the pentanomial model). Converting to Elo at s ≈ 0.5, dElo/ds ≈ 695, so:
**N = 300 → SE ≈ 20 Elo, 95% CI ≈ ±39 Elo. N = 1,000 → ±22 Elo. N = 10,000 → ±7 Elo.**
A "few hundred games" therefore cannot certify the small improvements (≤ 20 Elo) that a target/backup change
plausibly yields; it can only detect large regressions/improvements.
- Variance reduction that practitioners actually use: **paired openings** — play each start position twice with
colours swapped (cutechess `-repeat`), which cancels first-move/colour bias (in dobutsu: the gote-win bias,
which is large — the theoretical result is a 78-ply gote win — so unpaired matches are especially noisy here).

### 4.2 SPRT (primary doc: cutechess-cli man page; secondary: Chess Programming Wiki "Match Statistics")
- cutechess-cli `-sprt elo0=E0 elo1=E1 alpha beta`: sequential test of H0 (P1 stronger by ≥ E0... per the man
page's wording, H1: P1 stronger than P2 by at least E0 Elo; H0: not stronger by at least E1) with type-I/II
error bounds alpha/beta outside [E0, E1]; the match stops as soon as the log-likelihood ratio crosses a bound.
Man page: https://github.com/cutechess/cutechess (docs/man/cutechess-cli.6) — text verified via the Debian man
page reproduction.
- Community-standard settings (Stockfish Fishtest practice, via CPW): elo0=0, elo1=5, α=β=0.05 for "is this an
improvement", typically requiring **thousands to tens of thousands of games** for small effects; wider windows
(elo1=20–50) terminate much faster but only certify large effects.
- **Verification status: man-page text primary; game-count folklore secondary (CPW / practitioner blogs).**

**Implication for this program.** The oracle audit is a *far* more sensitive instrument than game matches at
this scale: each audited position is an independent Bernoulli trial against ground truth, so 3,000 audited
positions give SE ≈ 0.9% on a conversion rate — enough to detect a 89.5% → 92% change, which a 300-game match
(±39 Elo) would almost surely miss. Recommended protocol, consistent with all sources above:
1. Use **audit deltas** (conversion / preservation rates, stratified by preserving-move count per §1.3) as the
primary accept/reject signal for Exp 5 arms.
2. Use **paired-start matches** (fixed set of random-start positions, colours swapped, e.g. 200 starts × 2 = 400
games) only as a sanity check that audit gains translate to play, reporting the ±Elo CI honestly.
3. Reserve SPRT (elo0=0, elo1≈10–20 given the small scale) for the final candidate-vs-baseline decision, with a
pre-registered game budget; do not tune on SPRT outcomes mid-run.

---

## 5. Consolidated verification checklist (what to measure before / alongside Exp 5)

1. Mate-solver firing rate and value/move change rate, by phase and oracle category (§1.1, §3.1). → decides
mate-depth arm vs PO arm.
2. Audit stratified by # of outcome-preserving moves (§1.3). → tests whether leaks = "fragile positions".
3. Signed value bias (net − oracle) in win vs loss positions (§2.1). → tests minimax max-bias; guards the
symmetry-augmentation arm against the GPW 2025 overestimation failure mode.
4. Target-variance profile: variance of (s, r) targets per position across generations, by category (§1.4).
→ tests the variance theory of compression before adopting soft/short-term targets.
5. Drawn-game record share and their r-label agreement with oracle (§3.1). → decides a draw-data filtering arm.
6. Phase profile of r-vs-oracle agreement (last 10 plies vs mid-game) (§3.1, §3.3). → detects adjudication /
truncation bias in targets.
7. Colour win-rate of generated games per generation vs audit expectation (§2.2). → free bias dashboard.
8. Pattern-bucket stability of net signed error (§3.3). → go/no-go for bias-correction ideas.

## 6. Verification-status summary
- Read in full (primary): GPW 2020 soft-value paper; GPW 2021 dark-check dobutsu paper; GPW 2025 mini-2048
Double-Q paper; dlshogi `selfplay/self_play.cpp`; KataGo `KataGoMethods.md` (relevant sections); cutechess
man-page text (SPRT section).
- Abstract only (full text paywalled, details marked): TAAI 2018 checkmate-search paper; TAAI 2020
PPO loss paper; SSCI 2021 max-entropy paper; Shioda & Ito TAAI 2020; Yamashita et al. GPW 2022 (abstract via
CiNii/Ceek); Terauchi & Matsuzaki TAAI 2023 (abstract, truncated).
- Abstract verified via index records; body numbers from the project owner's copy (previous round): GPW 2019
AlphaZero survey.
- Secondary: Yamaoka interview (NicoNico/Bunshun, 2021) — direct developer quotes; CPW "Match Statistics" for
SPRT practice norms.
