# Prior-art check: AlphaZero-style / search-based self-play candidates (dobutsu shogi testbed)

Research date: 2026-11-23. Method: web index searches in English and Japanese (やねうら王ブログ, dlshogi blog/WCSC appeal documents, GPW proceedings, NeurIPS/ICLR/arXiv, nnue-pytorch wiki). All statements below are `index` unless noted; no live-browser verification was needed for bibliographic facts, and effect sizes are as reported by the cited source, not re-measured.

## Summary

| Candidate | Verdict | One-line reason |
|---|---|---|
| A Budget distillation | **Partially done — exact form appears open** | RootStrap/TreeStrap (2009) fit a *raw* evaluator to a *deep* search and state the exact recursive motivation; TD-Leaf supplies gradient-through-the-PV-leaf. No source found fitting the output of a *shallow search through the net* (gradient through the backup) to a *deep* search in an AlphaZero-style system. |
| B Coverage-shaped self-play data | **Partially done — exact design appears open** | Branching-from-selected-positions (Uncertainty-Guided AlphaZero, NeurIPS 2025; Go-Explore), KataGo sample weighting, dlshogi's curated initial-position set and Shogi816K all shape coverage; the Lc0 King's Gambit test splits strength by opening family. No source found doing biased-subset vs uniformly-thinned, equal-budget, with strength split by distance-to-training-data. |
| C Diversity ablation (equal size) | **Appears open as a controlled A/B; premise contested** | Shogi816K (GPW 2023) is a controlled diversity experiment in shogi RL and cuts *against* YaneuraO's 2020 claim for robustness; practitioner gensfen practice uses *plausible* (multi-PV-bounded) randomisation, not uniform random. No equal-size, diversity-only A/B measuring strength/value accuracy was found. YaneuraO's claim itself is argued, not measured, in the source post. |
| D Evaluator × search budget | **Components established; the specific interaction is NOT established** | Elo-per-doubling / diminishing returns and train↔test compute trade-offs are well measured. Whether weaker evaluators gain *less or more* from search, and whether model gaps shrink under deep search, has no settled answer in the sources found; minimax-pathology theory even allows deeper search to hurt with a noisy evaluator. Ceiling effects in a solved game will mechanically shrink gaps. |
| E Mirror augmentation in shogi | **Used by practitioners; isolated effect appears unmeasured in dlshogi/elmo/AobaZero mainline** | WCSC appeal documents document random left–right flipping (sakura, Kanade, Ari) with explicitly *unknown / negligible* attributed effect; HalfKP_vm exploits the symmetry architecturally in NNUE. No published isolated effect size for dlshogi, elmo or AobaZero was found. |
| F Re-search relabelling of interior tree nodes | **Partially done — exact form appears open** | MuZero Reanalyse (trajectory positions) and dlshogi inference-blend relabelling are the known cases; TreeStrap trains *all interior nodes* but from the *same* search, without re-searching them as roots; Stockfish/Lc0 data rescoring relabels *game* positions with a fresh strong search/eval. Re-searching harvested *interior, unplayed* nodes as roots was not found. |

---

## Findings

### Candidate A — "Budget distillation" (fit shallow-search output to deep-search result)

**Closest works**

1. **Veness, Silver, Uther & Blair, "Bootstrapping from Game Tree Search", NeurIPS 2009** — *peer-reviewed*. RootStrap: stochastic gradient descent on squared error between the **static (raw) evaluation and the minimax/alpha-beta search value of the root**. TreeStrap: the same for **all interior nodes** of the tree, towards their recursive minimax values from a single depth-limited search. Implemented in the chess program Meep (linear evaluator, self-play). The paper's stated motivation is almost exactly Candidate A's recursion: "if the heuristic can be adjusted to match the value of a deep search of depth D, then a search of depth k with the new heuristic would be equivalent to a search of depth k + D with the old heuristic."
   - https://proceedings.neurips.cc/paper_files/paper/2009/file/389bc7bb1e1c2a5e7e147703232a88f6-Paper.pdf
   - Summary mirror: https://chessprogramming.org/Meep
2. **Baxter, Tridgell & Weaver, TD-Leaf / KnightCap (c.1998–2000)**, as described in Veness et al. above — *peer-reviewed lineage*. The leaf node of the principal variation is adjusted (gradient through that leaf) towards the search value at a subsequent time step. This is the "TD-leaf machinery" the candidate cites; note its target is a *later* search / outcome-linked value, not a contemporaneous deeper search.
3. **Anthony, Tian & Barber, "Thinking Fast and Slow with Deep Learning and Tree Search" (Expert Iteration / ExIt), NeurIPS 2017** — *peer-reviewed*. Apprentice = raw network, expert = tree search using that network; apprentice is trained to imitate the expert. Apprentice and expert differ in search (none vs search), but the student being trained is still the raw network, not a shallow search.
   - https://papers.neurips.cc/paper/7120-thinking-fast-and-slow-with-deep-learning-and-tree-search.pdf ; https://arxiv.org/pdf/1705.08439
4. **Wu, "Accelerating Self-Play Learning in Go" (KataGo), arXiv:1902.10565, 2019/2020** — *preprint / AAAI-RLG workshop*. **Playout cap randomization (PCR) — what it actually does, for contrast:** each move is a "full" search (e.g. 600–1000 playouts, prob. ~0.25) or a "fast" search (~100–200). **Only full-search moves enter training as policy targets; the value target remains the game outcome for all moves.** PCR therefore does *not* train any output to be consistent across search budgets, and does not use a deep-search value as a target. It is a data-economics trick (more games → more value labels per unit compute; ~1.37–1.4× speedup reported), not budget distillation.
   - https://ar5iv.labs.arxiv.org/html/1902.10565
5. **Schrittwieser et al., MuZero (Nature 2020) and Schrittwieser et al., "Online and Offline Reinforcement Learning by Planning with a Learned Model" (MuZero Unplugged / Reanalyse), NeurIPS 2021** — *peer-reviewed*. Reanalyse re-runs MCTS with the *current* network on *old trajectory states* to make fresh policy/value targets for the raw network. Contrast: student is raw, teacher budget is the normal budget with a newer network — not a fixed shallow-student vs deep-teacher budget pair.
   - https://openreview.net/references/pdf?id=KFgWCeHbZ
6. **Willemsen, Baier & Kaisers, "Value targets in off-policy AlphaZero: a new greedy backup?", Neural Computing and Applications 34:1801–1814 (2021/2022), DOI 10.1007/s00521-021-05928-5** — *peer-reviewed*. A family of AlphaZero value targets including search-value (greedy-backup) targets instead of the game outcome, tested on Connect-Four and Breakthrough; the raw value head is trained towards search-derived values. Again: raw student, search teacher.
   - https://ir.cwi.nl/pub/30870/
7. **Guez et al., "Learning to Search with MCTSnets", ICML 2018** — *peer-reviewed*. Search (expansion, evaluation, backup) is placed *inside* a differentiable network and trained end-to-end by gradient — but towards the task objective on Sokoban, not towards a deeper search's output. Relevant only as prior art for "gradient through a backup is implementable."
   - https://ar5iv.labs.arxiv.org/abs/1802.04697

**Verdict: partially done / exact experiment appears open.** Every ingredient exists separately: raw→deep search distillation (RootStrap/TreeStrap, ExIt, A0GB), gradient to a PV leaf (TD-Leaf), differentiable backup (MCTSnet). What was *not* found anywhere: making the **student itself a shallow search** — i.e. loss on `minimax_backup_k=10 evals (fθ leaves)` vs `search_100 evals`, with gradient flowing through the shallow backup to the leaves — in a neural AlphaZero-style loop. Novelty that remains: that specific student definition, the budget pair as an experimental variable, and any measurement of whether the resulting raw net differs from a RootStrap-trained net when both are then used at 0 / shallow / deep budgets (in dobutsu shogi the solver gives exact targets, which none of the above had).

**Premise-contradicting / cautionary evidence:** dlshogi's relabelling series (see F) found that replacing deep-search labels with *raw-inference* (0-search) labels, blended, *improved* test accuracy, with the author's hypothesis that deep-search results may be *harder for the network to learn* ("教師データは、必ずしも深く探索したデータがよいとは限らない"). That does not refute A (A changes the student, arguably precisely to make deep targets learnable), but it refutes the naive version of the premise "closer fit to deep search is straightforwardly better" and is worth citing as motivation.
- https://tadaoyamaoka.hatenablog.com/entry/2025/01/12/111242

### Candidate B — Coverage-shaped training data in self-play (biased subset vs uniform thinning; ID vs OOD strength)

**Closest works**

1. **Cheng, Tsai, Hong & Kandemir, "Uncertainty-Guided Exploration for Efficient AlphaZero Training", NeurIPS 2025** — *peer-reviewed* (known to the requester; included only for completeness of the branching family). Branches self-play from high-uncertainty positions (Label Change Rate), 10 variations/game best; +58.5% (9×9, early) / +47.3% (19×19, late) sample efficiency. It shapes coverage by *uncertainty*, not by opening/cluster restriction, and does not report an ID/OOD strength split.
   - https://papers.nips.cc/paper_files/paper/2025/file/3a02da3fdfd592d9a5273101c3546611-Paper-Conference.pdf
2. **Ecoffet, Huizinga, Lehman, Stanley & Clune, "First return, then explore" (Go-Explore), Nature 590:580–586, 2021** — *peer-reviewed*. Archive of promising states; return to an archived state, then explore from it. Demonstrated on Atari/robotics, **not** board-game self-play, and no ID/OOD playing-strength measurement.
   - https://arxiv.org/pdf/2004.12919
3. **KataGo methods (Wu 2019 + KataGoMethods.md)** — *preprint + practitioner*. Coverage/weighting schemes actually used: playout cap randomization (above); **policy surprise weighting** (overweight samples whose policy target was highly surprising vs the prior); training across board sizes, komi, rules and handicap games against weaker selves — i.e. deliberate distribution broadening, evaluated by overall strength, not by distance-to-training-data.
   - https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
4. **dlshogi practice (Yamaoka blog, 2020-11-28)** — *practitioner*. dlshogi deliberately uses an **initial-position set heavily loaded with middlegame/endgame positions (~500M positions)** because DL was weak there, uses mate-search results as reward, and adds alpha-beta software to the league; mixing AobaZero games (strong in openings/nyugyoku, weak in middle/endgame) with dlshogi data is hypothesised by the author to work precisely because the two datasets barely overlap in position space. This is coverage shaping by design, reported anecdotally, with no controlled ID/OOD split.
   - https://tadaoyamaoka.hatenablog.com/feed/category/AobaZero (entry "AobaZeroの棋譜の加工", 2020-11-28)
5. **Demura & Kaneko, Shogi816K, GPW 2023** — see Candidate C; it is also a B-relevant controlled study: pretraining on a hugely diversified initial-position set improved *average performance and robustness across human opening variations* — an OOD-flavoured evaluation, though by opening family, not by a distance metric.
   - https://ipsj.ixsq.nii.ac.jp/record/229354/files/IPSJ-GPWS2023021.pdf
6. **Lc0 "Make Leela Queen of openings", GitHub Issue #872 (2019)** — *practitioner, and the closest ID/OOD-style measurement found*. ~170k King's Gambit book games mixed into ~1M standard games, retrained, then matched at 800 nodes **separately** on King's Gambit openings vs other openings: ≈ **+17 to +20 Elo on the trained opening family (≈+40 for a smaller net), ≈ −1.4 / no loss on other openings**. This is a genuine in-family vs out-of-family strength split, for one opening family, in chess.
   - https://github.com/LeelaChessZero/lc0/issues/872

**Verdict: partially done / exact experiment appears open.** Shaping self-play coverage is standard practice (KataGo, dlshogi, UG-AZ, Go-Explore), and one practitioner test (Lc0 KG) splits strength by trained-vs-untrained opening family. Not found: the candidate's controlled comparison — *biased subset (some clusters only, natural duplication) vs uniformly thinned coverage at equal data/compute* — nor any board-game self-play study reporting strength as a function of a defined **distance-to-training-data**. That measurement design, especially with a solver-defined distance in dobutsu shogi, remains novel.

### Candidate C — Controlled ablation of training-data diversity (equal size; random vs plausible-only)

**Closest works**

1. **Demura & Kaneko, "将棋81万：強化学習のための多様性を持った将棋初期局面集 / Shogi816K: A Diverse Collection of Starting Positions for Reinforcement Learning in Shogi", Game Programming Workshop 2023, pp.111–118** — *peer-reviewed (workshop)*. 816,480 randomised initial positions (Chess960-like, with constraints, mirror duplicates removed). Gumbel AlphaZero, 10M self-play games, three arms: Base (normal start only), Shogi816K only, Shogi816K-pretrain (60%) → Base adaptation. Result: **pretrain-then-adapt beat Base-only on average performance and robustness across the various opening strategies seen in human games.** This is the single most on-point controlled diversity experiment in shogi RL.
   - https://ipsj.ixsq.nii.ac.jp/record/229354/files/IPSJ-GPWS2023021.pdf
2. **YaneuraO, "強化学習用の教師生成時のランダムムーブについて", やねうら王公式サイト, 2020-10-28** — *practitioner (the claim's source)*. Argues uniform random moves are harmful: with 30 legal first moves, uniform randomisation over-represents e.g. rook-swinging openings (6/30 rook moves; 1/30 for 7-8 rook) versus their real-game frequency, so "ランダムムーブを入れると実戦で進行しないような、多種多様な局面を学習してしまうので損" and, with limited model capacity, training should be deliberately biased to plausible positions ("多種多様なものを学習すれば賢くなる…は人間の認知バイアスにすぎず…実現性の高いものだけに絞って学習したほうが良い"). **Evidence offered in the post is argument plus an uncontrolled 2018 observation that starting from floodgate game positions was "少し強くなった" — no equal-size A/B, no numbers for the random-move comparison.** The claim as published is therefore *unsupported by controlled evidence in its own source*.
   - https://yaneuraou.yaneu.com/2020/10/28/random-moves-in-a-self-play/
3. **YaneuraOu gensfen practice** — *practitioner*. Teacher generation exposes `random_move_minply/maxply/count`, and crucially `random_multi_pv` (choose randomly among the top-N moves) — i.e. the deployed form of "randomness" is *plausibility-bounded*, not uniform. Stockfish's generator likewise: `generate_training_data depth 9 … random_multi_pv 4 random_multi_pv_diff 100 … book noob_3moves.epd` (nnue-pytorch wiki) — randomisation restricted to moves within 100 cp of best, from a 3-move book; later datasets use fixed 5000-node generation from UHO/DFRC books and are reported to produce "the best playing nets for the UHO books, and at least on par" otherwise. No published A/B isolating the random-move settings was found in either project.
   - https://deepwiki.com/yaneurao/YaneuraOu/6-learning-system
   - https://github-wiki-see.page/m/official-stockfish/nnue-pytorch/wiki/Training-datasets
4. **Lc0 King's Gambit forced-diversity test (Issue #872, above)** — *practitioner, measured*: adding off-distribution opening data gained strength in that family *without* losing strength elsewhere — direct evidence against a blanket "diversity wastes capacity" claim, at least for a large net and a modest admixture.
5. **"Study of the Proper NNUE Dataset", arXiv:2412.17948 (2024)** — *preprint* (Chinese chess NNUE). Argues for generating/filtering to *quiet*, tactically stable positions and against indiscriminate random position/score pairs — supports the "plausible/clean data" side, but its axis is position *quality/noise*, not opening diversity, and it is not an equal-size diversity A/B.
   - https://arxiv.org/html/2412.17948v1

**Verdict: appears open as specified; premise contested.** No published controlled test was found that holds dataset size fixed and varies *only* diversity (uniform-random starts vs plausible/game-like only) with strength or value accuracy as the outcome — in NNUE teacher generation or in AlphaZero self-play. The nearest controlled evidence (Shogi816K) points the *other way* from YaneuraO for the robustness/OOD criterion, while practitioner consensus (multi-PV-bounded randomisation, quiet-position filtering) supports him for the *in-distribution peak-strength* criterion. A well-designed dobutsu experiment should therefore pre-register which criterion it tests: the two literatures are arguably talking past each other (peak ID strength vs robustness), and an experiment that measures both would directly arbitrate the 2020 claim.

### Candidate D — Evaluator quality × search budget interaction

**What is established**

- **Diminishing, roughly log-linear Elo gains from search.** Thompson's Belle experiments (≈200 Elo per extra ply, early era) via Chess Programming Wiki "Depth / Diminishing Returns": https://chessprogramming.org/Depth . ELF OpenGo (Tian et al., "ELF OpenGo: An Analysis and Open Reimplementation of AlphaZero", arXiv:1902.04522): "even for the final model, doubling the rollouts in gameplay still boosts its strength by ≈200 Elo, indicating strength is constrained by model capacity": https://arxiv.org/html/1902.04522?fallback=original . KataGo's training site gives the rule of thumb "each doubling of speed may be worth somewhere around 150–250 Elo": https://katagotraining.org/
- **Train-time and test-time (search) compute trade off.** Jones, "Scaling Scaling Laws with Board Games", arXiv:2104.03113 (2021) — *preprint*: with AlphaZero/Hex, "for each extra order of magnitude of train-time compute, we can reduce test-time compute by a similar factor while leaving performance unchanged" (≈10× train ↔ ≈15× test in his summary). https://arxiv.org/html/2104.03113?fallback=original
- **Strength scales as a power law in model size/compute** (when not bottlenecked): Neumann & Gros, "Scaling Laws for a Multi-Agent Reinforcement Learning Model", ICLR 2023 (Connect Four, Pentago). https://iclr.cc/virtual/2023/poster/11489

**Closest direct design**

- **"On the Road to Perfection? Evaluating Leela Chess Zero Against Endgame Tablebases", Advances in Computer Games 2021** — *peer-reviewed*. Explicitly asks: how do stronger vs weaker networks differ, how does search improve each, how do stronger policies improve search, how does reducing the search budget affect correctness, and are there cases where search *hurts*? Uses tablebase ground truth in 3–4-piece endgames, intermediate vs strong net, raw policy vs full MCTS. This is the nearest published evaluator×budget factorial — but on move-accuracy against tablebases in tiny endgames, not Elo-vs-nodes curves across evaluator strengths. https://icga.org/wp-content/uploads/2021/11/ACG_2021_paper_13.pdf

**Theory that cuts against a simple monotonic story**

- **Minimax pathology**: Nau, "Pathology on Game Trees: A Summary of Results", AAAI-80, and Pearl (1983): under plausible error models, deeper minimax *amplifies* leaf-evaluation noise and can make decisions worse — i.e. a sufficiently noisy evaluator can gain *negatively* from search. Luštrek & Bratko's real-valued minimax work (and "Why Minimax Works", IJCAI 2005) explains why real game trees usually escape pathology (sibling-value dependence / value separation). https://ijcai.org/proceedings/2005/146 ; http://rakaposhi.eas.asu.edu/game-tree-pathology.pdf

**Verdict: the specific interaction claim is NOT established.** The literature establishes (i) search helps, with diminishing per-doubling returns, and (ii) evaluator quality and search are partial substitutes. It does **not** establish a sign for the interaction — "weaker evaluators gain less from search" and "weaker evaluators gain more (search compensates)" both circulate as folk claims, and the sources above support each in different regimes (compensation: Jones trade-off, MPV-MCTS-era observations that smaller/faster nets with more simulations can win under a fixed budget; less-to-gain / pathology: Nau/Pearl for noisy evaluators; ceiling: ELF's remark that the *strong* final model was still capacity-bound). Likewise "model differences shrink under deep search" is plausible and often asserted but was not found as a measured, controlled result across evaluator strengths. **Flag for the dobutsu design:** in a *solved* game, deep enough search reaches the solver for *every* evaluator, so gaps shrink to zero mechanically — the experiment must use budgets well below solving depth, or report value accuracy on non-trivial positions, for the interaction to be informative rather than tautological.

### Candidate E — Left–right mirror (左右反転) augmentation in shogi NNs

**What was found**

1. **sakura, WCSC31 appeal document (2021)** — *practitioner*. A dlshogi clone whose listed training tweaks include "Data Augmentationとしてランダムで局面を左右反転しています" (random left–right flipping of positions). The same document concludes most of its tweaks "ほとんど意味はなく" (were largely meaningless) and that teacher quality mattered more — **no isolated flip effect size, and the bundled impression is null-ish**.
   - https://www.apply.computer-shogi.org/wcsc31/appeal/sakura/sakura_appeal_2021.pdf
2. **Kanade, WCSC35 appeal document (2025)** — *practitioner*. Used *all* positions additionally left–right flipped as data augmentation in early training, and states explicitly that how much this contributed to accuracy is **not clear** ("この処理が精度向上にどの程度寄与しているかは明確ではありません"), with no controlled comparison performed.
   - https://www.apply.computer-shogi.org/wcsc35/appeal/Kanade/Kanade_appeal_detail_v2.pdf
3. **Ari Shogi and Friends, WCSC35 appeal document (2025)** — *practitioner*. Lists "左右反転などのデータ拡張" among implemented techniques for training a new DL evaluator; no effect measurement reported.
   - https://www.apply.computer-shogi.org/wcsc35/appeal/Ari_Shogi_and_Friends/Ari_Shogi_and_Friends_WCSC35_appeal_0.pdf
4. **HalfKP_vm (tanuki-, "マメット・ブンブク"), via YaneuraO blog 2022-05-17** — *practitioner*. Not augmentation but **architectural symmetry exploitation** in NNUE: when the king is on files 6–9, the board is fed left–right flipped, giving a symmetry-dimension reduction, fewer parameters and reportedly faster learning; subsequent verification found playing strength **roughly equal to Suisho** (standard NNUE). Note the follow-up correction in the same post: the team ultimately switched to a different architecture (halfkp_1024x2-8-32).
   - https://yaneuraou.yaneu.com/2022/05/17/the-shogi-ai-tournament-turned-into-a-book-match/
5. **dlshogi / elmo / AobaZero mainline:** no use of mirror augmentation, and no measurement, was found in the dlshogi blog/repo documentation or elmo/AobaZero descriptions searched. (Absence of evidence here — the dlshogi training code is public, so this is checkable in-source, but no documentation claims it.) AlphaZero's own symmetry handling is a useful contrast: it used board symmetries for Go but noted symmetry augmentation "does not hold for Chess or Shogi" in the full dihedral sense — the left–right reflection is precisely the one symmetry that *does* survive in shogi, which the Japanese practitioners above exploit.

**Verdict: used, not measured — the measurement appears open.** Mirror augmentation is established *practice* among dlshogi-derived WCSC entrants, but every source either bundles it with other changes, reports a null-ish overall impression, or explicitly says the contribution is unknown. No reported effect size (Elo or value/policy accuracy, with vs without, all else equal) was found for dlshogi, elmo, or AobaZero. A clean isolated measurement — trivially controlled in dobutsu shogi, where the symmetry is also exact — would be a genuine, if small, contribution; claiming the *technique* itself would not be novel.

### Candidate F — Re-search relabelling (harvest interior tree nodes, re-search each as root, replace label)

**Beyond the two known cases (MuZero Reanalyse; dlshogi inference-blend relabelling)**

1. **TreeStrap (Veness et al. 2009, above)** — *peer-reviewed, closest conceptual precedent for the "interior nodes" half*. All interior nodes of a search tree are trained, towards the minimax values computed by that same single search. Difference from F: the interior node's label comes from the *original* search's backup; it is **not** re-searched as a root with a fresh full budget. F's extra step (fresh root search per interior node) is exactly what TreeStrap avoids.
2. **Stockfish NNUE data rescoring / Lc0 relabelling (nnue-pytorch wiki)** — *practitioner*. Large Lc0 self-play datasets are **rescored**: positions are relabelled by a strong Lc0 network (the `relabel` branch, e.g. `linrock_relabel_1/2`, rescoring with a BT4-class net) and/or by Syzygy tablebases, and these rescored datasets "are of higher quality… and generally end up producing better networks" than Stockfish-generated data. Difference from F: the harvested positions are *game* positions from self-play records, not interior nodes of a search tree, and the new label is a network/TB rescoring rather than a fresh full root search per position (in most documented runs).
   - https://github-wiki-see.page/m/official-stockfish/nnue-pytorch/wiki/Training-datasets
3. **dlshogi relabelling series (Yamaoka, 2024–2025)** — *practitioner (known to requester; one new datum)*. The series relabels NNUE-generated teacher data with dlshogi model *inference* (policy/value/win-loss), blended by weighted average; part 5 (2025-01-12) shows that even relabelling dlshogi's *own* ~7500-node self-play data with the same model's **raw inference** improved test policy/value accuracy most at blend coefficient 0.9 — i.e. in this system *removing* search from the label helped learnability. This is evidence that F's direction (adding a fresh search to harvested positions) is not obviously beneficial and needs the learnability caveat from Candidate A.
   - https://tadaoyamaoka.hatenablog.com/entry/2025/01/12/111242
4. **Reanalyse itself (for the boundary)** re-searches *trajectory* states — positions that were actually played and stored — with the current model. Interior MCTS nodes are, by construction, mostly *unplayed* positions; no source was found that harvests them, deduplicates them, and re-searches each as a root to build a training set in a board-game AlphaZero/MuZero system. (Interior-node training *without* re-search is TreeStrap; trajectory re-search *without* interior harvesting is Reanalyse.)

**Verdict: partially done / exact experiment appears open.** The two halves of F each have strong prior art, and practitioner rescoring shows relabelling per se is established and beneficial in NNUE pipelines. The specific combination — **interior-node harvesting + fresh full root re-search as the label source, in board-game RL** — was not found. Remaining novelty: that combination, plus the natural controls it demands (vs TreeStrap-style same-search labels, vs Reanalyse-style trajectory-only re-search, vs raw-inference relabelling à la dlshogi, at equal label-generation compute).

---

## Could not verify

- No source was found that *proves* dlshogi mainline does/does not use mirror flipping in its current training code — only that no documentation claims it (Candidate E). A definitive answer requires reading the current dlshogi training source, not the web index.
- No controlled A/B for YaneuraOu gensfen `random_multi_pv` / random-move settings was found in blogs, GPW/CSA proceedings or the nnue-pytorch wiki; if one exists it is likely in Japanese forum/X posts not indexed by the searches run.
- The AlphaZero *Science* (2018) thinking-time-vs-Elo figure was not re-read in full text during this run; Candidate D therefore rests on ELF OpenGo, KataGo, Jones and the tablebase study rather than on that figure.
- Effect sizes quoted (Lc0 KG +17/+40 Elo; UG-AZ +58.5%/+47.3%; ELF +200 Elo/doubling) are as reported by their sources (`index`), not independently verified.

## Sources

Peer-reviewed / proceedings:
- Veness et al., Bootstrapping from Game Tree Search, NeurIPS 2009 — https://proceedings.neurips.cc/paper_files/paper/2009/file/389bc7bb1e1c2a5e7e147703232a88f6-Paper.pdf
- Anthony et al., Thinking Fast and Slow with Deep Learning and Tree Search, NeurIPS 2017 — https://arxiv.org/pdf/1705.08439
- Guez et al., Learning to Search with MCTSnets, ICML 2018 — https://ar5iv.labs.arxiv.org/abs/1802.04697
- Ecoffet et al., First return, then explore, Nature 2021 — https://arxiv.org/pdf/2004.12919
- Willemsen et al., Value targets in off-policy AlphaZero, Neural Comput. Appl. 2021 — https://ir.cwi.nl/pub/30870/
- Schrittwieser et al., Online and Offline RL by Planning with a Learned Model, NeurIPS 2021 — https://openreview.net/references/pdf?id=KFgWCeHbZ
- Neumann & Gros, Scaling Laws for a Multi-Agent RL Model, ICLR 2023 — https://iclr.cc/virtual/2023/poster/11489
- Cheng et al., Uncertainty-Guided Exploration for Efficient AlphaZero Training, NeurIPS 2025 — https://papers.nips.cc/paper_files/paper/2025/file/3a02da3fdfd592d9a5273101c3546611-Paper-Conference.pdf
- Demura & Kaneko, Shogi816K, GPW 2023 — https://ipsj.ixsq.nii.ac.jp/record/229354/files/IPSJ-GPWS2023021.pdf
- Evaluating Lc0 Against Endgame Tablebases, ACG 2021 — https://icga.org/wp-content/uploads/2021/11/ACG_2021_paper_13.pdf
- Nau, Pathology on Game Trees, AAAI-80 — http://rakaposhi.eas.asu.edu/game-tree-pathology.pdf ; Why Minimax Works, IJCAI 2005 — https://ijcai.org/proceedings/2005/146

Preprints:
- Wu, Accelerating Self-Play Learning in Go (KataGo), arXiv:1902.10565 — https://ar5iv.labs.arxiv.org/html/1902.10565
- Tian et al., ELF OpenGo, arXiv:1902.04522 — https://arxiv.org/html/1902.04522?fallback=original
- Jones, Scaling Scaling Laws with Board Games, arXiv:2104.03113 — https://arxiv.org/html/2104.03113?fallback=original
- Study of the Proper NNUE Dataset, arXiv:2412.17948 — https://arxiv.org/html/2412.17948v1

Practitioner:
- YaneuraO, random moves in teacher generation, 2020-10-28 — https://yaneuraou.yaneu.com/2020/10/28/random-moves-in-a-self-play/
- YaneuraO, WCSC32 / HalfKP_vm, 2022-05-17 — https://yaneuraou.yaneu.com/2022/05/17/the-shogi-ai-tournament-turned-into-a-book-match/
- Yamaoka, dlshogi relabelling part 5, 2025-01-12 — https://tadaoyamaoka.hatenablog.com/entry/2025/01/12/111242
- Yamaoka, AobaZero games processing / dlshogi initial-position set, 2020-11-28 — https://tadaoyamaoka.hatenablog.com/feed/category/AobaZero
- KataGo methods — https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md ; https://katagotraining.org/
- Stockfish nnue-pytorch, Training datasets wiki — https://github-wiki-see.page/m/official-stockfish/nnue-pytorch/wiki/Training-datasets
- YaneuraOu learning system (gensfen parameters) — https://deepwiki.com/yaneurao/YaneuraOu/6-learning-system
- Lc0 Issue #872, King's Gambit forced openings — https://github.com/LeelaChessZero/lc0/issues/872
- sakura WCSC31 appeal — https://www.apply.computer-shogi.org/wcsc31/appeal/sakura/sakura_appeal_2021.pdf
- Kanade WCSC35 appeal — https://www.apply.computer-shogi.org/wcsc35/appeal/Kanade/Kanade_appeal_detail_v2.pdf
- Ari Shogi WCSC35 appeal — https://www.apply.computer-shogi.org/wcsc35/appeal/Ari_Shogi_and_Friends/Ari_Shogi_and_Friends_WCSC35_appeal_0.pdf
- Chess Programming Wiki, Depth / Diminishing Returns — https://chessprogramming.org/Depth ; Meep — https://chessprogramming.org/Meep
