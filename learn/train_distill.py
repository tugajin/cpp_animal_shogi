# 予算の蒸留 用 学習ハーネス（train_distill.py）
# ============================================================
# 問い（実験候補 A）: 現在の学習は「モデルの素の値（0回探索）を、生成時の
# 深い探索（po=100）の結果 s に一致させる」ものだが、それを
# 「浅い探索（po=10）の出力を、深い探索の結果に近づける」学習に変える。
# 学生は素のネットではなく浅い探索そのもので、勾配は backup を通して
# 木の葉（ネットの評価）に流れる（TD-leaf の機械＋深い探索ターゲット）。
# 動機: dlshogi の示唆（深い探索ラベルは「読めないモデルには学習が難しい」、
# 山岡氏）への構造的な回答として、学生側に浅い探索を持たせる。
# 概念上の祖先は Veness et al. (NeurIPS 2009) の RootStrap/TreeStrap
# （深さの bootstrap 再帰）だが、学生が探索を持つ形は先行研究に無い。
#
# 仕組み:
#   1. 生成は arm 0 と同一（po=100 の自己対局、root レコード {p, s}）。
#      教師 s はその局面の po=100 探索結果そのもの。
#   2. 生成後、同じモデルで各 root 局面に po=10 の探索を回し、探索木を
#      C++ の dump モードで書き出す（ai/main.cpp / ai/selfplay.hpp）。
#   3. 学習: 木の葉のうち未解決のものを学習中モデルで評価し直し、
#      backup した root 値を教師 s に MSE で合わせる。
#      backup は ubfm.hpp の update_node の規則を厳密に再現する:
#        - 解決済みのノード（state != Unknown）は、その部分木ごと
#          ダンプされた w を定数として使う（証明はネットに依存しない）。
#        - 未解決の内部ノードに「負け確定の子」があれば、update_node の
#          早期リターンにより、子の並びで最初の負け確定の子の値で決まる
#          （最速の証明ではなく最初に見つかった証明。plain な max とは
#          ここだけが違う — v2 までは max で近似していたため、証明済みの
#          子を複数持つノードがある木の約8%で root_w を再現できなかった。
#          誤差は格子刻み 1/2000 の数倍以内で最大 0.001。2026-11-23 特定）。
#        - それ以外は子の符号反転の最大（draw 規則は max と一致する）。
#   4. --backup soft を選ぶと backup を Boltzmann 加重平均（温度 --tau）に
#      替える（勾配が全葉に分散する変種。探索自体は hard のまま）。
#
# 対照は arm 0（同じ生成・同じ局面・同じ予算で、素の値を s に合わせる）。
# 予算会計も arm 0 と同一（累計 5,000 局 / 100,000 局面、1 回 5,000 局面）。
# 引き分け対局の除外も root arm の規則をそのまま使う（千日手の scope-out）。
# r は使わない（教師は s のみ）。
#
# 実行（ホスト側 learn/ ディレクトリで、プロジェクトの .venv を使う）:
#   # 初期化（seed は全 arm と共通の 123）
#   ../.venv/bin/python train_distill.py --model-dir model/distill --init-only
#   # イテレーション（生成と dump は run_distill.sh が先に行う）
#   ../.venv/bin/python train_distill.py --model-dir model/distill \
#       --data-dir data/distill_gen0 --trees data/distill_gen0/trees.txt \
#       --init model/distill/best_single.h5 --iter-positions 5000
#   # dump の検証だけ行う（backup 再現性の関門。初回は必ず実行）
#   ../.venv/bin/python train_distill.py --model-dir model/distill \
#       --trees data/distill_gen0/trees.txt --verify

import argparse
import random
import sys
import time
from pathlib import Path

from train_arm import (build_model, evaluate_ans, filter_drawn, load_games,
                       load_state, save_model_files, save_state,
                       select_within_budget)

VERSION = "2026-11-23 v3（backup を update_node の厳密な規則に一致させた）"

NODE_UNKNOWN = -1  # ubfm::NodeState::NodeUnknown
NODE_DRAW = 0      # ubfm::NodeState::NodeDraw
NODE_WIN = 1       # ubfm::NodeState::NodeWin
NODE_LOSE = 2      # ubfm::NodeState::NodeLose


# ============================================================
# 純粋ロジック（torch / gamelibs 不要、self-test の対象）
# ============================================================

def parse_dump(text):
    """dump ファイルのテキストを解析する。
    戻り値: (trees, n_skip)。trees は root キー → {"root_w": float,
    "nodes": [(key, state, w, children), ...]}（nodes[0] が root、
    children はノード番号のタプル）。"""
    trees = {}
    n_skip = 0
    cur = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("DUMP_PARAMS"):
            continue
        if line.startswith("TREE_SKIP"):
            n_skip += 1
            cur = None
            continue
        if line.startswith("TREE "):
            _t, key, root_w = line.split()
            cur = {"root_w": float(root_w), "nodes": {}}
            trees[int(key)] = cur
            continue
        if line.startswith("N ") and cur is not None:
            parts = line.split()
            idx = int(parts[1])
            key = int(parts[2])
            state = int(parts[3])
            w = float(parts[4])
            n_child = int(parts[5])
            children = tuple(int(x) for x in parts[6:6 + n_child])
            cur["nodes"][idx] = (key, state, w, children)
            continue
        if line == "END":
            cur = None
            continue
    for tree in trees.values():
        nodes = tree["nodes"]
        tree["nodes"] = [nodes[i] for i in range(len(nodes))]
    return trees, n_skip


def backup_float(nodes, leaf_value):
    """float 版 backup（update_node の厳密な再現）。leaf_value は未解決の
    葉のキー → 値を返す関数。解決済みのノードは部分木ごとダンプされた w
    を定数として使う。未解決の内部ノードに負け確定の子があれば、子の
    並びで最初の負け確定の子の符号反転が値になる（update_node の早期
    リターン）。それ以外は子の符号反転の最大。"""
    memo = {}

    def rec(i):
        if i in memo:
            return memo[i]
        key, state, w, children = nodes[i]
        if state != NODE_UNKNOWN:
            v = w
        elif not children:
            v = leaf_value(key)
        else:
            vals = [rec(c) for c in children]
            v = None
            for ci, c in enumerate(children):
                if nodes[c][1] == NODE_LOSE:
                    v = -vals[ci]
                    break
            if v is None:
                v = max(-x for x in vals)
        memo[i] = v
        return v

    return rec(0)


def verify_trees(trees, tol=1.0e-6):
    """検証: 未解決の葉にダンプされた w を入れると root_w が再現されるか。
    戻り値: (検査した木の数, tol を超えた木の数, 最大誤差)。"""
    n_bad = 0
    max_err = 0.0
    for tree in trees.values():
        nodes = tree["nodes"]
        dumped = {}
        for key, state, w, children in nodes:
            if not children and state == NODE_UNKNOWN:
                dumped[key] = w
        v = backup_float(nodes, lambda k: dumped[k])
        err = abs(v - tree["root_w"])
        max_err = max(max_err, err)
        if err > tol:
            n_bad += 1
    return len(trees), n_bad, max_err


def recompute_all(nodes):
    """全ノードの backup 再計算値を返す（未解決の葉はダンプ値を使う）。
    verify が通らない木の切り分け用: 戻り値 recomputed[i] と
    ダンプされた w が食い違うノードのうち、「子は全て一致しているのに
    自分だけ食い違う」ノードが backup 規則の破れの発生点。"""
    recomputed = [None] * len(nodes)

    def rec(i):
        if recomputed[i] is not None:
            return recomputed[i]
        _key, state, w, children = nodes[i]
        if state != NODE_UNKNOWN or not children:
            v = w
        else:
            vals = [rec(c) for c in children]
            v = None
            for ci, c in enumerate(children):
                if nodes[c][1] == NODE_LOSE:
                    v = -vals[ci]
                    break
            if v is None:
                v = max(-x for x in vals)
        recomputed[i] = v
        return v

    rec(0)
    return recomputed


def diagnose_dump(trees, tol=1.0e-6, max_report=5):
    """root_w を再現できない木について、破れの発生点を報告する。
    戻り値: (最悪の木のキー, 最大誤差, 発生点のリスト)。"""
    worst_key, worst_err, worst = None, -1.0, None
    for key, tree in trees.items():
        nodes = tree["nodes"]
        dumped = {k: w for k, st, w, ch in nodes
                  if not ch and st == NODE_UNKNOWN}
        err = abs(backup_float(nodes, lambda k: dumped[k]) - tree["root_w"])
        if err > worst_err:
            worst_key, worst_err, worst = key, err, tree
    if worst is None or worst_err <= tol:
        return worst_key, worst_err, []
    nodes = worst["nodes"]
    recomputed = recompute_all(nodes)
    origins = []
    for i, (key, state, w, children) in enumerate(nodes):
        if abs(recomputed[i] - w) <= tol:
            continue
        if all(abs(recomputed[c] - nodes[c][2]) <= tol for c in children):
            origins.append(i)
    report = []
    for i in origins[:max_report]:
        key, state, w, children = nodes[i]
        kids = [(nodes[c][0], nodes[c][1], nodes[c][2]) for c in children]
        report.append((i, key, state, w, recomputed[i], kids))
    return worst_key, worst_err, report


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'OK' if cond else 'NG'}: {name}")
        ok = ok and cond

    # 単純な2手読み: 葉 0.3 / -0.5 → root は max(-0.3, 0.5) = 0.5
    text = (
        "DUMP_PARAMS positions=1 po_num=10 workers=12\n"
        "TREE 100 0.500000000\n"
        "N 0 100 -1 0.500000000 2 1 2\n"
        "N 1 101 -1 0.300000000 0\n"
        "N 2 102 -1 -0.500000000 0\n"
        "END\n"
        "TREE_SKIP 999\n"
    )
    trees, n_skip = parse_dump(text)
    check("parse: 木1件と SKIP 1件", len(trees) == 1 and n_skip == 1)
    n, bad, err = verify_trees(trees)
    check("verify: ダンプ値で root_w を再現", n == 1 and bad == 0 and err < 1e-9)

    # 解決済みの葉は定数: 子が負け確定（state=2, w=-0.9994）なら
    # leaf_value が何であれ root は 0.9994
    nodes = [(100, -1, 0.0, (1,)), (101, 2, -0.9994, ())]
    v = backup_float(nodes, lambda k: 0.0)
    check("解決済みの葉は定数として効く", abs(v - 0.9994) < 1e-12)

    # 引き分け規則の再現: 子が draw（w=0）と勝ち確定（w=+0.9994）なら
    # plain max でも root は 0（update_node の draw 規則と一致）
    nodes = [(100, -1, 0.0, (1, 2)), (101, 0, 0.0, ()), (102, 1, 0.9994, ())]
    v = backup_float(nodes, lambda k: 9.9)
    check("draw+勝ち確定の子で root は 0", abs(v - 0.0) < 1e-12)

    # 3手読み: A = max(-0.2, -0.4) = -0.2、root = max(-A, -0.1) = 0.2
    nodes = [
        (100, -1, 0.0, (1, 2)),
        (101, -1, 0.0, (3, 4)),
        (102, -1, 0.1, ()),
        (103, -1, 0.2, ()),
        (104, -1, 0.4, ()),
    ]
    leaf = {102: 0.1, 103: 0.2, 104: 0.4}
    v = backup_float(nodes, lambda k: leaf[k])
    check("3手読みの backup", abs(v - 0.2) < 1e-12)

    # 葉の値を差し替えると root が変わる（学習の勾配が届く先の確認）。
    # 103 を -0.6 にすると A = max(0.6, -0.4) = 0.6 になり、
    # root = max(-0.6, -0.1) = -0.1（符号反転が2回入る）。
    leaf2 = {102: 0.1, 103: -0.6, 104: 0.4}
    v2 = backup_float(nodes, lambda k: leaf2[k])
    check("葉の差し替えが root に伝わる", abs(v2 - (-0.1)) < 1e-12)

    # update_node の早期リターン: 負け確定の子が複数あり証明の深さが
    # 違う場合、値は「最初の負け確定の子」で決まる（plain max ではない）。
    # 子: Lose w=-0.9984（先）/ Lose w=-0.9994（後）→ root は 0.9984。
    nodes = [
        (100, -1, 0.9984, (1, 2)),
        (101, 2, -0.9984, ()),
        (102, 2, -0.9994, ()),
    ]
    v = backup_float(nodes, lambda k: 9.9)
    check("負け確定の子は最初の子が優先（update_node の早期リターン）",
          abs(v - 0.9984) < 1e-12)

    # 解決済みの内部ノードは部分木ごと定数: 勝ち確定の内部ノードの配下に
    # 未解決の葉があっても、その値は root に影響しない。
    nodes = [
        (100, -1, 0.0, (1,)),
        (101, 1, 0.9990, (2,)),
        (102, -1, 0.0, ()),
    ]
    v = backup_float(nodes, lambda k: 9.9)
    check("解決済みの内部ノードは定数として効く", abs(v - (-0.9990)) < 1e-12)

    print("self-test:", "OK" if ok else "NG")
    return ok


# ============================================================
# モード: キーファイル作成 / 検証
# ============================================================

def make_keys(args):
    games, n_skipped = load_games(args.data_dir)
    keys = []
    seen = set()
    for _name, records in games:
        for rec in records:
            p = rec["p"]
            if p not in seen:
                seen.add(p)
                keys.append(p)
    out = Path(args.keys_out)
    with out.open("w", encoding="utf-8") as fh:
        for k in keys:
            fh.write(f"{k}\n")
    print(f"make-keys: {args.data_dir} から {len(games):,} 局 / "
          f"異なる root キー {len(keys):,} 件を {out} に書きました"
          f"（読めなかった・空: {n_skipped:,} ファイル）")
    return True


def verify(args):
    text = Path(args.trees).read_text(encoding="utf-8")
    trees, n_skip = parse_dump(text)
    n, bad, err = verify_trees(trees)
    print(f"verify: 木 {n:,} 件（SKIP {n_skip:,} 件）/ "
          f"再現できない木 {bad:,} 件 / 最大誤差 {err:.2e}")
    if bad:
        print("** backup が再現できない木があります。中止してください **")
        return False
    print("verify: OK（ダンプした葉の値で全ての root_w が再現されました）")
    return True


def diagnose(args):
    text = Path(args.trees).read_text(encoding="utf-8")
    trees, n_skip = parse_dump(text)
    key, err, report = diagnose_dump(trees)
    print(f"diagnose: 木 {len(trees):,} 件（SKIP {n_skip:,} 件）")
    if err < 0:
        print("diagnose: 木がありません")
        return False
    print(f"diagnose: 最悪の木 key={key} / root の誤差 {err:.6f}")
    if not report:
        print("diagnose: tol 超えの発生点はありません（全て再現できます）")
        return True
    state_name = {-1: "Unknown", 0: "Draw", 1: "Win", 2: "Lose"}
    for i, k, state, w, rec_v, kids in report:
        print(f"  発生点 node {i}: key={k} state={state_name.get(state, state)} "
              f"dump_w={w:.6f} recomputed={rec_v:.6f}")
        for ck, cs, cw in kids:
            print(f"    子: key={ck} state={state_name.get(cs, cs)} w={cw:.6f}")
    return True


# ============================================================
# 学習
# ============================================================

def backup_torch(nodes, value_of, torch, backup="hard", tau=0.1):
    """torch 版 backup。value_of は未解決の葉のキー → テンソル（スカラー）。
    解決済みのノードは部分木ごと定数（勾配は流れない。証明はネットに
    依存しないため）。
    hard: update_node の厳密な再現 — 負け確定の子があれば最初の子の
    符号反転、それ以外は符号反転の max（勾配は選ばれた子に流れる）。
    soft: 符号反転した子値の Boltzmann 加重平均（勾配が全葉に分散する
    変種。負け確定の子の早期リターンは適用しない）。"""
    memo = {}

    def rec(i):
        if i in memo:
            return memo[i]
        key, state, w, children = nodes[i]
        if state != NODE_UNKNOWN:
            v = torch.tensor(w, dtype=torch.float32,
                             device=value_of.device)
        elif not children:
            v = value_of[key]
        else:
            flipped = torch.stack([-rec(c) for c in children])
            if backup == "soft":
                weights = torch.softmax(flipped / tau, dim=0)
                v = torch.sum(weights * flipped)
            else:
                v = None
                for ci, c in enumerate(children):
                    if nodes[c][1] == NODE_LOSE:
                        v = flipped[ci]
                        break
                if v is None:
                    v = torch.max(flipped)
        memo[i] = v
        return v

    return rec(0)


class ValueMap:
    """キー → テンソルの対応を、バッチ先頭の forward 結果から引くための薄い器。
    backup_torch から .device と __getitem__ だけを使う。"""

    def __init__(self, keys, values):
        self.index = {k: i for i, k in enumerate(keys)}
        self.values = values
        self.device = values.device

    def __getitem__(self, key):
        return self.values[self.index[key]]


def train(args):
    import numpy as np
    import torch
    import torch.optim as optim

    from game import from_hash
    from single_network import DN_INPUT_SHAPE

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(model_dir)
    if state["arm"] is not None and state["arm"] != "D":
        print(f"** arm_state.json の arm={state['arm']} と蒸留用 model-dir が"
              "食い違うため中止（model-dir の取り違え防止）**")
        return False
    state["arm"] = "D"

    log_lines = []

    def log(line=""):
        log_lines.append(line)
        print(line)

    log(f"## train_distill: iteration {state['iterations']}"
        f"（{time.strftime('%Y-%m-%d %H:%M:%S')}）")
    log()

    if args.init_only:
        model = build_model("random", args.seed)
        h5, onnx, jit = save_model_files(model_dir, model)
        save_state(model_dir, state)
        log(f"- 初期モデル（seed={args.seed} の乱数初期化）を保存: {h5} / {onnx} / {jit}")
        log(f"- 累計: {state['games']:,} 局 / {state['positions']:,} 局面")
        with (model_dir / "train_log.md").open("a", encoding="utf-8") as fh:
            fh.write("\n".join(log_lines) + "\n\n")
        return True

    if args.data_dir is None or args.trees is None or args.init is None:
        print("** --data-dir と --trees と --init が必要です"
              "（初期化だけなら --init-only）**")
        return False

    # ---- データ準備（arm 0 と同じ規則: 引き分け除外 → 予算で間引き） ----
    games, n_skipped = load_games(args.data_dir)
    total_positions = sum(len(records) for _, records in games)
    log(f"- データ: `{args.data_dir}` から {len(games):,} 局 / {total_positions:,} 局面"
        f"（読めなかった・空: {n_skipped:,} ファイル）")
    if not games:
        log("** 対局が0件のため中止 **")
        with (model_dir / "train_log.md").open("a", encoding="utf-8") as fh:
            fh.write("\n".join(log_lines) + "\n\n")
        return False
    games, n_drawn, _detectable = filter_drawn(games, 0)
    log(f"- 引き分け対局の除外: {n_drawn:,} 局（残り {len(games):,} 局）")
    selected, sel_positions, bound = select_within_budget(
        games, state, args.budget_games, args.budget_positions, args.seed,
        iter_positions=args.iter_positions)
    if not selected:
        log("** 累計予算を使い切っているため中止（arm_state.json を確認）**")
        with (model_dir / "train_log.md").open("a", encoding="utf-8") as fh:
            fh.write("\n".join(log_lines) + "\n\n")
        return False
    bound_name = {"games": "対局数上限", "positions": "局面数上限",
                  "iter": "イテレーション上限",
                  "all": "母集団を使い切った", "exhausted": "予算切れ"}[bound]
    log(f"- 予算で間引き: {len(selected):,} 局 / {sel_positions:,} 局面"
        f"（上限に当たったのは {bound_name}）")

    # ---- 木の読み込みとサンプル構築 ----
    trees, n_skip = parse_dump(Path(args.trees).read_text(encoding="utf-8"))
    log(f"- 木: {len(trees):,} 件（SKIP {n_skip:,} 件、学生 po={args.student_po}）")
    _n_trees, n_bad, max_err = verify_trees(trees)
    if n_bad:
        log(f"** 警告: backup を再現できない木が {n_bad:,} 件あります"
            f"（最大誤差 {max_err:.4f}）。--diagnose で発生点を確認してください **")
    samples = []
    n_missing = 0
    for _name, records in selected:
        for rec in records:
            tree = trees.get(rec["p"])
            if tree is None:
                n_missing += 1
                continue
            samples.append((rec["p"], float(rec["s"]), tree["nodes"]))
    log(f"- 学習サンプル: {len(samples):,} 件（木が無いレコード {n_missing:,} 件を除外）")
    if not samples:
        log("** サンプルが0件のため中止 **")
        with (model_dir / "train_log.md").open("a", encoding="utf-8") as fh:
            fh.write("\n".join(log_lines) + "\n\n")
        return False

    # ---- 特徴量のキャッシュ（葉のキーは木の間で何度も現れる） ----
    file, rank, channel = DN_INPUT_SHAPE
    feature_cache = {}

    def feature_of(key):
        feat = feature_cache.get(key)
        if feat is None:
            feat = np.array(from_hash(key).feature()).reshape(channel, file, rank)
            feature_cache[key] = feat
        return feat

    def leaf_keys_of(nodes):
        return [key for key, st, _w, ch in nodes
                if not ch and st == NODE_UNKNOWN]

    # ---- 学習 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.init, args.seed).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.00001)
    rng = random.Random(args.seed)

    start = time.time()
    avg_loss = None
    for epoch in range(args.epochs):
        order = list(range(len(samples)))
        rng.shuffle(order)
        sum_loss = 0.0
        sum_gap = 0.0
        n_batch = 0
        for b0 in range(0, len(order), args.batch_size):
            batch = [samples[i] for i in order[b0:b0 + args.batch_size]]
            keys = []
            seen = set()
            for _p, _s, nodes in batch:
                for k in leaf_keys_of(nodes):
                    if k not in seen:
                        seen.add(k)
                        keys.append(k)
            feats = torch.from_numpy(
                np.stack([feature_of(k) for k in keys])).float().to(device)
            model.train()
            values = torch.squeeze(model(feats))
            if values.dim() == 0:
                values = values.reshape(1)
            value_of = ValueMap(keys, values)
            loss = None
            for _p, s, nodes in batch:
                root_val = backup_torch(nodes, value_of, torch,
                                         backup=args.backup, tau=args.tau)
                term = (root_val - s) ** 2
                loss = term if loss is None else loss + term
                sum_gap += abs(root_val.item() - s)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            n_batch += 1
        avg_loss = sum_loss / max(n_batch, 1)
        mean_gap = sum_gap / len(samples)
        print(f"epoch:{epoch} avg loss:{avg_loss:.4f} "
              f"mean|backup-s|:{mean_gap:.4f} sec:{int(time.time() - start)}")

    # ---- 保存・評価・状態の更新 ----
    h5, onnx, jit = save_model_files(model_dir, model)
    log(f"- 保存: {h5} / {onnx} / {jit}（最終 epoch の平均損失 {avg_loss:.4f}）")
    ans = evaluate_ans(model_dir)
    if ans is not None:
        log(f"- ans（参考）: 全体 {ans['ans_all']:.4f}（{ans['n_all']:,} 局面） / "
            f"win/lose のみ {ans['ans_winlose']:.4f}（{ans['n_winlose']:,} 局面）")
    else:
        log("- ans: ../oracle/problem.txt が無いため未実施")
    state["games"] += len(selected)
    state["positions"] += sel_positions
    state["iterations"] += 1
    save_state(model_dir, state)
    log(f"- 累計: {state['games']:,} 局 / {state['positions']:,} 局面"
        f"（予算 {args.budget_games:,} 局 / {args.budget_positions:,} 局面）")
    log()
    with (model_dir / "train_log.md").open("a", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="予算の蒸留 用 学習ハーネス")
    parser.add_argument("--model-dir", required=True,
                        help="蒸留専用のモデル出力先（例: model/distill）")
    parser.add_argument("--data-dir", default=None,
                        help="このイテレーションの対局データ（*.json のあるディレクトリ）")
    parser.add_argument("--trees", default=None,
                        help="C++ dump モードが出した木ファイル")
    parser.add_argument("--init", default=None,
                        help="初期重み: state_dict のパス、または random")
    parser.add_argument("--init-only", action="store_true",
                        help="乱数初期モデルを作って保存するだけ（学習しない）")
    parser.add_argument("--make-keys", action="store_true",
                        help="data-dir の異なる root キーを --keys-out に書くだけ")
    parser.add_argument("--keys-out", default="keys.txt")
    parser.add_argument("--verify", action="store_true",
                        help="--trees の backup 再現性だけを検証して終わる")
    parser.add_argument("--diagnose", action="store_true",
                        help="再現できない木について backup の破れの発生点を報告する")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="1 バッチの木の数")
    parser.add_argument("--budget-games", type=int, default=5000)
    parser.add_argument("--budget-positions", type=int, default=100000)
    parser.add_argument("--iter-positions", type=int, default=5000,
                        help="1イテレーションで学習する局面数の上限")
    parser.add_argument("--student-po", type=int, default=10,
                        help="学生の探索予算（dump の po と一致させる。記録用）")
    parser.add_argument("--backup", choices=["hard", "soft"], default="hard",
                        help="学習時の backup（探索自体は常に hard）")
    parser.add_argument("--tau", type=float, default=0.1,
                        help="--backup soft の温度")
    args = parser.parse_args()

    print(f"train_distill.py {VERSION}")
    if not self_test():
        print("** self-test 失敗のため中止 **")
        sys.exit(1)
    if args.make_keys:
        if args.data_dir is None:
            print("** --make-keys には --data-dir が必要です **")
            sys.exit(1)
        sys.exit(0 if make_keys(args) else 1)
    if args.verify:
        if args.trees is None:
            print("** --verify には --trees が必要です **")
            sys.exit(1)
        sys.exit(0 if verify(args) else 1)
    if args.diagnose:
        if args.trees is None:
            print("** --diagnose には --trees が必要です **")
            sys.exit(1)
        sys.exit(0 if diagnose(args) else 1)
    if not train(args):
        sys.exit(1)


if __name__ == "__main__":
    main()
