# 柔らかいバックアップの診断（soft_backup_diag.py）
# ====================================
# 問い: 探索そのものは今のまま（硬い minimax）にして、学習の教師に
# する値だけを「柔らかいバックアップ」（子の値の Boltzmann 加重平均）
# で計算し直すと、正解（oracle）との一致は硬い教師より良くなるか。
# 文献では未検証の点（探索を硬く保ったまま教師だけ柔らかくする形）。
#
# 方法: 記録済みの探索木（予算蒸留の dump、data/distill_gen*/trees.txt）
# を使い、葉の値はダンプされた w のまま（モデルの再評価はしない）、
# 内部ノードの値を2通りで計算し直して root の値を比べる。
#   - 硬い: train_distill.py v3 と同じ update_node の厳密な規則。
#     ダンプされた root_w が再現されることが前提（最初に検証する）
#   - 柔らかい: 解決済みノードは定数のまま、未解決の内部ノードでは
#     子の符号反転値の Boltzmann 加重平均（温度 τ）。
#     最初の負け確定の子で決める規則は使わない（別の演算子として定義）
# root のカテゴリ（±0.2）が oracle と一致する割合を、硬い教師と
# 各 τ で比べる。主たる対象は root が未解決の木（解決済みの root は
# どちらの計算でも定数で同じになる）。
#
# 読み方（事前登録）:
# - ある τ で柔らかいほうが未解決 root の一致率が明確に上がる
#   → 柔らかい教師は学習実験の候補になる
# - どの τ でも硬いほう以下 → 教師を柔らかくする線はここで閉じる
# - |柔らかい − 硬い| の平均（ドリフト）が τ に対して大きい場合、
#   柔らかい値はゲームの値としての意味がずれる。その是非も一致率で判断する
#
# 実行（learn/ ディレクトリで）:
#   ../.venv/bin/python soft_backup_diag.py
#   ../.venv/bin/python soft_backup_diag.py --glob "data/distill_gen*/trees.txt" --out soft_backup_result.md

import argparse
import glob
import math
import sys
from pathlib import Path

VERSION = "2026-11-25 v1"

NODE_UNKNOWN = -1  # ubfm::NodeState::NodeUnknown
NODE_LOSE = 2      # ubfm::NodeState::NodeLose

DEFAULT_TAUS = [0.02, 0.05, 0.1, 0.2, 0.5]

md_lines = []


def md(line=""):
    md_lines.append(line)
    print(line)


def pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


def cat_of_value(v):
    """プロジェクト共通のカテゴリ規約（±0.2）"""
    if v > 0.2:
        return 1
    if v < -0.2:
        return -1
    return 0


# ============================================================
# 純粋ロジック（self-test の対象。parse_dump / backup_float は
# train_distill.py v3 と同一の規則）
# ============================================================

def parse_dump(text):
    """戻り値: (trees, n_skip)。trees は root キー →
    {"root_w": float, "nodes": [(key, state, w, children), ...]}"""
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


def leaf_map(nodes):
    """未解決の葉のキー → ダンプされた w"""
    return {k: w for k, st, w, ch in nodes
            if not ch and st == NODE_UNKNOWN}


def backup_float(nodes, leaf_value):
    """硬い backup（update_node の厳密な再現）"""
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


def backup_soft(nodes, leaf_value, tau):
    """柔らかい backup: 解決済みノードは定数、未解決の内部ノードは
    子の符号反転値の Boltzmann 加重平均（温度 tau）"""
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
            vals = [-rec(c) for c in children]
            m = max(vals)
            exps = [math.exp((x - m) / tau) for x in vals]
            v = sum(e * x for e, x in zip(exps, vals)) / sum(exps)
        memo[i] = v
        return v

    return rec(0)


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("  OK: " if cond else "  NG: ") + name)
        ok = ok and cond

    check("カテゴリ規約", cat_of_value(0.21) == 1 and cat_of_value(-0.21) == -1
          and cat_of_value(0.2) == 0)

    text = (
        "DUMP_PARAMS positions=1\n"
        "TREE 111 0.998400\n"
        "N 0 111 -1 0.998400 2 1 2\n"
        "N 1 222 2 -0.998400 0\n"
        "N 2 333 2 -0.999400 0\n"
        "END\n"
        "TREE_SKIP 999\n"
    )
    trees, n_skip = parse_dump(text)
    check("parse_dump", n_skip == 1 and len(trees) == 1
          and trees[111]["nodes"][0][3] == (1, 2))
    # 硬い backup: 最初の負け確定の子（-0.998400 の符号反転）が勝ち、
    # 大きいほう（0.999400）ではない
    v = backup_float(trees[111]["nodes"], lambda k: 0.0)
    check("硬い backup（最初の負け確定の子）", abs(v - 0.9984) < 1e-9)

    # 柔らかい backup の手計算: 子（葉）の値 +0.6 / -0.2（子側視点）
    # 符号反転すると -0.6 / +0.2。τ=0.1 では最大 0.2 のすぐ下、
    # τ=10 では平均 -0.2 の近くになる
    nodes = [
        (1000, NODE_UNKNOWN, 0.0, (1, 2)),
        (1001, NODE_UNKNOWN, 0.6, ()),
        (1002, NODE_UNKNOWN, -0.2, ()),
    ]
    leaves = leaf_map(nodes)
    hard = backup_float(nodes, leaves.__getitem__)
    check("硬い backup（通常の最大）", abs(hard - 0.2) < 1e-12)
    soft01 = backup_soft(nodes, leaves.__getitem__, 0.1)
    check("柔らかい backup τ=0.1（最大のすぐ下）",
          0.19 < soft01 < 0.2)
    soft100 = backup_soft(nodes, leaves.__getitem__, 100.0)
    check("柔らかい backup τ=100（平均の近く）",
          abs(soft100 - (-0.2)) < 0.01)
    check("柔らかいは最大以下・最小以上",
          -0.6 <= soft01 <= 0.2 and -0.6 <= soft100 <= 0.2)

    # 解決済みの root はどちらの計算でも定数
    nodes_r = [
        (2000, 1, 0.9994, (1,)),
        (2001, NODE_UNKNOWN, -0.9, ()),
    ]
    leaves_r = leaf_map(nodes_r)
    check("解決済み root は定数",
          backup_float(nodes_r, leaves_r.__getitem__) == 0.9994
          and backup_soft(nodes_r, leaves_r.__getitem__, 0.1) == 0.9994)
    return ok


def main():
    parser = argparse.ArgumentParser(description="柔らかいバックアップの診断")
    parser.add_argument("--glob", action="append", default=None,
                        help="trees.txt のパターン（複数指定可）")
    parser.add_argument("--taus", default=None,
                        help="カンマ区切りの温度リスト（既定: 0.02,0.05,0.1,0.2,0.5）")
    parser.add_argument("--out", default="soft_backup_result.md")
    args = parser.parse_args()
    patterns = args.glob if args.glob else ["data/distill_gen*/trees.txt"]
    taus = ([float(x) for x in args.taus.split(",")] if args.taus
            else DEFAULT_TAUS)

    print(f"soft_backup_diag.py {VERSION}（柔らかいバックアップの診断）")
    print("self-test:")
    if not self_test():
        print("self-test: NG のため中止")
        return 1
    print("self-test: OK")

    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        print(f"中止: パターン {patterns} に trees.txt がありません。"
              "予算蒸留の生成データ（data/distill_gen*/trees.txt）が"
              "残っているか確認してください。")
        return 1

    trees = {}
    n_skip = 0
    for jf in files:
        with open(jf, encoding="utf-8", errors="replace") as fh:
            t, s = parse_dump(fh.read())
        trees.update(t)
        n_skip += s
    if not trees:
        print("中止: 木が1つも読めませんでした。")
        return 1

    # 前提の検証: 硬い backup で root_w が再現されるか
    n_bad = 0
    max_err = 0.0
    for tree in trees.values():
        nodes = tree["nodes"]
        leaves = leaf_map(nodes)
        v = backup_float(nodes, leaves.__getitem__)
        err = abs(v - tree["root_w"])
        max_err = max(max_err, err)
        if err > 1.0e-6:
            n_bad += 1
    if n_bad > 0:
        print(f"中止: 硬い backup で root_w を再現できない木が {n_bad} 件"
              f"あります（最大誤差 {max_err:.2e}）。"
              "データとツールの前提が食い違っているため、判定できません。")
        return 1

    from check_unregistered2 import LabelOracle, lookup_variants
    oracle = LabelOracle()

    def oracle_label(key):
        _d, _m, normalized, normalized_mirror = lookup_variants(oracle, key)
        return normalized if normalized is not None else normalized_mirror

    # 集計: 全 root / 未解決 root の2通り
    def new_agg():
        return {"n": 0, "hard_ok": 0,
                "tau": {t: {"ok": 0, "drift": 0.0,
                            "soft_only": 0, "hard_only": 0}
                        for t in taus}}

    agg_all = new_agg()
    agg_unres = new_agg()
    n_covered = 0
    for key, tree in trees.items():
        lab = oracle_label(key)
        if lab is None:
            continue
        n_covered += 1
        nodes = tree["nodes"]
        leaves = leaf_map(nodes)
        hard = backup_float(nodes, leaves.__getitem__)
        hard_ok = cat_of_value(hard) == lab
        unresolved = nodes[0][1] == NODE_UNKNOWN
        for agg in (agg_all, agg_unres if unresolved else None):
            if agg is None:
                continue
            agg["n"] += 1
            agg["hard_ok"] += hard_ok
        for t in taus:
            soft = backup_soft(nodes, leaves.__getitem__, t)
            soft_ok = cat_of_value(soft) == lab
            for agg in (agg_all, agg_unres if unresolved else None):
                if agg is None:
                    continue
                a = agg["tau"][t]
                a["ok"] += soft_ok
                a["drift"] += abs(soft - hard)
                if soft_ok and not hard_ok:
                    a["soft_only"] += 1
                elif hard_ok and not soft_ok:
                    a["hard_only"] += 1

    md("# 柔らかいバックアップの診断（soft_backup_diag.py）")
    md()
    md(f"- 対象: {patterns}（{len(files)} ファイル）")
    md(f"- 木の数: {len(trees):,}（SKIP {n_skip}）/ 正解を引けた root: "
       f"{n_covered:,}")
    md(f"- 硬い backup の再現検証: 全木で root_w を再現（最大誤差 "
       f"{max_err:.1e}）")
    md()
    for title, agg in (("## 全 root", agg_all),
                       ("## 未解決 root のみ（主たる対象）", agg_unres)):
        md(title)
        md()
        if agg["n"] == 0:
            md("- 対象なし")
            md()
            continue
        md(f"- 硬い教師の一致: **{pct(agg['hard_ok'], agg['n']):.1f}%**"
           f"（{agg['n']:,} 件）")
        for t in taus:
            a = agg["tau"][t]
            md(f"- 柔らかい τ={t}: 一致 "
               f"**{pct(a['ok'], agg['n']):.1f}%**"
               f"（硬いとの差 {pct(a['ok'], agg['n']) - pct(agg['hard_ok'], agg['n']):+.1f}pt）"
               f" / |柔らかい − 硬い| の平均 {a['drift'] / agg['n']:.4f}"
               f" / 柔らかいのみ正解 {a['soft_only']} 件"
               f" / 硬いのみ正解 {a['hard_only']} 件")
        md()
    md("## 読み方")
    md()
    md("- ある τ で未解決 root の一致が明確に上がる → 柔らかい教師は"
       "学習実験の候補")
    md("- どの τ でも硬いほう以下 → 教師を柔らかくする線は閉じる")
    md("- ドリフトが大きい τ では、値はゲームの値としての意味から"
       "ずれている。一致が上がらない限り採用理由はない")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"結果を {args.out} に書きました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
