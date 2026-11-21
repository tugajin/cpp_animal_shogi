# 実験4.5 調査スクリプト
# ====================
# 実験4（ans 最高 0.8738 で頭打ち）の原因を切り分けるための一括調査。
# 学習は行わない。既存モデルの評価と自己対局データの監査だけを行う。
#
# 実行（learn/ ディレクトリで、プロジェクトの .venv を使う）:
#   cd ~/Documents/cpp_animal_shogi/learn
#   ../.venv/bin/python investigate45.py                # 全調査（オラクルDB使用）
#   ../.venv/bin/python investigate45.py --skip-oracle  # 軽量版（A+Bのみ、数分）
#
# 出力:
#   - 標準出力に要約
#   - investigate45_result.md に EXPERIMENT.md へ貼れる Markdown 形式で全結果
#
# 調査項目:
#   A. problem.txt (win 5001 / lose 5001) の誤答内訳
#      … 逆判定か未決か、win/lose の非対称、閾値感度
#   B. 自己対局データ監査（data/**/*.json を再帰走査、世代別）
#      … 件数・ユニーク率・重複、世代間の重なり、search_score / game_result の分布と一致率
#   C. 最新世代データでの教師信号の質（オラクル照合、実験3.5と同形式）
#      … model / search_score / game_result の正解率を win/lose/draw 別に比較
#   D. オラクルサンプルでのカテゴリ別精度（実験3.5と同形式、seed=42）
#      … win / lose / draw 各10万局面サンプル
#
# 注意:
#   - C/D は ../data/{win,lose,draw}.dat（プロジェクト直下の data/）を使う。
#     win.dat は約1.6GBあり、読み込みとソートにメモリと時間がかかる。
#   - 既存の analyze_*.py は data/*.json の非再帰 glob で gen_N/ を取りこぼすため、
#     本スクリプトでは再帰 glob に直してある。
#
# 改修履歴:
#   - 2026-11-20: C のオラクル照合を「直接照合のみ」から「正規化+鏡像」
#     （check_unregistered2.lookup_variants）に変更。DBのキーは先手番正規化+
#     鏡像正規化されているため、従来の C は後手番局面がすべて「未登録」として
#     落ち、先手番かつ正規形の局面だけを集計していた（実験4.5の C の数値は
#     その部分集合のものとして読み替えること）。あわせて C に手番別の内訳表を
#     追加。A/B/D は変更なし（A の problem.txt は先手番のみが仕様、D は DB の
#     正規形キーを直接引くため従来どおり実験3.5と比較可能）。

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from check_unregistered import decode_key
from check_unregistered2 import lookup_variants

THRESHOLD = 0.2
ORACLE_DIR = "../data"
WIN_COUNT_FALLBACK = 196_773_087  # ファイルサイズから読めない場合の予備値（既存スクリプトと同値）

md_lines = []


def md(line=""):
    md_lines.append(line)
    print(line)


def classify(score, threshold=THRESHOLD):
    if score > threshold:
        return 1
    if score < -threshold:
        return -1
    return 0


def breakdown(scores, labels, label_value):
    """labels == label_value の局面について 正解/逆判定/未決 を返す"""
    mask = labels == label_value
    n = int(mask.sum())
    if n == 0:
        return n, 0, 0, 0
    preds = np.array([classify(s) for s in scores[mask]])
    correct = int((preds == label_value).sum())
    if label_value == 0:
        reverse = 0
    else:
        reverse = int((preds == -label_value).sum())
    undecided = n - correct - reverse
    return n, correct, reverse, undecided


def pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


# ----------------------------------------------------------------------
# A. problem.txt の誤答内訳
# ----------------------------------------------------------------------

def section_a(model, device, batch_predict):
    md("## A. problem.txt の誤答内訳")
    md()
    problem_path = "../oracle/problem.txt"
    if not os.path.isfile(problem_path):
        md(f"- {problem_path} が見つからないためスキップ")
        md()
        return
    keys, labels = [], []
    with open(problem_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, r = line.split(",")
            keys.append(int(k))
            labels.append(int(r))
    labels = np.array(labels)
    from game import from_hash
    states = [from_hash(k) for k in keys]
    scores = batch_predict(model, states, device)

    ans = float((np.array([classify(s) for s in scores]) == labels).mean())
    md(f"- 局面数: {len(keys)}（win {(labels == 1).sum()} / lose {(labels == -1).sum()}）")
    md(f"- **ans（閾値±0.2）: {ans:.4f}**")
    md()
    md("| 正解ラベル | 局面数 | 正解 | 逆判定 | 未決 |")
    md("|---|---|---|---|---|")
    for lv, name in ((1, "win (+1)"), (-1, "lose (-1)")):
        n, c, r, u = breakdown(scores, labels, lv)
        md(f"| {name} | {n} | {c} ({pct(c, n):.1f}%) | {r} ({pct(r, n):.1f}%) | {u} ({pct(u, n):.1f}%) |")
    md()
    md("閾値感度（閾値を下げたときに ans が大きく上がるなら、頭打ちの一部は測り方の問題）:")
    md()
    md("| 閾値 | ±0.05 | ±0.10 | ±0.15 | ±0.20 | ±0.25 | ±0.30 | ±0.35 | ±0.40 |")
    row = []
    for th in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        preds = np.array([classify(s, th) for s in scores])
        row.append(f"{(preds == labels).mean():.4f}")
    md("| ans | " + " | ".join(row) + " |")
    md()
    # スコア絶対値の分布（縮みの確認）
    abs_scores = np.abs(scores)
    md(f"- スコア: 平均 {scores.mean():.4f} / 標準偏差 {scores.std():.4f} / |score|<0.2 の割合 {pct(int((abs_scores < 0.2).sum()), len(scores)):.1f}%")
    md()


# ----------------------------------------------------------------------
# B. 自己対局データ監査
# ----------------------------------------------------------------------

def load_selfplay_by_group():
    """data/**/*.json を親ディレクトリ名（gen_N など）ごとに読む"""
    groups = {}
    files = sorted(glob.glob("data/**/*.json", recursive=True))
    for jf in files:
        group = Path(jf).parent.name  # gen_12 / data（直下）
        try:
            with open(jf) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            print(f"skip {jf}: {e}")
            continue
        groups.setdefault(group, []).extend(data)
    return groups


def group_sort_key(name):
    # gen_数字 は数字順、それ以外は後ろ
    if name.startswith("gen_"):
        try:
            return (0, int(name[4:]), name)
        except ValueError:
            pass
    return (1, 0, name)


def section_b(groups):
    md("## B. 自己対局データ監査")
    md()
    if not groups:
        md("- data/**/*.json が見つからないためスキップ（learn/ で実行しているか確認）")
        md()
        return None
    md("| 世代 | エントリ数 | ユニーク局面 | 重複率 | |s|<0.2率 | s・r符号一致率 |")
    md("|---|---|---|---|---|---|")
    seen_keys = set()
    total_entries = 0
    latest_group = None
    for name in sorted(groups, key=group_sort_key):
        entries = groups[name]
        keys = [e["p"] for e in entries]
        uniq = set(keys)
        dup_rate = pct(len(keys) - len(uniq), len(keys)) if keys else 0.0
        s_vals = np.array([e["s"] for e in entries], dtype=float)
        undecided_rate = pct(int((np.abs(s_vals) < THRESHOLD).sum()), len(s_vals)) if len(s_vals) else 0.0
        agree, agree_n = 0, 0
        for e in entries:
            r = e["r"]
            if r in (-1, 0, 1):
                agree_n += 1
                if classify(float(e["s"])) == r:
                    agree += 1
        agree_rate = pct(agree, agree_n) if agree_n else 0.0
        md(f"| {name} | {len(entries):,} | {len(uniq):,} | {dup_rate:.1f}% | {undecided_rate:.1f}% | {agree_rate:.1f}%（n={agree_n:,}） |")
        new_keys = uniq - seen_keys
        if total_entries:
            md(f"  - 前世代までとの新規局面: {len(new_keys):,}（この世代のユニークの {pct(len(new_keys), len(uniq)):.1f}%）")
        seen_keys |= uniq
        total_entries += len(entries)
        latest_group = name
    md()
    md(f"- 合計: {total_entries:,} エントリ / 全体ユニーク {len(seen_keys):,} 局面")
    md(f"- 最新世代として C で使うグループ: **{latest_group}**")
    md()
    return latest_group


# ----------------------------------------------------------------------
# オラクル読み込み（二分探索用にソート済み配列で保持する）
# ----------------------------------------------------------------------

class Oracle:
    def __init__(self, base_dir=ORACLE_DIR):
        self.arrays = {}
        for name in ("win", "lose", "draw"):
            path = os.path.join(base_dir, f"{name}.dat")
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            arr = np.fromfile(path, dtype=np.uint64)
            arr.sort()
            self.arrays[name] = arr

    def lookup(self, key):
        k = np.uint64(key)
        for name, value in (("win", 1), ("lose", -1), ("draw", 0)):
            arr = self.arrays[name]
            idx = np.searchsorted(arr, k)
            if idx < len(arr) and arr[idx] == k:
                return value
        return None

    # check_unregistered2.lookup_variants が要求する名前（中身は lookup と同一）
    def label(self, key):
        return self.lookup(key)

    def sample_keys(self, name, n, rng):
        arr = self.arrays[name]
        if len(arr) <= n:
            return arr.tolist()
        idx = rng.choice(len(arr), n, replace=False)
        return arr[idx].tolist()


# ----------------------------------------------------------------------
# C. 最新世代データでの教師信号の質
# ----------------------------------------------------------------------

def accuracy_table(rows):
    """rows: (oracle_label, model_pred, search_pred, result_pred or None) のリスト。
    win/lose/draw 別に model/search/result の正解率を Markdown 表にする"""
    md("| カテゴリ | 件数 | model正解率 | search_score正解率 | game_result正解率 |")
    md("|---|---|---|---|---|")
    for lv, name in ((1, "win"), (-1, "lose"), (0, "draw")):
        sub = [r for r in rows if r[0] == lv]
        if not sub:
            continue
        n = len(sub)
        m = pct(sum(1 for r in sub if r[1] == lv), n)
        s = pct(sum(1 for r in sub if r[2] == lv), n)
        gr = [r for r in sub if r[3] is not None]
        g = pct(sum(1 for r in gr if r[3] == lv), len(gr)) if gr else float("nan")
        g_str = f"{g:.1f}%" if gr else "—"
        md(f"| {name} | {n} | {m:.1f}% | {s:.1f}% | {g_str} |")
    md()


def section_c(model, device, batch_predict, groups, latest_group, oracle, sample_size):
    md("## C. 最新世代データでの教師信号の質（オラクル照合）")
    md()
    if groups is None or latest_group is None:
        md("- 自己対局データがないためスキップ")
        md()
        return
    from game import from_hash
    entries = groups[latest_group]
    rng = np.random.default_rng(123)
    if len(entries) > sample_size:
        idx = rng.choice(len(entries), sample_size, replace=False)
        entries = [entries[i] for i in idx]
        md(f"- 対象: {latest_group} から {sample_size:,} 件をサンプル（seed=123）")
    else:
        md(f"- 対象: {latest_group} の全 {len(entries):,} 件")
    states, metas = [], []
    skipped_terminal = 0
    for e in entries:
        state = from_hash(int(e["p"]))
        if state.pos.is_win() or state.pos.is_lose():
            skipped_terminal += 1
            continue
        states.append(state)
        metas.append(e)
    scores = batch_predict(model, states, device)
    rows = []
    unknown = 0
    for score, e in zip(scores, metas):
        key = int(e["p"])
        # 正規化+鏡像で照合する（DBキーは先手番正規化+鏡像正規化されている）。
        # 返るラベルは手番側視点で、model / s / r の視点と一致する
        _direct, _mirror, normalized, normalized_mirror = lookup_variants(oracle, key)
        label = normalized if normalized is not None else normalized_mirror
        if label is None:
            unknown += 1
            continue
        turn_is_white, _, _ = decode_key(key)
        r = e["r"]
        rows.append((
            label,
            classify(float(score)),
            classify(float(e["s"])),
            int(r) if r in (-1, 0, 1) else None,
            turn_is_white,
        ))
    md(f"- 終局除外: {skipped_terminal} 件 / オラクル未登録: {unknown} 件 / 集計: {len(rows):,} 件")
    md()
    md("照合は正規化+鏡像（2026-11-20改修）。実験4.5当時の C は直接照合のみで、先手番かつ正規形の局面だけが集計対象だったため、数値はそのまま比較できない。実験3.5の同表（当時: win は model 85.0% / search 94.5% / result 86.6%、lose は 56.1% / 70.7% / 73.5%、draw は 32.0% / 26.1% / 6.2%）も直接照合時代の値として参考に留めること。")
    md()
    md("### 合算")
    md()
    accuracy_table(rows)
    for tw, tname in ((False, "先手番"), (True, "後手番")):
        sub = [r for r in rows if r[4] == tw]
        if not sub:
            continue
        md(f"### {tname}（{len(sub):,} 件）")
        md()
        accuracy_table(sub)


# ----------------------------------------------------------------------
# D. オラクルサンプルでのカテゴリ別精度
# ----------------------------------------------------------------------

def section_d(model, device, batch_predict, oracle, sample_size):
    md("## D. オラクルサンプルでのカテゴリ別精度")
    md()
    from game import from_hash
    md("| カテゴリ | 有効局面数 | 正解（±0.2） | 逆判定 | 未決 |")
    md("|---|---|---|---|---|")
    for name, lv in (("win", 1), ("lose", -1), ("draw", 0)):
        rng = np.random.default_rng(42)  # カテゴリごとに再シード（analyze_oracle.py / 実験3.5と同一サンプルにするため）
        keys = oracle.sample_keys(name, sample_size, rng)
        states = []
        for k in keys:
            state = from_hash(int(k))
            if state.pos.is_win() or state.pos.is_lose():
                continue
            states.append(state)
        scores = batch_predict(model, states, device)
        labels = np.full(len(states), lv)
        n, c, r, u = breakdown(scores, labels, lv)
        if lv == 0:
            md(f"| {name} | {n:,} | {c:,} ({pct(c, n):.1f}%) | — | {u:,} ({pct(u, n):.1f}%) |")
        else:
            md(f"| {name} | {n:,} | {c:,} ({pct(c, n):.1f}%) | {r:,} ({pct(r, n):.1f}%) | {u:,} ({pct(u, n):.1f}%) |")
    md()
    md("実験3.5の同表（win 84.6% / lose 88.0% / draw 18.6%）と比較すること。")
    md()


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="実験4.5 調査スクリプト")
    parser.add_argument("--model", default="./model/best_single.h5")
    parser.add_argument("--out", default="investigate45_result.md")
    parser.add_argument("--skip-oracle", action="store_true", help="C/D を省略（A+Bのみ）")
    parser.add_argument("--sample", type=int, default=20000, help="C のサンプル上限")
    parser.add_argument("--group", default=None, help="C の対象グループ名（例: gen_16）。省略時は推定した最新グループ")
    parser.add_argument("--oracle-sample", type=int, default=100000, help="D のカテゴリ別サンプル数")
    args = parser.parse_args()

    md("# 実験4.5 調査結果（investigate45.py）")
    md()
    md(f"- モデル: `{args.model}`")
    md()

    groups = load_selfplay_by_group()

    model = None
    device = None
    batch_predict = None
    if os.path.isfile(args.model):
        import torch
        from single_network import SingleNet, DN_INPUT_SHAPE

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SingleNet()
        model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        md(f"- デバイス: {device}")
        md()

        def batch_predict(model_, states, device_, batch_size=4096):
            file, rank, channel = DN_INPUT_SHAPE
            out = []
            for i in range(0, len(states), batch_size):
                batch = states[i:i + batch_size]
                feats = np.array([np.array(s.feature()).reshape(channel, file, rank) for s in batch])
                x = torch.from_numpy(feats).to(device_).float()
                with torch.no_grad():
                    out.append(model_(x).squeeze(-1).cpu().numpy())
            return np.concatenate(out) if out else np.array([])

        section_a(model, device, batch_predict)
    else:
        md(f"- モデル {args.model} が見つからないため A/C/D をスキップ")
        md()

    latest_group = section_b(groups)
    if args.group is not None:
        if args.group in groups:
            latest_group = args.group
            md(f"- C の対象グループを `--group` 指定により **{latest_group}** に変更")
            md()
        else:
            md(f"- `--group {args.group}` がデータ内にないため、推定した最新グループ {latest_group} を使う")
            md()

    if not args.skip_oracle and model is not None:
        try:
            oracle = Oracle()
        except FileNotFoundError as e:
            md(f"- オラクルDBが見つからないため C/D をスキップ: {e}")
            md()
            oracle = None
        if oracle is not None:
            section_c(model, device, batch_predict, groups, latest_group, oracle, args.sample)
            section_d(model, device, batch_predict, oracle, args.oracle_sample)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\n結果を {args.out} に書きました")


if __name__ == "__main__":
    main()
