# A2 事前診断スクリプト（学習なし）
# ================================
# 問い: A2「短期ターゲット」は s より真値に近いか。学習を回す前に既存レコードだけで判定する。
#
# 方法: 自己対局レコードから対局順の局面列を復元し（audit_order.py と同一の接続判定）、
# 各局面の教師 s_t を、同じ対局の未来の s たちの指数平均に置き換える。
#   s'_t = Σ_k w_k · (−1)^k · s_{t+k},  w_k ∝ λ^k（λ≈0.8、平均ホライズン約5手、対局終端で打ち切って正規化）
# 符号の交互反転は必須: s は手番側視点なので、1手先の s は相手視点であり、
# p_t のプレイヤー視点に直すには (−1)^k を掛ける。r には触らない。
# 計算は後ろからの漸化式で O(n): A_t = s_t − λ·A_{t+1}, W_t = 1 + λ·W_{t+1}, s'_t = A_t / W_t。
#
# 判定: オラクル照合（正規化+鏡像、investigate45.py の C と同一）で
# s と s' のカテゴリ正解率（閾値±0.2）を比べる。全体＋漏れセル（先手劣勢・win・draw）別。
#   - s' が全体で s を上回り、漏れセルでも改善 → A2 は学習に進む価値あり
#   - s' が全体で下回るか、漏れセルで悪化 → A2 は学習前に棄却
#
# 実行（ホスト側 learn/ ディレクトリで、プロジェクトの .venv を使う。
# data/ と ../data/*.dat（オラクルDB）と gamelibs が必要）:
#   cd ~/Documents/cpp_animal_shogi/learn
#   ../.venv/bin/python diagnose_a2.py --group gen_po500
#   ../.venv/bin/python diagnose_a2.py --group gen_po100 --lam 0.8
#
# 出力: 標準出力に要約、diagnose_a2_result.md に Markdown 形式で全結果。
#
# 注意:
#   - 対局列の復元は audit_order.py と同一の前提に立つ（連続エントリが1手で
#     つながる間を同一対局とする。ε-greedy のランダム手も「実際に指された手」
#     として列に含まれるため、その影響は s' にも少し混ざる）。
#   - 終局局面は評価対象から除外する（investigate45.py の C と同一）。
#   - gamelibs の MoveList は __getitem__ に境界チェックが無く、for で回すと
#     範囲外を読み続けて segfault する。必ず len() + インデックスでアクセスすること。

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from check_unregistered import decode_key, self_test
from check_unregistered2 import LabelOracle, lookup_variants

THRESHOLD = 0.2

md_lines = []


def md(line=""):
    md_lines.append(line)
    print(line)


def pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


def classify(score, threshold=THRESHOLD):
    if score > threshold:
        return 1
    if score < -threshold:
        return -1
    return 0


def legal_moves_list(state):
    """MoveList を安全にリスト化する（イテレーションは segfault するのでインデックスで）"""
    ml = state.legal_actions()
    return [ml[i] for i in range(len(ml))]


def reconstruct_games(group, max_positions=60000):
    """data/**/*.json から対局列を復元する。audit_order.py と同一の接続判定。
    返すのは対局（エントリ dict のリスト）のリスト。"""
    from game import from_hash

    files = sorted(glob.glob("data/**/*.json", recursive=True))
    games = []
    current = None
    prev_state = None
    total = 0
    for jf in files:
        if Path(jf).parent.name != group:
            continue
        with open(jf) as fh:
            try:
                entries = json.load(fh)
            except json.JSONDecodeError:
                continue
        prev_state = None  # ファイルをまたいで対局はつながらない
        for e in entries:
            if total >= max_positions:
                break
            key = int(e["p"])
            state = from_hash(key)
            if prev_state is not None:
                child_keys = {prev_state.next(a).hash_key()
                              for a in legal_moves_list(prev_state)}
                if key in child_keys and current is not None:
                    current.append(e)
                else:
                    current = [e]
                    games.append(current)
            else:
                current = [e]
                games.append(current)
            prev_state = state
            total += 1
    return games


def compute_s_prime(game, lam):
    """1対局分の s' を後ろからの漸化式で計算する。game はエントリのリスト。
    返すのは s' のリスト（game と同じ順序・長さ）。"""
    n = len(game)
    s_prime = [0.0] * n
    a_next = 0.0  # A_{t+1}: Σ_{j>=0} λ^j (−1)^j s_{t+1+j}
    w_next = 0.0  # W_{t+1}: Σ_{j>=0} λ^j
    for t in range(n - 1, -1, -1):
        s_t = float(game[t]["s"])
        a_t = s_t - lam * a_next
        w_t = 1.0 + lam * w_next
        s_prime[t] = a_t / w_t
        a_next, w_next = a_t, w_t
    return s_prime


def main():
    parser = argparse.ArgumentParser(description="A2 短期ターゲットの事前診断（学習なし）")
    parser.add_argument("--group", default="gen_po500",
                        help="対象グループ名（例: gen_po500）。data/<group>/*.json を読む")
    parser.add_argument("--lam", type=float, default=0.8,
                        help="指数平均の減衰 λ（平均ホライズン約 1/(1-λ) 手）")
    parser.add_argument("--out", default="diagnose_a2_result.md")
    parser.add_argument("--max-positions", type=int, default=60000,
                        help="復元する局面数の上限（audit_order.py と同一）")
    args = parser.parse_args()

    md("# A2 事前診断（短期ターゲット s' vs 現行教師 s）")
    md()
    md(f"- 対象グループ: `{args.group}` / λ={args.lam}（平均ホライズン約 {1.0 / (1.0 - args.lam):.1f} 手）")
    md()
    if not self_test():
        md("**self-test 失敗のため中止**")
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md_lines) + "\n")
        return

    oracle = LabelOracle()
    games = reconstruct_games(args.group, args.max_positions)
    n_pos = sum(len(g) for g in games)
    md(f"- 復元した対局数: {len(games):,} / 局面数: {n_pos:,}")
    md()

    from game import from_hash

    # ---- s' を構成し、オラクル照合で s と s' を比べる ----
    rows = []  # (label, s_pred, sp_pred, turn_is_white, sente_losing)
    unknown = 0
    skipped_terminal = 0
    for game in games:
        s_prime = compute_s_prime(game, args.lam)
        for e, sp in zip(game, s_prime):
            key = int(e["p"])
            state = from_hash(key)
            if state.is_win() or state.is_lose():
                skipped_terminal += 1
                continue
            _direct, _mirror, normalized, normalized_mirror = lookup_variants(oracle, key)
            label = normalized if normalized is not None else normalized_mirror
            if label is None:
                unknown += 1
                continue
            turn_is_white, _, _ = decode_key(key)
            sente_losing = (label == -1 and not turn_is_white) or (label == 1 and turn_is_white)
            rows.append((label, classify(float(e["s"])), classify(float(sp)),
                         turn_is_white, sente_losing, float(e["s"]), float(sp)))
    md(f"- 終局除外: {skipped_terminal:,} 件 / オラクル未登録: {unknown} 件 / 集計: {len(rows):,} 件")
    md()
    if not rows:
        md("- 集計対象がないため中止（data/ とオラクルDBの場所を確認）")
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md_lines) + "\n")
        return

    flips = sum(1 for r in rows if r[1] != r[2])
    mean_abs_diff = float(np.mean([abs(r[6] - r[5]) for r in rows]))
    md(f"- s' による分類の変化: {flips:,} 件（{pct(flips, len(rows)):.1f}%）/ |s'−s| の平均 {mean_abs_diff:.3f}")
    md()

    def acc_table(sub, title):
        md(f"### {title}（{len(sub):,} 件）")
        md()
        md("| カテゴリ | 件数 | s 正解率 | s' 正解率 | 差分 |")
        md("|---|---|---|---|---|")
        for lv, name in ((1, "win"), (-1, "lose"), (0, "draw")):
            ss = [r for r in sub if r[0] == lv]
            if not ss:
                continue
            a_s = pct(sum(1 for r in ss if r[1] == lv), len(ss))
            a_sp = pct(sum(1 for r in ss if r[2] == lv), len(ss))
            md(f"| {name} | {len(ss):,} | {a_s:.1f}% | {a_sp:.1f}% | {a_sp - a_s:+.1f}pt |")
        all_s = pct(sum(1 for r in sub if r[1] == r[0]), len(sub))
        all_sp = pct(sum(1 for r in sub if r[2] == r[0]), len(sub))
        md(f"| **全体** | {len(sub):,} | **{all_s:.1f}%** | **{all_sp:.1f}%** | **{all_sp - all_s:+.1f}pt** |")
        md()
        return all_s, all_sp

    md("## s vs s' のカテゴリ正解率（閾値±0.2、オラクル照合）")
    md()
    overall_s, overall_sp = acc_table(rows, "全局面")

    md("## 漏れセル別")
    md()
    sl_s, sl_sp = acc_table([r for r in rows if r[4]], "先手劣勢局面")
    acc_table([r for r in rows if not r[4]], "それ以外")

    md("## 手番別")
    md()
    acc_table([r for r in rows if not r[3]], "先手番")
    acc_table([r for r in rows if r[3]], "後手番")

    md("## 判定")
    md()
    if overall_sp > overall_s and sl_sp >= sl_s:
        md(f"- **A2 は学習に進む価値あり**: 全体 {overall_s:.1f}% → {overall_sp:.1f}%（{overall_sp - overall_s:+.1f}pt）、"
           f"先手劣勢セルも {sl_s:.1f}% → {sl_sp:.1f}%（{sl_sp - sl_s:+.1f}pt）と改善。")
    elif overall_sp <= overall_s:
        md(f"- **A2 は学習前に棄却**: 全体 {overall_s:.1f}% → {overall_sp:.1f}%（{overall_sp - overall_s:+.1f}pt）と改善なし。"
           "s' は s より真値に近くないため、学習を回しても期待できない。")
    else:
        md(f"- **判断保留**: 全体は {overall_s:.1f}% → {overall_sp:.1f}%（{overall_sp - overall_s:+.1f}pt）と改善したが、"
           f"先手劣勢セルは {sl_s:.1f}% → {sl_sp:.1f}%（{sl_sp - sl_s:+.1f}pt）。漏れセルの内訳（win/draw 別）を見て決めること。")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"\n結果を {args.out} に書きました")


if __name__ == "__main__":
    main()
