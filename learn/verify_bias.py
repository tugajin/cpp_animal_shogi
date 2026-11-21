# 先手楽観バイアスの検証（verify_bias.py）
# ========================================
# 仮説: モデルと探索スコアの評価は「先手に楽観的」に一方向へ歪んでおり、
# そのために先手が負けている局面でのみ精度が落ちる（実験4.5再計算の4セル表）。
#
# 検証の設計（天井・床効果に騙されないよう3つの指標を使う）:
#   T1. 真の引き分け局面（オラクルラベル 0）での平均値:
#       対称な評価なら先手視点の値は平均 0 のはず。系統的に + なら先手楽観
#   T2. 対称性指標: mean(v | 先手勝ち) + mean(v | 先手負け):
#       対称なら 0。+ なら全体が先手側に寄っている
#   T3. 逆転率: 先手負け局面で v > +0.2 と言う割合 vs 先手勝ち局面で v < -0.2 と言う割合
#
# 値は探索スコア s とモデル出力の両方で測る。全て先手視点に換算して比較する。
# s とモデル出力の視点規約（手番視点か先手固定か）は、ラベルとの符号一致率で
# 実測してから換算する（規約を仮定しない）。
#
# ラベルは正規化照合（先手: 直接+鏡像 / 後手: 回転正規化+鏡像）で付ける。
# サンプルは実験4.5-C と同一（data グループ、seed=123、20,000件、終局除外）。
#
# 実行（learn/ ディレクトリで）:
#   ../.venv/bin/python verify_bias.py
# 出力: verify_bias_result.md

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from check_unregistered import decode_key, self_test
from check_unregistered2 import LabelOracle, lookup_variants

md_lines = []


def md(line=""):
    md_lines.append(line)
    print(line)


def pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


def main():
    parser = argparse.ArgumentParser(description="先手楽観バイアスの検証")
    parser.add_argument("--model", default="./model/best_single.h5")
    parser.add_argument("--out", default="verify_bias_result.md")
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--group", default="data")
    args = parser.parse_args()

    md("# 先手楽観バイアスの検証（verify_bias.py）")
    md()
    if not self_test():
        md("**self-test 失敗のため中止**")
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md_lines) + "\n")
        return

    oracle = LabelOracle()

    files = sorted(glob.glob("data/**/*.json", recursive=True))
    entries = []
    for jf in files:
        if Path(jf).parent.name == args.group:
            with open(jf) as fh:
                try:
                    entries.extend(json.load(fh))
                except json.JSONDecodeError:
                    continue
    rng = np.random.default_rng(123)
    if len(entries) > args.sample:
        idx = rng.choice(len(entries), args.sample, replace=False)
        entries = [entries[i] for i in idx]

    from game import from_hash

    keys, s_vals, turn_white, labels_turn = [], [], [], []
    states = []
    for e in entries:
        key = int(e["p"])
        state = from_hash(key)
        if state.pos.is_win() or state.pos.is_lose():
            continue
        direct, mirror, normalized, normalized_mirror = lookup_variants(oracle, key)
        label = normalized if normalized is not None else normalized_mirror
        if label is None:
            continue
        tw, _, _ = decode_key(key)
        keys.append(key)
        s_vals.append(float(e["s"]))
        turn_white.append(tw)
        labels_turn.append(label)
        states.append(state)

    n = len(keys)
    s_vals = np.array(s_vals)
    turn_white = np.array(turn_white, dtype=bool)
    labels_turn = np.array(labels_turn, dtype=int)
    # 先手視点ラベル: 先手番なら手番ラベルそのまま、後手番なら反転
    labels_sente = np.where(turn_white, -labels_turn, labels_turn)
    md(f"対象: {n:,} 件（ラベル付与成功分）")
    md()

    # ---- モデル出力 ----
    model_vals = None
    if os.path.isfile(args.model):
        import torch
        from single_network import SingleNet, DN_INPUT_SHAPE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SingleNet()
        model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        file, rank, channel = DN_INPUT_SHAPE
        out = []
        for i in range(0, len(states), 4096):
            batch = states[i:i + 4096]
            feats = np.array([np.array(s.feature()).reshape(channel, file, rank) for s in batch])
            x = torch.from_numpy(feats).to(device).float()
            with torch.no_grad():
                out.append(model(x).squeeze(-1).cpu().numpy())
        model_vals = np.concatenate(out)
        md(f"- モデル: `{args.model}`（{device}）")
        md()

    # ---- 視点規約の実測と先手視点への換算 ----
    def to_sente(values, name):
        """符号一致率が高い方の規約を採用し、先手視点に換算して返す"""
        decided = labels_turn != 0
        pred_turn = np.sign(values[decided])
        acc_turn = np.mean(pred_turn == labels_turn[decided])
        acc_sente = np.mean(pred_turn == labels_sente[decided])
        if acc_turn >= acc_sente:
            md(f"- {name}: 手番視点の規約と判定（一致率 {acc_turn:.3f} vs 先手固定 {acc_sente:.3f}）")
            return np.where(turn_white, -values, values)
        md(f"- {name}: 先手固定視点の規約と判定（一致率 {acc_sente:.3f} vs 手番 {acc_turn:.3f}）")
        return values.copy()

    series = {}
    series["探索スコア s"] = to_sente(s_vals, "探索スコア s")
    if model_vals is not None:
        series["モデル出力"] = to_sente(model_vals, "モデル出力")
    md()

    # ---- T1〜T3 ----
    for name, v in series.items():
        md(f"## {name}")
        md()
        draw = labels_sente == 0
        win = labels_sente == 1
        lose = labels_sente == -1
        md(f"- T1 真の引き分け局面での平均値: **{v[draw].mean():+.4f}**（n={draw.sum():,}、対称なら 0）")
        md(f"- T2 対称性指標 mean(v|先手勝ち) + mean(v|先手負け): **{v[win].mean() + v[lose].mean():+.4f}**"
           f"（内訳: 勝ち {v[win].mean():+.4f} / 負け {v[lose].mean():+.4f}、対称なら合計 0）")
        cross_lose = pct(np.sum(v[lose] > 0.2), lose.sum())
        cross_win = pct(np.sum(v[win] < -0.2), win.sum())
        md(f"- T3 逆転率: 先手負け局面で +0.2 超え **{cross_lose:.1f}%** vs 先手勝ち局面で -0.2 未満 **{cross_win:.1f}%**")
        md()
        md("手番別の T1（引き分け局面の平均値）:")
        md()
        md("| 手番 | 件数 | 平均値 |")
        md("|---|---|---|")
        for tw, tname in ((False, "先手番"), (True, "後手番")):
            m = draw & (turn_white == tw)
            if m.sum():
                md(f"| {tname} | {m.sum():,} | {v[m].mean():+.4f} |")
        md()

    md("## 判定の目安")
    md()
    md("- T1 が +0.05 を超えてプラス、T2 もプラス、T3 で先手負け側の逆転率が勝ち側より明確に高い → 先手楽観バイアスを支持")
    md("- T1≈0 かつ T2≈0 で、精度差だけが残る → 歪みではなく「先手劣勢局面そのものの難しさ」（読みの深さ等）が原因という解釈に切り替える")
    md("- s とモデルで傾向が違う場合（例: モデルは対称だが s が歪む）は、歪みの発生箇所が探索側にあることになる")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"\n結果を {args.out} に書きました")


if __name__ == "__main__":
    main()
