# 証明ラベルの同定可能性チェック（proven_label_check.py）
# ========================================================
# 新手法の premise を検証する診断スクリプト。学習・生成は一切しない。
#
# 背景:
#   探索で証明された局面の値（resolved）は score_win(ply) = 0.9999 - ply/2000
#   という格子上の値になる（ai/ubfm.hpp）。一方、自己対局レコード {p, s, r} は
#   「その s が証明値か推定値か」を記録していない（VERIFICATION_PLAN 以来の既知の欠落）。
#   もし s の値そのものから証明ラベルを同定できるなら、ラベルの「確からしさ」で
#   層別する学習（証明ラベルは r 項を切って高重み、推定ラベルは現行通り）が可能になる。
#
# 検証すること:
#   Q1. s の分布に格子点 0.9999 - k/2000（k は整数）へのスパイクがあるか
#       （半格子ずらした対照バンドとの比較で偽陽性を差し引く）
#   Q2. 格子点に一致したレコードは、オラクル照合で本当にその勝敗か（同定の精度）
#   Q3. 証明ラベル候補は全体の何割で、どういう局面に付いているか（終端手数の分布）
#   Q4. モデルは証明ラベル候補を既に再現しているか（|v - s| を推定ラベルと比較）
#
# 実行（learn/ ディレクトリで）:
#   ../.venv/bin/python proven_label_check.py
# 出力: proven_label_result.md
#
# 注意:
#   - オラクルは評価専用（同定の正しさの検証のみ）。学習には使わない
#   - ラベル照合は正規化+鏡像（check_unregistered2 の方式）

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from check_unregistered import decode_key, self_test
from check_unregistered2 import LabelOracle, lookup_variants

BASE = 0.9999
STEP = 1.0 / 2000.0
# 許容誤差: 証明値は float の丸め誤差（~1e-7）の範囲で格子点に正確に乗る。
# 格子間隔は 5e-4 しかないため、許容誤差は間隔より十分小さく取る必要がある
# （1e-4 では帯域の 40% が「一致」してしまい、推定値と区別できない）。
TOL = 1.0e-5

md_lines = []


def md(line=""):
    md_lines.append(line)
    print(line)


def pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


def lattice_k(s_abs):
    """|s| が格子点 BASE - k*STEP に一致すれば k を、しなければ None を返す"""
    d = (BASE - s_abs) / STEP
    k = int(round(d))
    if k < 0:
        return None
    if abs(BASE - k * STEP - s_abs) < TOL:
        return k
    return None


def control_k(s_abs):
    """対照バンド: 半格子ずらした点 BASE - (k+0.5)*STEP に一致すれば k を返す。
    推定値が一様に分布していれば、格子バンドと対照バンドの該当数は同程度になるはずで、
    差分が「本物の証明ラベルのスパイク」の見積もりになる"""
    d = (BASE - s_abs) / STEP - 0.5
    k = int(round(d))
    if k < 0:
        return None
    if abs(BASE - (k + 0.5) * STEP - s_abs) < TOL:
        return k
    return None


def oracle_label_sente(oracle, key):
    """正規化照合でラベルを引き、先手視点 (+1 先手勝ち / 0 引分 / -1 先手負け) に直す"""
    _d, _m, normalized, normalized_mirror = lookup_variants(oracle, key)
    label = normalized if normalized is not None else normalized_mirror
    if label is None:
        return None
    turn_is_white, _, _ = decode_key(key)
    return -label if turn_is_white else label


def self_test_local():
    # 格子上の値は同定でき、格子外の値は同定できないこと
    assert lattice_k(0.9999) == 0
    assert lattice_k(0.9999 - 37 * STEP) == 37
    assert lattice_k(0.9999 - 37 * STEP + 5e-7) == 37  # float 丸め程度のずれは同定できる
    assert lattice_k(0.5) is None  # 最も近い格子点 0.4999 から 1e-4 離れている
    assert lattice_k(0.9999 - 37 * STEP + 0.0003) is None  # 許容誤差の外
    assert control_k(0.9999 - 37 * STEP) is None
    assert control_k(0.9999 - (37 + 0.5) * STEP) == 37
    return True


def main():
    parser = argparse.ArgumentParser(description="証明ラベルの同定可能性チェック")
    parser.add_argument("--group", default="data")
    parser.add_argument("--model", default="./model/best_single.h5")
    parser.add_argument("--out", default="proven_label_result.md")
    parser.add_argument("--model-sample", type=int, default=10000, help="Q4 で推論する片側の局面数上限")
    args = parser.parse_args()

    md("# 証明ラベルの同定可能性チェック（proven_label_check.py）")
    md()
    if not (self_test() and self_test_local()):
        md("**self-test 失敗のため中止**")
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md_lines) + "\n")
        return
    md("- self-test: OK")

    # ---- レコード読み込み ----
    files = sorted(glob.glob("data/**/*.json", recursive=True))
    entries = []
    used = 0
    for jf in files:
        if Path(jf).parent.name != args.group:
            continue
        used += 1
        with open(jf) as fh:
            try:
                entries.extend(json.load(fh))
            except json.JSONDecodeError:
                continue
    md(f"- 対象グループ: `{args.group}`（{used} ファイル、{len(entries):,} レコード）")
    md()

    # ---- Q1. 格子スパイク ----
    cand = []       # (key, s, k) 証明ラベル候補
    ctrl = 0        # 対照バンド該当数
    by_key_s = defaultdict(list)
    for e in entries:
        s = float(e["s"])
        by_key_s[int(e["p"])].append(s)
        if abs(s) > 0.5:  # 格子は |s| が大きい側にしか現れない（k が大きいと推定値と分離不能）
            k = lattice_k(abs(s))
            if k is not None:
                cand.append((int(e["p"]), s, k))
            if control_k(abs(s)) is not None:
                ctrl += 1
    md("## Q1. 格子スパイクの有無")
    md()
    md(f"- |s|>0.5 のレコードのうち格子点に一致（証明ラベル候補）: {len(cand):,} 件")
    md(f"- 対照バンド（半格子ずらし）に一致: {ctrl:,} 件 → スパイクの超過分は約 {len(cand) - ctrl:,} 件")
    md(f"- 候補の割合: 全レコードの {pct(len(cand), len(entries)):.1f}% / |s|>0.5 のレコードの {pct(len(cand), max(1, sum(1 for e in entries if abs(float(e['s'])) > 0.5))):.1f}%")
    md()

    # ---- Q3. 候補の分布（終端手数 k の分布） ----
    md("## Q3. 証明ラベル候補の分布")
    md()
    ks = Counter(k for _, _, k in cand)
    if ks:
        k_arr = np.array([k for _, _, k in cand], dtype=float)
        md(f"- 含意される終端手数 k: 平均 {k_arr.mean():.1f} / 中央値 {np.median(k_arr):.0f} / 最大 {int(k_arr.max())}")
        wins = sum(1 for _, s, _ in cand if s > 0)
        md(f"- 勝ち証明 {wins:,} 件 / 負け証明 {len(cand) - wins:,} 件")
        uniq = len({k_ for k_, _, _ in cand})
        md(f"- 候補のユニーク局面数: {uniq:,}")
    md()

    # ---- Q2. オラクルでの同定精度 ----
    md("## Q2. 同定の精度（オラクル照合、評価専用）")
    md()
    oracle = LabelOracle()
    uniq_cand = {}
    for key, s, k in cand:
        uniq_cand.setdefault(key, (s, k))
    agree = disagree = unknown = 0
    disagree_examples = []
    disagree_ks = []
    for key, (s, k) in uniq_cand.items():
        lab_sente = oracle_label_sente(oracle, key)
        if lab_sente is None:
            unknown += 1
            continue
        turn_is_white, _, _ = decode_key(key)
        lab_turn = -lab_sente if turn_is_white else lab_sente  # 手番側視点に直す
        expect = 1 if s > 0 else -1
        if lab_turn == expect:
            agree += 1
        else:
            disagree += 1
            disagree_ks.append(k)
            if len(disagree_examples) < 5:
                disagree_examples.append((key, s, lab_turn))
    tot = agree + disagree
    md(f"- 照合できた候補: {tot:,} 件（未登録 {unknown:,} 件）")
    md(f"- **オラクルと一致: {agree:,} 件（{pct(agree, tot):.2f}%）** / 不一致: {disagree:,} 件")
    if disagree_examples:
        md(f"- 不一致の例（key, s, オラクルの手番側ラベル）: {disagree_examples}")
    md()
    md("読み方: 一致率が 99% 以上なら同定は実用になる。不一致が多い場合は、格子一致が偶然か、"
       "証明値の伝播方法の想定（バックアップで値がそのまま伝わる）が間違っている。")
    md()

    # ---- Q5. 証明レコードでの r 汚染（r 項を切る利得の大きさ） ----
    md("## Q5. 証明レコードでの r 汚染")
    md()
    n_r = n_sign_dis = n_r_draw = 0
    dev_all, dev_dis = [], []
    for e in entries:
        s = float(e["s"])
        if abs(s) <= 0.5 or lattice_k(abs(s)) is None:
            continue
        r = float(e.get("r", -99))
        if r <= -90:  # r=-99 は未記録
            continue
        n_r += 1
        dev = abs((s + r) / 2.0 - s)  # 混合教師 (s+r)/2 が証明値 s からどれだけずれるか
        dev_all.append(dev)
        if r == 0:
            n_r_draw += 1
        elif (r > 0) != (s > 0):
            n_sign_dis += 1
            dev_dis.append(dev)
    if n_r:
        md(f"- r が記録されている証明候補: {n_r:,} 件")
        md(f"- r と s が符号不一致: {n_sign_dis:,} 件（{pct(n_sign_dis, n_r):.1f}%） / r=0（引き分け）: {n_r_draw:,} 件（{pct(n_r_draw, n_r):.1f}%）")
        md(f"- 混合教師の証明値からのずれ |(s+r)/2 − s|: 全体平均 {float(np.mean(dev_all)):.4f} / 符号不一致のみ平均 {float(np.mean(dev_dis)) if dev_dis else 0.0:.4f}")
        md()
        md("読み方: 証明局面では s が正解なので、混合教師のずれはそのまま r 項が注入する誤差。"
           "符号不一致率が高ければ「証明レコードでは r 項を切る」arm の利得が大きい。")
    else:
        md("- r が記録されている証明候補が無いためスキップ")
    md()

    # ---- Q6. k=200 スパイク（NN 出力クランプ ±0.8999 の混入チェック） ----
    md("## Q6. k 分布のスパイク（クランプ混入チェック）")
    md()
    near200 = {kk: ks.get(kk, 0) for kk in range(197, 204)}
    md(f"- k=197〜203 の候補数: {near200}")
    topk = ks.most_common(3)
    md(f"- 候補数が多い k: {topk}")
    if disagree_ks:
        dk = Counter(disagree_ks)
        md(f"- Q2 不一致の k 分布（上位）: {dk.most_common(5)}")
    md()
    md("読み方: predict のクランプ（|出力|≥1 → ±0.8999）は格子点 k=200 と正確に一致する。"
       "k=200 にスパイクがあり不一致も k=200 に集中するなら、k=200 の候補は証明ではなく"
       "飽和した推定値の混入であり、同定からは k=200 を除外すべき（po=100 では深さ200の証明木は作れない）。")
    md()

    # ---- Q4. モデルは証明ラベルを再現しているか ----
    md("## Q4. モデルの再現度（証明ラベル候補 vs 推定ラベル）")
    md()
    import os
    if not os.path.isfile(args.model):
        md(f"- モデル {args.model} が見つからないためスキップ")
        md()
    else:
        import torch
        from game import from_hash
        from single_network import SingleNet, DN_INPUT_SHAPE

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SingleNet()
        model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        file, rank, channel = DN_INPUT_SHAPE

        def batch_predict(states):
            out = []
            for i in range(0, len(states), 4096):
                batch = states[i:i + 4096]
                feats = np.array([np.array(st.feature()).reshape(channel, file, rank) for st in batch])
                x = torch.from_numpy(feats).to(device).float()
                with torch.no_grad():
                    out.append(model(x).squeeze(-1).cpu().numpy())
            return np.concatenate(out) if out else np.array([])

        rng = np.random.default_rng(123)
        cand_keys = list(uniq_cand.keys())
        if len(cand_keys) > args.model_sample:
            idx = rng.choice(len(cand_keys), args.model_sample, replace=False)
            cand_keys = [cand_keys[i] for i in idx]
        cand_set = set(uniq_cand.keys())
        other_keys = [k for k in by_key_s if k not in cand_set and abs(float(np.mean(by_key_s[k]))) > 0.5]
        if len(other_keys) > args.model_sample:
            idx = rng.choice(len(other_keys), args.model_sample, replace=False)
            other_keys = [other_keys[i] for i in idx]

        for name, keys_ in (("証明ラベル候補", cand_keys), ("推定ラベル（|s|>0.5）", other_keys)):
            if not keys_:
                continue
            states = [from_hash(k) for k in keys_]
            vals = batch_predict(states)
            s_mean = np.array([float(np.mean(by_key_s[k])) for k in keys_])
            err = np.abs(vals - s_mean)
            sign_ok = np.mean(np.sign(vals) == np.sign(s_mean))
            md(f"- {name}: {len(keys_):,} 局面 / |v − s| 平均 {err.mean():.4f} / 中央値 {np.median(err):.4f} / 符号一致 {100.0 * sign_ok:.1f}%")
        md()
        md("読み方: 証明ラベル候補で |v − s| が大きいなら、正確なラベルが付いている局面ですら"
           "モデルが再現できておらず、確信度層別（証明ラベルへの高重み付け）に意味がある。"
           "逆にほぼ 0 なら、証明ラベルは既に学習され切っていて層別の利得は小さい。")
        md()

    md("## 判定")
    md()
    md("- 候補が全レコードの数% 以上あり、Q2 の一致率が 99% 以上なら「証明ラベルの同定」は成立し、"
       "確信度層別学習（証明レコードでは r 項を切り、s を高重みにする）の arm 化に進める")
    md("- 候補が僅少、または一致率が低い場合は、この手法は不成立として下げる")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"\n結果を {args.out} に書きました")


if __name__ == "__main__":
    main()
