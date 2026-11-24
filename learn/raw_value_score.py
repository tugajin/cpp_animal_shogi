# 素のモデル値のスコア付け（raw_value_score.py）
# ==============================================
# 問い: 同じ局面・同じモデルで、探索なしの素の値と探索ありの root 値では、
# oracle に対するカテゴリ一致率はどのくらい変わるのか。
# probe の P（手の選択）は探索あり/なしの比較が既に測れているが、root 値
# （R）の方は局面セットが違うため厳密な比較がまだ無い（ans は problem.txt、
# R は probe 局面）。このスクリプトは probe と同じ局面リストに対して
# ONNX モデルの素の値を直接推論し、同じ ±0.2 境界で分類して比べる。
#
# 手順（learn/ ディレクトリで、プロジェクト .venv を使用）:
#   1) probe と同じ局面リストを用意（probe_score.py --make-list で生成済み
#      の probe_keys_arm0.txt / probe_keys_arm1.txt をそのまま使う）。
#   2) 素の値を測る（モデルごとに ONNX を直接推論。探索は回さない）:
#        ../.venv/bin/python raw_value_score.py --keys probe_keys_arm0.txt \
#            --model arm0=model/arm0/best_single.onnx \
#            --model arm2=model/arm2/best_single.onnx \
#            --probe arm0=probe_arm0.txt --probe arm2=probe_arm2.txt
#      --probe を省くと素の値だけを測る。
#   3) 出力: raw_value_result.md（--out で変更可）
#
# 指標（probe_score.py と同じ定義・同じ真理）:
#   R_raw. 素の値のカテゴリ一致率（±0.2 境界。ans と同じ分類）。
#   paired. 同じ局面での素の値 vs 探索 root 値の正誤の食い違い
#      （探索のみ正解 / 素のみ正解）。探索が較正をどれだけ補正するかを見る。
#   |raw - root_w|. 素の値と探索 root 値の平均絶対差（全体・カテゴリ別）。
#
# 前提・注意:
#   - 値の符号規約は evaluate_problem() と同じ（手番視点の値を ±0.2 で
#     分類して oracle カテゴリと照合）。probe の R も同じ照合なので、
#     R_raw と R は同じ局面・同じ境界での比較になる。
#   - ONNX の入出力契約は export_onnx.py と同じ: 入力 "input" float32
#     (N, 42, 3, 4)、出力 "value" float32 (N, 1)。特徴量は
#     state.feature() を evaluate_network.predict() と同じ reshape で渡す。
#   - 推論は CPU プロバイダ（約2,200局面の一括推論で十分速い。GPU が
#     必要なら providers を ["CUDAExecutionProvider"] に変える）。
#   - 真理の構築は probe_score.py の score() と同じ（全子局面の結果が
#     分かる局面だけを対象にする）。千日手は probe と同じく対象外。

import argparse
from pathlib import Path

import numpy as np

from check_unregistered import decode_key, self_test
from audit_order import legal_moves_list, child_outcome, pct
from probe_score import cat_of, load_probe

VERSION = "2026-11-23 v1（raw: 素のモデル値の oracle 照合）"

md_lines = []


def md(line=""):
    md_lines.append(line)
    print(line)


def build_truth(keys_path):
    """probe_score.score() と同じ真理: 局面ごとの最良カテゴリ等。"""
    from check_unregistered2 import LabelOracle
    from game import from_hash

    keys = []
    with open(keys_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                keys.append(int(line))
    md(f"- 局面リスト: {keys_path}（{len(keys):,} 局面）")

    oracle = LabelOracle()
    truth = {}
    for key in keys:
        state = from_hash(key)
        children = [state.next(a) for a in legal_moves_list(state)]
        outcomes = [child_outcome(oracle, c) for c in children]
        if any(o is None for o in outcomes):
            continue
        best = int(max(outcomes))
        tw, _, _ = decode_key(key)
        truth[key] = {
            "best": best,
            "sente_losing": (best == -1 and not tw) or (best == 1 and tw),
        }
    md(f"- 真理を用意できた局面: {len(truth):,}")
    return truth


def raw_values(model_path, keys):
    """ONNX モデルの素の値を一括推論する。{key: float} を返す。"""
    import onnxruntime as ort
    from game import from_hash

    feats = []
    for key in keys:
        state = from_hash(key)
        feats.append(np.array(state.feature(), dtype=np.float32)
                     .reshape(1, 42, 3, 4))
    batch = np.concatenate(feats, axis=0)
    sess = ort.InferenceSession(model_path,
                                providers=["CPUExecutionProvider"])
    out = sess.run(["value"], {"input": batch})[0]
    return {key: float(v[0]) for key, v in zip(keys, out)}


def summarize_raw(label, rows, title):
    if not rows:
        return
    ok = sum(1 for r in rows if r["raw_cat"] == r["best"])
    md(f"  - {title}（{len(rows):,}）: 一致 "
       f"**{pct(ok, len(rows)):.1f}%**")


def score(args):
    truth = build_truth(args.keys)
    keys = list(truth.keys())

    raw_per_label = {}
    for spec in args.model:
        label, _, path = spec.partition("=")
        if not label or not path:
            print(f"--model は label=パスの形で指定してください: {spec}")
            return
        vals = raw_values(path, keys)
        rows = {}
        for key, t in truth.items():
            if key not in vals:
                continue
            rows[key] = {
                "best": t["best"],
                "sente_losing": t["sente_losing"],
                "raw": vals[key],
                "raw_cat": cat_of(vals[key]),
            }
        raw_per_label[label] = rows
        md(f"- {label}: 素の値 {len(vals):,} 件 <- {path}")

    probe_per_label = {}
    for spec in args.probe:
        label, _, path = spec.partition("=")
        res, skip = load_probe(path)
        probe_per_label[label] = res
        md(f"- {label}: probe 結果 {len(res):,} 件 / SKIP {len(skip):,} 件")

    md()
    for label, rows in raw_per_label.items():
        rlist = list(rows.values())
        ok = sum(1 for r in rlist if r["raw_cat"] == r["best"])
        md(f"## {label}")
        md()
        md(f"- R_raw. 素の値のカテゴリ一致: **{pct(ok, len(rlist)):.1f}%**"
           f"（{len(rlist):,} 局面）")
        summarize_raw(label, rlist, "全体")
        summarize_raw(label, [r for r in rlist if r["best"] == 1],
                      "勝ち局面")
        summarize_raw(label, [r for r in rlist if r["best"] == 0],
                      "引分局面（参考）")
        summarize_raw(label, [r for r in rlist if r["sente_losing"]],
                      "先手劣勢局面")

        if label in probe_per_label:
            res = probe_per_label[label]
            common = [k for k in rows if k in res]
            both_ok = sum(1 for k in common
                          if rows[k]["raw_cat"] == rows[k]["best"]
                          and cat_of(res[k][0]) == rows[k]["best"])
            probe_only = sum(1 for k in common
                             if rows[k]["raw_cat"] != rows[k]["best"]
                             and cat_of(res[k][0]) == rows[k]["best"])
            raw_only = sum(1 for k in common
                           if rows[k]["raw_cat"] == rows[k]["best"]
                           and cat_of(res[k][0]) != rows[k]["best"])
            diffs = [abs(rows[k]["raw"] - res[k][0]) for k in common]
            md(f"- paired（共通 {len(common):,} 局面）: 両方正解 {both_ok} 件 / "
               f"探索のみ正解 {probe_only} 件 / 素のみ正解 {raw_only} 件")
            md(f"- |素の値 - 探索 root 値| 平均: "
               f"{float(np.mean(diffs)):.3f}"
               f"（中央値 {float(np.median(diffs)):.3f}）")


def main():
    print(f"raw_value_score.py {VERSION}")
    parser = argparse.ArgumentParser(description="素のモデル値のスコア付け")
    parser.add_argument("--keys", default="probe_keys.txt",
                        help="局面リスト（probe_keys_*.txt）")
    parser.add_argument("--model", action="append", default=[],
                        help="label=ONNXパス（複数可）")
    parser.add_argument("--probe", action="append", default=[],
                        help="比較用: label=probe出力ファイル（複数可）")
    parser.add_argument("--out", default="raw_value_result.md")
    args = parser.parse_args()

    if not self_test():
        print("**self-test 失敗のため中止**")
        return
    if not args.model:
        print("--model label=ONNXパス を1つ以上指定してください")
        return
    md("# 素のモデル値のスコア（探索なしの oracle 照合）")
    md()
    score(args)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"\n結果を {args.out} に書きました")


if __name__ == "__main__":
    main()
