# 探索予算の応答曲線（budget_curve.py）
# ======================================
# 問い: 同じモデル・同じ局面で、探索予算 po を 0 / 10 / 100 / 500 と変えた
# とき、探索出力の oracle 一致（R: root 値、P: 選んだ手）はどう変わるか。
# 特に po=10 と po=100 の差が「浅い探索を深い探索に近づける学習」
# （予算の蒸留）の改善余地そのものになる。
#
#   po=0   : モデルの素の値（このスクリプトが ONNX を直接推論する。
#            手の選択は audit A と同じく子局面の素の値の argmax）
#   po>0   : probe モードの出力（root 値と選択した子キー）を使う
#
# 手順（learn/ ディレクトリで、プロジェクト .venv を使用）:
#   1) po=100 の probe 出力は既存のものを再利用できる
#      （probe_arm0.txt / probe_arm2.txt / probe_arm*_g1.txt）。
#   2) po=10 と po=500 を追加で回す（モデルを差し替えて）:
#        cp model/arm0/best_single.onnx model/best_single.onnx
#        ./cpp_animal_shogi probe probe_keys_arm0.txt 10  > probe_arm0_po10.txt 2>&1
#        ./cpp_animal_shogi probe probe_keys_arm0.txt 500 > probe_arm0_po500.txt 2>&1
#   3) 曲線を作る（モデルごとに1回）:
#        ../.venv/bin/python budget_curve.py --keys probe_keys_arm0.txt \
#            --model arm0=model/arm0/best_single.onnx \
#            --probe 10=probe_arm0_po10.txt --probe 100=probe_arm0.txt \
#            --probe 500=probe_arm0_po500.txt
#      出力: budget_curve_result.md（--out で変更可）
#
# 指標（probe_score.py / raw_value_score.py と同じ真理・同じ定義）:
#   R. root 値のカテゴリ一致率（±0.2 境界）
#   P. 選んだ手のカテゴリ維持率（全体 / 勝ち局面 / 先手劣勢）と勝ち→負け率
#   段差: 隣り合う予算同士の paired 食い違い（P と R、それぞれ
#        「浅い方のみ正解 / 深い方のみ正解」の件数と差）
#
# 注意:
#   - po=0 の手の選択は終局の子局面にもモデルの素の値をそのまま使う
#     （audit_order.py の A と同じ規約）。
#   - 探索内部の VL ノイズ（σ=0.2）は全予算で残るため、同一予算でも
#     再実行で数局面は手が変わりうる。段差の解釈はその分を差し引くこと。
#   - 千日手は probe と同じく対象外（静的キーから復元できない）。

import argparse

import numpy as np

from check_unregistered import decode_key, self_test
from audit_order import legal_moves_list, child_outcome, pct
from probe_score import cat_of, load_probe

VERSION = "2026-11-23 v1（budget_curve: 探索予算の応答曲線）"

md_lines = []


def md(line=""):
    md_lines.append(line)
    print(line)


def build_truth(keys_path):
    """probe_score.score() と同じ真理: 最良カテゴリ、子キー→結果、先手劣勢。"""
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
            "child_outcome": {c.hash_key(): int(o)
                              for c, o in zip(children, outcomes)},
            "children": children,
            "sente_losing": (best == -1 and not tw) or (best == 1 and tw),
        }
    md(f"- 真理を用意できた局面: {len(truth):,}")
    return truth


def evaluate_states(model_path, states):
    """ONNX モデルの素の値を一括推論する（raw_value_score.py と同じ契約）。"""
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path,
                                providers=["CPUExecutionProvider"])
    out = []
    for i in range(0, len(states), 4096):
        batch_states = states[i:i + 4096]
        feats = np.array([np.array(s.feature(), dtype=np.float32)
                          .reshape(42, 3, 4) for s in batch_states])
        out.append(sess.run(["value"], {"input": feats})[0].reshape(-1))
    return np.concatenate(out) if out else np.array([])


def rows_po0(model_path, truth):
    """po=0: 素の値で R と P（子 argmax）を測る。"""
    from game import from_hash

    keys = list(truth.keys())
    pos_states = [from_hash(k) for k in keys]
    pos_vals = evaluate_states(model_path, pos_states)
    all_children = []
    for k in keys:
        all_children.extend(truth[k]["children"])
    child_vals = evaluate_states(model_path, all_children)

    rows = {}
    ofs = 0
    for k, pv in zip(keys, pos_vals):
        t = truth[k]
        n = len(t["children"])
        cvals = child_vals[ofs:ofs + n]
        ofs += n
        argmax_idx = int(np.argmax(-cvals))  # 手番視点 → 指した側視点
        chosen_key = t["children"][argmax_idx].hash_key()
        rows[k] = {
            "best": t["best"],
            "sente_losing": t["sente_losing"],
            "root_cat": cat_of(float(pv)),
            "chosen": t["child_outcome"][chosen_key],
        }
    return rows


def rows_probe(path, truth):
    """po>0: probe 出力から R と P を測る。"""
    res, skip = load_probe(path)
    rows = {}
    unmatched = 0
    for k, t in truth.items():
        if k not in res:
            continue
        root_w, child_key = res[k]
        chosen = t["child_outcome"].get(child_key)
        if chosen is None:
            unmatched += 1
            continue
        rows[k] = {
            "best": t["best"],
            "sente_losing": t["sente_losing"],
            "root_cat": cat_of(root_w),
            "chosen": chosen,
        }
    md(f"- probe {path}: 結果 {len(res):,} 件 / SKIP {len(skip):,} 件 / "
       f"子キー不一致 {unmatched} 件")
    return rows


def summarize_budget(budget, rows):
    rlist = list(rows.values())
    n = len(rlist)
    r_ok = sum(1 for r in rlist if r["root_cat"] == r["best"])
    keep = sum(1 for r in rlist if r["chosen"] == r["best"])
    win = [r for r in rlist if r["best"] == 1]
    win_keep = sum(1 for r in win if r["chosen"] == 1)
    win_lose = sum(1 for r in win if r["chosen"] == -1)
    sl = [r for r in rlist if r["sente_losing"]]
    sl_keep = sum(1 for r in sl if r["chosen"] == r["best"])
    md(f"| {budget} | {n:,} | **{pct(r_ok, n):.1f}%** | "
       f"**{pct(keep, n):.1f}%** | {pct(win_keep, len(win)):.1f}% | "
       f"{pct(win_lose, len(win)):.1f}% | {pct(sl_keep, len(sl)):.1f}% |")


def paired_step(lo_budget, lo_rows, hi_budget, hi_rows):
    common = [k for k in lo_rows if k in hi_rows]
    if not common:
        return
    md(f"- {lo_budget} → {hi_budget}（共通 {len(common):,} 局面）:")
    for field, name in (("chosen", "P（手の維持）"), ("root_cat", "R（root 値）")):
        lo_only = sum(1 for k in common
                      if lo_rows[k][field] == lo_rows[k]["best"]
                      and hi_rows[k][field] != hi_rows[k]["best"])
        hi_only = sum(1 for k in common
                      if hi_rows[k][field] == hi_rows[k]["best"]
                      and lo_rows[k][field] != lo_rows[k]["best"])
        md(f"  - {name}: 浅い方のみ正解 {lo_only} 件 / 深い方のみ正解 "
           f"{hi_only} 件（差 {hi_only - lo_only:+d} 件 = "
           f"{pct(hi_only - lo_only, len(common)):+.2f}pt）")


def main():
    print(f"budget_curve.py {VERSION}")
    parser = argparse.ArgumentParser(description="探索予算の応答曲線")
    parser.add_argument("--keys", default="probe_keys.txt",
                        help="局面リスト（probe_keys_*.txt）")
    parser.add_argument("--model", required=True,
                        help="label=ONNXパス（po=0 の計算用、1モデルだけ）")
    parser.add_argument("--probe", action="append", default=[],
                        help="po=probe出力ファイル（複数可。例: 100=probe_arm0.txt）")
    parser.add_argument("--out", default="budget_curve_result.md")
    args = parser.parse_args()

    if not self_test():
        print("**self-test 失敗のため中止**")
        return
    label, _, model_path = args.model.partition("=")
    if not label or not model_path:
        print("--model は label=パスの形で指定してください")
        return

    md(f"# 探索予算の応答曲線（モデル: {label}）")
    md()
    truth = build_truth(args.keys)

    budgets = {0: rows_po0(model_path, truth)}
    md(f"- {label}: po=0（素の値）を計算")
    for spec in args.probe:
        po_str, _, path = spec.partition("=")
        try:
            po = int(po_str)
        except ValueError:
            print(f"--probe は po=パスの形で指定してください: {spec}")
            return
        budgets[po] = rows_probe(path, truth)

    md()
    md("| po | 局面数 | R（root 値一致） | P（手の維持・全体） | "
       "P（勝ち局面） | 勝ち→負け | P（先手劣勢） |")
    md("|---|---|---|---|---|---|---|")
    for po in sorted(budgets):
        summarize_budget(po, budgets[po])

    md()
    md("## 段差（隣り合う予算の paired 比較）")
    md()
    ordered = sorted(budgets)
    for lo, hi in zip(ordered, ordered[1:]):
        paired_step(lo, budgets[lo], hi, budgets[hi])

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"\n結果を {args.out} に書きました")


if __name__ == "__main__":
    main()
