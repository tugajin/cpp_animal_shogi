# probe のスコア付け（probe_score.py）
# ====================================
# 問い: モデル単体の順序（audit の A）が良くなると、それを内蔵した探索の
# 出力（選ぶ手・root 値）も oracle に近づくのか。
# audit の C は記録された実着手で測るためデータの性質だったが、固定局面に
# 対して各モデルで探索を回し直せば（バイナリの probe モード）、C を
# モデルの指標として測れる。
#
# 手順:
#   1) 局面リスト生成（audit_order.py と同じ復元・同じ seed で抽くので、
#      対象局面は audit の調査局面と一致する）:
#        ../.venv/bin/python probe_score.py --make-list --group arm0_gen_19 \
#            --keys-out probe_keys_arm0.txt
#   2) モデルを差し替えて probe を回す（モデルの数だけ繰り返す）:
#        cp model/arm0/best_single.onnx model/best_single.onnx
#        ./cpp_animal_shogi_arm0 probe probe_keys_arm0.txt > probe_arm0.txt 2>&1
#        cp model/arm2/best_single.onnx model/best_single.onnx
#        ./cpp_animal_shogi_arm2 probe probe_keys_arm0.txt > probe_arm2.txt 2>&1
#   3) スコア付け:
#        ../.venv/bin/python probe_score.py --keys probe_keys_arm0.txt \
#            --probe arm0=probe_arm0.txt --probe arm2=probe_arm2.txt
#
# 指標:
#   P. 探索が選んだ手のカテゴリ維持（audit の C のモデル指標版）。
#      勝ち局面での内訳（維持 / 引分へ低下 / 負けへ低下）を必ず見る。
#   R. 探索の root 値のカテゴリ一致率（±0.2 境界。「探索版 ans」=
#      そのモデルが生む教師の質）。
#
# 注意:
#   - probe の手選択は ε=0（choice_best_move_e_greedy から 10% ランダムと
#     微小なタイブレーク乱数を除いたもの）。生成時の実着手とはこの分だけ違う。
#   - 探索内部の VL ノイズ（σ=0.2）は残るので、同一モデルでも再実行で
#     数局面は手が変わりうる。3,000 局面規模では平均化される。
#   - 千日手は静的キーから復元できないため、probe 局面は通常局面として
#     探索される（audit の子局面評価も同じ前提）。

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from check_unregistered import decode_key, self_test
from audit_order import legal_moves_list, child_outcome, CAT_NAME, pct

VERSION = "2026-11-22 v1（probe: 探索出力の oracle 照合）"

md_lines = []


def md(line=""):
    md_lines.append(line)
    print(line)


def reconstruct_positions(group):
    """audit_order.py の対局復元をそのまま写したもの（同じ順序・同じ上限）。"""
    from game import from_hash

    files = sorted(glob.glob("data/**/*.json", recursive=True))
    positions = []  # (key, played_child_key or None)
    for jf in files:
        if Path(jf).parent.name != group:
            continue
        with open(jf) as fh:
            try:
                entries = json.load(fh)
            except json.JSONDecodeError:
                continue
        prev_state = None
        prev_key = None
        for e in entries:
            if len(positions) >= 60000:
                break
            key = int(e["p"])
            if prev_state is not None:
                child_keys = set()
                for a in legal_moves_list(prev_state):
                    child_keys.add(prev_state.next(a).hash_key())
                if key in child_keys:
                    positions[-1] = (prev_key, key)
            positions.append((key, None))
            prev_state = from_hash(key)
            prev_key = key
    return positions


def eligible(oracle, key):
    """audit_order.py の調査対象条件と同じ: 終局でなく、全子局面の結果が分かる。"""
    from game import from_hash

    state = from_hash(key)
    if state.is_win() or state.is_lose():
        return False
    actions = legal_moves_list(state)
    if len(actions) == 0:
        return False
    children = [state.next(a) for a in actions]
    outcomes = [child_outcome(oracle, c) for c in children]
    return all(o is not None for o in outcomes)


def make_list(args):
    from check_unregistered2 import LabelOracle

    oracle = LabelOracle()
    positions = reconstruct_positions(args.group)
    md(f"- 復元した局面数: {len(positions):,}（group={args.group}）")
    rng = np.random.default_rng(123)
    if len(positions) > args.sample:
        idx = rng.choice(len(positions), args.sample, replace=False)
        positions = [positions[i] for i in idx]
    keys = []
    skipped = 0
    for key, _played in positions:
        if eligible(oracle, key):
            keys.append(key)
        else:
            skipped += 1
    with open(args.keys_out, "w", encoding="utf-8") as fh:
        fh.write(f"# probe keys group={args.group} sample={args.sample} "
                 f"version={VERSION}\n")
        for k in keys:
            fh.write(f"{k}\n")
    md(f"- 書き出したキー: {len(keys):,}（対象外 {skipped:,}） -> {args.keys_out}")


def load_probe(path):
    """PROBE 行を {key: (root_w, child_key)} に、PROBE_SKIP を集合で返す。"""
    res = {}
    skip = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 4 and parts[0] == "PROBE":
                res[int(parts[1])] = (float(parts[2]), int(parts[3]))
            elif len(parts) == 2 and parts[0] == "PROBE_SKIP":
                skip.add(int(parts[1]))
    return res, skip


def cat_of(value):
    if value > 0.2:
        return 1
    if value < -0.2:
        return -1
    return 0


def score(args):
    from check_unregistered2 import LabelOracle
    from game import from_hash

    keys = []
    with open(args.keys, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                keys.append(int(line))
    md(f"- 局面リスト: {args.keys}（{len(keys):,} 局面）")

    oracle = LabelOracle()
    # 局面ごとの真理: best（=局面カテゴリ）、子キー→結果、先手劣勢フラグ
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
            "sente_losing": (best == -1 and not tw) or (best == 1 and tw),
        }
    md(f"- 真理を用意できた局面: {len(truth):,}")

    labels = []
    per_label = {}
    for spec in args.probe:
        label, _, path = spec.partition("=")
        res, skip = load_probe(path)
        labels.append(label)
        rows = {}
        unmatched = 0
        for key, t in truth.items():
            if key not in res:
                continue
            root_w, child_key = res[key]
            chosen = t["child_outcome"].get(child_key)
            if chosen is None:
                unmatched += 1
                continue
            rows[key] = {
                "best": t["best"],
                "chosen": chosen,
                "root_cat": cat_of(root_w),
                "sente_losing": t["sente_losing"],
            }
        per_label[label] = {
            "rows": rows, "n_probe_skip": len(skip), "unmatched": unmatched,
        }
        md(f"- {label}: probe 結果 {len(res):,} 件 / SKIP {len(skip):,} 件 / "
           f"子キー不一致 {unmatched} 件")

    def summarize(label, sub, title):
        if not sub:
            return
        keep = sum(1 for r in sub if r["chosen"] == r["best"])
        d1 = sum(1 for r in sub if r["best"] - r["chosen"] == 1)
        d2 = sum(1 for r in sub if r["best"] - r["chosen"] == 2)
        md(f"  - {title}（{len(sub):,}）: 維持 **{pct(keep, len(sub)):.1f}%** / "
           f"1段階低下 {pct(d1, len(sub)):.1f}% / 2段階低下 {pct(d2, len(sub)):.1f}%")

    for label in labels:
        rows = list(per_label[label]["rows"].values())
        md()
        md(f"## {label}")
        md()
        root_ok = sum(1 for r in rows if r["root_cat"] == r["best"])
        md(f"- R. root 値のカテゴリ一致: **{pct(root_ok, len(rows)):.1f}%**"
           f"（{len(rows):,} 局面）")
        md(f"- P. 探索の手のカテゴリ維持:")
        summarize(label, rows, "全体")
        summarize(label, [r for r in rows if r["best"] == 1], "勝ち局面")
        summarize(label, [r for r in rows if r["best"] == 0], "引分局面（参考）")
        summarize(label, [r for r in rows if r["sente_losing"]], "先手劣勢局面")

    if len(labels) >= 2:
        base = labels[0]
        md()
        md(f"## paired 比較（基準: {base}）")
        md()
        for label in labels[1:]:
            a = per_label[base]["rows"]
            b = per_label[label]["rows"]
            common = [k for k in a if k in b]
            a_only = sum(1 for k in common
                         if a[k]["chosen"] == a[k]["best"]
                         and b[k]["chosen"] != b[k]["best"])
            b_only = sum(1 for k in common
                         if b[k]["chosen"] == b[k]["best"]
                         and a[k]["chosen"] != a[k]["best"])
            md(f"- {label}: 共通 {len(common):,} 局面で、{base} のみ維持 "
               f"{a_only} 件 / {label} のみ維持 {b_only} 件"
               f"（差 {b_only - a_only:+d} 件 = "
               f"{pct(b_only - a_only, len(common)):+.2f}pt）")


def main():
    print(f"probe_score.py {VERSION}")
    parser = argparse.ArgumentParser(description="probe のスコア付け")
    parser.add_argument("--make-list", action="store_true",
                        help="局面リストを生成して終わる")
    parser.add_argument("--group", default="arm0_gen_19",
                        help="--make-list: 復元対象のデータグループ名")
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--keys-out", default="probe_keys.txt")
    parser.add_argument("--keys", default="probe_keys.txt",
                        help="スコア付け: 局面リスト")
    parser.add_argument("--probe", action="append", default=[],
                        help="スコア付け: label=probe出力ファイル（複数可）")
    parser.add_argument("--out", default="probe_result.md")
    args = parser.parse_args()

    if not self_test():
        print("**self-test 失敗のため中止**")
        return
    if args.make_list:
        make_list(args)
        return
    if not args.probe:
        print("--probe label=ファイル を1つ以上指定してください")
        return
    md("# probe スコア（探索出力の oracle 照合）")
    md()
    score(args)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"\n結果を {args.out} に書きました")


if __name__ == "__main__":
    main()
