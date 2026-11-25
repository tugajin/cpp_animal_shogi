# 深さ別の測定（diagnose_depth.py）
# ====================================
# 問い: B1 対照実験で「深い局面を学ばないほうが良い」ことが分かったが、
# その理由はどれか。学習は追加せず、既存の生成データと正解データベース
# （oracle）だけで、理由の候補2つを直接測る。
#
#   理由2（評価値の裏付けが薄い）:
#     §1 で、深さごとに記録の評価値 s のカテゴリが正解とどれだけ
#     一致するかを見る。深さとともに一致率が下がるなら、深い局面の
#     評価値は薄い（= 探索による修正が効いていない）ことの支持になる。
#     下がらないなら、評価値の質は理由ではない。
#
#   理由1（後で本番として学び直される）:
#     §2 で、深さ1以上の記録の局面が、同じ生成データの中で
#     「手を選ぶ局面（深さ0の記録）」として現れる割合を見る。
#     先読みの中だけの局面が、実際に対局が進んで本番の局面になるなら、
#     その時点で（その局面自身を出発点とする探索付きで）学び直される
#     ので、先読み段階で学ぶ必要は薄い。
#
#   理由3（枠の奪い合い）は算数なので測らない。§0 の深さ構成比が
#   そのまま根拠になる（深さ2以上が記録の約7割）。
#
# 対象データ: 深さ "d" を持つ木レコードの生成グループ。
# 既定は data/arm2d1_gen_*（DEPTH arm の生成データ。全レコードに
# "d" が付いていることが学習実行時に確認済み）。
# "d" の無いレコードが混ざっていたら、黙って混ぜずに中止する。
#
# 注意:
# - 深さは対局の進み具合と相関する（深い記録ほど対局の後半寄りに
#   なりやすい）。§1 の深さ勾配はその前提で読むこと
# - §2 の「現れる」は同一生成グループ内と全グループ通算の2通り。
#   どちらも順序なしの集合判定（その局面が本番の局面として
#   データに存在するか）で、同一対局内での前後関係までは見ない
# - 引き分け局面の数値は参考（千日手はスコープ外の方針）
#
# 実行（learn/ ディレクトリで）:
#   ../.venv/bin/python diagnose_depth.py
#   ../.venv/bin/python diagnose_depth.py --glob "data/arm2d1_gen_*" --out diagnose_depth_result.md

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

VERSION = "2026-11-24 v1"

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


def depth_band(d):
    if d <= 4:
        return str(d)
    return "5+"


DEPTH_BANDS = ["0", "1", "2", "3", "4", "5+"]


def parse_entries(entries):
    """1ファイル分の JSON 配列を (key, s, d) のリストにする。
    "d" の無いレコードは数えて返す（呼び出し側で中止判断する）"""
    recs = []
    missing_d = 0
    for e in entries:
        if "p" not in e or "s" not in e:
            continue
        if "d" not in e:
            missing_d += 1
            continue
        recs.append((int(e["p"]), float(e["s"]), int(e["d"])))
    return recs, missing_d


def load_groups(patterns):
    """生成グループ（ディレクトリ）ごとにレコードを読む。
    戻り値: (groups, total_files, missing_d)
    groups = [{"name": str, "records": [(key, s, d), ...]}, ...]"""
    dirs = []
    loose_files = []
    for pat in patterns:
        for m in sorted(glob.glob(pat)):
            if Path(m).is_dir():
                dirs.append(m)
            else:
                loose_files.append(m)
    groups = []
    total_files = 0
    missing_d = 0

    def read_dir(name, files):
        nonlocal total_files, missing_d
        recs_all = []
        for jf in files:
            total_files += 1
            try:
                with open(jf) as fh:
                    entries = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            if not entries:
                continue
            recs, miss = parse_entries(entries)
            missing_d += miss
            recs_all.extend(recs)
        if recs_all:
            groups.append({"name": name, "records": recs_all})

    for d in dirs:
        read_dir(d, sorted(glob.glob(d + "/*.json")))
    if loose_files:
        read_dir("(loose files)", sorted(set(loose_files)))
    return groups, total_files, missing_d


def analyze(groups, lookup):
    """lookup(key) -> 正解ラベル (+1/0/-1) または None。
    戻り値: 深さ帯ごとの集計 dict"""
    # 本番の局面（深さ0）のキー集合: グループ内 / 全体
    all_root_keys = set()
    for g in groups:
        g["root_keys"] = {k for (k, _s, d) in g["records"] if d == 0}
        all_root_keys |= g["root_keys"]

    label_cache = {}

    def label_of(key):
        if key not in label_cache:
            label_cache[key] = lookup(key)
        return label_cache[key]

    stats = {}
    for band in DEPTH_BANDS:
        stats[band] = {
            "n": 0, "covered": 0, "agree": 0,
            "win_n": 0, "win_agree": 0,
            "lose_n": 0, "lose_agree": 0,
            "s_sum_win": 0.0, "s_sum_lose": 0.0,
            "appear_group": 0, "appear_all": 0,
        }
    total = 0
    for g in groups:
        for (key, s, d) in g["records"]:
            total += 1
            st = stats[depth_band(d)]
            st["n"] += 1
            if key in g["root_keys"]:
                st["appear_group"] += 1
            if key in all_root_keys:
                st["appear_all"] += 1
            lab = label_of(key)
            if lab is None:
                continue
            st["covered"] += 1
            if cat_of_value(s) == lab:
                st["agree"] += 1
            if lab == 1:
                st["win_n"] += 1
                st["s_sum_win"] += s
                if cat_of_value(s) == 1:
                    st["win_agree"] += 1
            elif lab == -1:
                st["lose_n"] += 1
                st["s_sum_lose"] += s
                if cat_of_value(s) == -1:
                    st["lose_agree"] += 1
    return stats, total, len(label_cache)


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("  OK: " if cond else "  NG: ") + name)
        ok = ok and cond

    check("カテゴリ規約", cat_of_value(0.21) == 1 and cat_of_value(-0.21) == -1
          and cat_of_value(0.2) == 0)
    check("深さ帯", depth_band(0) == "0" and depth_band(4) == "4"
          and depth_band(9) == "5+")

    recs, miss = parse_entries([
        {"p": 10, "s": 0.5, "d": 2},
        {"p": 11, "s": -0.5},
        {"p": 12, "s": 0.0, "d": 0},
    ])
    check("parse_entries（d 無しを数える）",
          recs == [(10, 0.5, 2), (12, 0.0, 0)] and miss == 1)

    # 合成データ: 出発点キーは {100, 200, 102, 300}
    groups = [{"name": "synthetic", "records": [
        (100, 0.9, 0), (101, 0.8, 1), (102, 0.7, 2), (103, -0.9, 3),
        (200, -0.8, 0), (201, 0.1, 2),
        (102, 0.75, 0), (300, 0.0, 0), (301, -0.5, 5),
    ]}]
    labels = {100: 1, 101: 1, 102: 1, 103: -1, 200: -1,
              201: 0, 300: 0, 301: -1}
    stats, total, _nlab = analyze(groups, labels.get)
    check("合成データ総数", total == 9)
    check("深さ0の一致", stats["0"]["n"] == 4 and stats["0"]["agree"] == 4)
    check("深さ2の一致", stats["2"]["n"] == 2 and stats["2"]["agree"] == 2)
    check("深さ2の本番到達（102のみ）",
          stats["2"]["appear_group"] == 1 and stats["2"]["appear_all"] == 1)
    check("深さ1の本番到達なし", stats["1"]["appear_group"] == 0)
    check("win セルの集計", stats["0"]["win_n"] == 2
          and stats["0"]["win_agree"] == 2)
    return ok


def main():
    parser = argparse.ArgumentParser(description="深さ別の測定")
    parser.add_argument("--glob", action="append", default=None,
                        help="生成グループのパターン（複数指定可）")
    parser.add_argument("--out", default="diagnose_depth_result.md")
    args = parser.parse_args()
    patterns = args.glob if args.glob else ["data/arm2d1_gen_*"]

    print(f"diagnose_depth.py {VERSION}（深さ別の測定）")
    print("self-test:")
    if not self_test():
        print("self-test: NG のため中止")
        return 1
    print("self-test: OK")

    groups, total_files, missing_d = load_groups(patterns)
    total_records = sum(len(g["records"]) for g in groups)
    if missing_d > 0:
        print(f'中止: "d" の無いレコードが {missing_d} 件あります。'
              "深さを持つ生成グループだけを指定してください。")
        return 1
    if total_records == 0:
        print(f"中止: パターン {patterns} にレコードがありません。")
        return 1

    from check_unregistered2 import LabelOracle, lookup_variants
    oracle = LabelOracle()

    def lookup(key):
        _d, _m, normalized, normalized_mirror = lookup_variants(oracle, key)
        return normalized if normalized is not None else normalized_mirror

    stats, _total, n_unique = analyze(groups, lookup)

    md("# 深さ別の測定（diagnose_depth.py）")
    md()
    md(f"- 対象: {patterns}（{len(groups)} グループ / {total_files} ファイル）")
    md(f"- レコード総数: {total_records:,}（うち正解を引けた局面は "
       f"{n_unique:,} 種類）")
    md()
    md("## §0 深さの構成比（枠の奪い合いの算数）")
    md()
    for band in DEPTH_BANDS:
        st = stats[band]
        md(f"- 深さ {band}: {st['n']:,} 件（{pct(st['n'], total_records):.1f}%）")
    deep = sum(stats[b]["n"] for b in ["2", "3", "4", "5+"])
    md(f"- 深さ2以上 合計: {deep:,} 件（{pct(deep, total_records):.1f}%）")
    md()
    md("## §1 深さ別の評価値の正解一致（理由2: 裏付けの薄さ）")
    md()
    md("記録の評価値 s のカテゴリ（±0.2 で勝ち/引分/負け）が、"
       "正解と一致する割合。")
    md()
    for band in DEPTH_BANDS:
        st = stats[band]
        if st["covered"] == 0:
            md(f"- 深さ {band}: 正解を引けた記録なし")
            continue
        mean_win = st["s_sum_win"] / st["win_n"] if st["win_n"] else 0.0
        mean_lose = st["s_sum_lose"] / st["lose_n"] if st["lose_n"] else 0.0
        md(f"- 深さ {band}: 全体 **{pct(st['agree'], st['covered']):.1f}%**"
           f"（{st['covered']:,} 件）"
           f" / 正解が勝ちの局面 {pct(st['win_agree'], st['win_n']):.1f}%"
           f"（{st['win_n']:,} 件、s の平均 {mean_win:+.3f}）"
           f" / 正解が負けの局面 {pct(st['lose_agree'], st['lose_n']):.1f}%"
           f"（{st['lose_n']:,} 件、s の平均 {mean_lose:+.3f}）")
    md()
    md("## §2 本番到達率（理由1: 後で本番として学び直される）")
    md()
    md("その深さの記録の局面が、「手を選ぶ局面（深さ0）」として"
       "データに現れる割合。")
    md()
    for band in ["1", "2", "3", "4", "5+"]:
        st = stats[band]
        if st["n"] == 0:
            continue
        md(f"- 深さ {band}: 同じ生成グループ内 "
           f"**{pct(st['appear_group'], st['n']):.1f}%**"
           f" / 全グループ通算 **{pct(st['appear_all'], st['n']):.1f}%**"
           f"（{st['n']:,} 件）")
    md()
    md("## 読み方")
    md()
    md("- §1 で一致率が深さとともに明確に下がる → 理由2（深い局面の"
       "評価値は裏付けが薄い）を支持。下がらない → 評価値の質は理由ではない")
    md("- §2 で深さ2の到達率が高い → 理由1（後で本番として学び直される）"
       "を支持。深さとともに急に下がるはず")
    md("- 両方とも弱い → 主因は残りの2つ（枠の奪い合い＝§0、"
       "実際には現れない局面の多さ）")
    md("- 注意: 深さは対局の進み具合と相関する（深い記録ほど後半寄り）。"
       "§1 の勾配はその前提で読む")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"結果を {args.out} に書きました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
