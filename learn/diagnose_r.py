# 結果錨付けの事前診断（diagnose_r.py）
# ====================================
# 問い: 対局結果 r は、探索値 s が持っていない「学習可能な構造」を持っているか。
# s と r が食い違う局面が、運（ε グリーディのランダム手や残り手数の長さ）で
# 説明できる範囲を超えて、特定の条件に再現性よく集中しているなら、
# r を学習に入れる価値がある。食い違いが運の露出に比例して消えるなら、
# r は薄い信号しか持たない。
#
# 方法: root レコード（実 r を持つ root arm の生成データ）について、
#   0. 規約チェック（r は手番側視点のはず → s と r の符号の一致を見る）
#   1. 較正表: s の帯域ごとに実際の勝率（r の分布）
#   2. 食い違いの分布: |s|>0.2 で s のカテゴリと r が食い違う率を、
#      終局までの残り手数 × |s| の帯域で見る（運の露出 1-0.9^k を併記）
#   3. oracle 条件付け: oracle の win/draw/lose ごとに r の分布。
#      特に oracle win で r=-1 になる率（取りこぼし）の残り手数プロファイルと、
#      決着局内の oracle draw 局面に r=±1 が付く汚染率
#   4. split-half 再現性: 対局を半分に分け、セル（oracle カテゴリ×残り手数）ごとの
#      「oracle に反する結果になった率」が両半分で一致するか（相関）。
#      一致するセル = 運ではなく構造
#
# 注意:
# - 対象は root arm のデータ（arm0 / arm1 の生成グループ）。木レコードの
#   グループ（arm2 系）は r を持たないので使えない（r=-99 の割合で警告する）
# - 千日手はスコープ外の方針に従い、引き分け対局は分けて報告し、
#   主な読みは決着局で行う
# - r は ε=10% のランダム手を含む対局の結果。完璧な評価でも結果は予測
#   できない部分があり、食い違い率は「運の露出」と並べて読むこと
#
# 実行（learn/ ディレクトリで）:
#   ../.venv/bin/python diagnose_r.py
#   ../.venv/bin/python diagnose_r.py --glob "data/arm1_gen_*" --out diagnose_r_arm1.md

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

from check_unregistered2 import LabelOracle, lookup_variants

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


def s_band(s):
    a = abs(s)
    if a >= 0.9:
        return "0.9~"
    if a >= 0.5:
        return "0.5-0.9"
    if a >= 0.2:
        return "0.2-0.5"
    return "0-0.2"


S_BANDS = ["0-0.2", "0.2-0.5", "0.5-0.9", "0.9~"]


def ply_band(k):
    """終局までの残り手数（その局面の手を含む）"""
    if k <= 2:
        return "1-2"
    if k <= 5:
        return "3-5"
    if k <= 10:
        return "6-10"
    if k <= 20:
        return "11-20"
    return "21~"


PLY_BANDS = ["1-2", "3-5", "6-10", "11-20", "21~"]
PLY_REP = {"1-2": 1.5, "3-5": 4.0, "6-10": 8.0, "11-20": 15.0, "21~": 25.0}


def oracle_label(oracle, key):
    """正規化（無ければ正規化鏡像）ラベル。手番側視点 +1/0/-1、不明は None"""
    _d, _m, normalized, normalized_mirror = lookup_variants(oracle, key)
    return normalized if normalized is not None else normalized_mirror


def load_rows(patterns, max_games=None):
    """root レコードを (s, r, plies_to_end, key, drawn_game, game_id) の行にする"""
    files = []
    for pat in patterns:
        for m in glob.glob(pat):
            if Path(m).is_dir():
                files.extend(glob.glob(m + "/*.json"))
            else:
                files.append(m)
    files = sorted(set(files))
    rows = []
    games = 0
    skipped_r99 = 0
    total_records = 0
    for jf in files:
        if max_games is not None and games >= max_games:
            break
        try:
            with open(jf) as fh:
                entries = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not entries:
            continue
        recs = []
        for e in entries:
            if "p" not in e or "s" not in e or "r" not in e:
                continue
            recs.append((int(e["p"]), float(e["s"]), int(round(float(e["r"])))))
        if not recs:
            continue
        games += 1
        n = len(recs)
        drawn = all(r == 0 for (_p, _s, r) in recs)
        for i, (p, s, r) in enumerate(recs):
            total_records += 1
            if r == -99:
                skipped_r99 += 1
                continue
            rows.append({
                "key": p, "s": s, "r": r,
                "plies_to_end": n - i,
                "drawn": drawn,
                "game": games,
            })
    return rows, games, total_records, skipped_r99


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("  OK: " if cond else "  NG: ") + name)
        ok = ok and cond

    check("カテゴリ規約", cat_of_value(0.21) == 1 and cat_of_value(-0.21) == -1
          and cat_of_value(0.2) == 0)
    check("s 帯域", s_band(0.95) == "0.9~" and s_band(0.5) == "0.5-0.9"
          and s_band(0.2) == "0.2-0.5" and s_band(0.0) == "0-0.2")
    check("残り手数帯域", ply_band(1) == "1-2" and ply_band(5) == "3-5"
          and ply_band(10) == "6-10" and ply_band(30) == "21~")
    # 合成行での集計チェック（較正表の1セル）
    rows = [
        {"key": 0, "s": 0.8, "r": 1, "plies_to_end": 3, "drawn": False, "game": 1},
        {"key": 0, "s": 0.8, "r": -1, "plies_to_end": 3, "drawn": False, "game": 1},
        {"key": 0, "s": 0.8, "r": 1, "plies_to_end": 12, "drawn": False, "game": 2},
    ]
    dec = [r for r in rows if not r["drawn"]]
    win_rate = sum(1 for r in dec if r["r"] == 1) / len(dec)
    check("勝率の集計", abs(win_rate - 2 / 3) < 1e-9)
    dis = [r for r in dec if abs(r["s"]) > 0.2
           and cat_of_value(r["s"]) != r["r"]]
    check("食い違いの検出", len(dis) == 1 and dis[0]["r"] == -1)
    print("self-test: " + ("OK" if ok else "FAILED"))
    return ok


def analyze(rows, games, total_records, skipped_r99, oracle):
    md("# 結果錨付けの事前診断（diagnose_r.py）")
    md()
    md(f"- 対局数: {games} / レコード総数: {total_records}"
       f" / r=-99（除外）: {skipped_r99} / 分析対象: {len(rows)}")
    if total_records and skipped_r99 / total_records > 0.5:
        md("- **警告: r=-99 が過半数です。木レコードのグループ（結果なし）を"
           "対象にしている可能性が高く、この診断は意味を持ちません**")
    decisive = [r for r in rows if not r["drawn"]]
    drawn_rows = [r for r in rows if r["drawn"]]
    md(f"- 決着局のレコード: {len(decisive)} / 引き分け対局のレコード: {len(drawn_rows)}")
    md()

    # ---- 0. 規約チェック ----
    md("## 0. 規約チェック")
    md()
    if decisive:
        nz = [r for r in decisive if r["s"] != 0]
        agree_n = sum(1 for r in nz if (r["s"] > 0) == (r["r"] > 0))
        md(f"- s と r の符号一致率（決着局、s≠0 の {len(nz)} 件）: "
           f"{pct(agree_n, len(nz)):.1f}%（r が手番側視点なら高いはず）")
    md()

    # ---- 1. 較正表 ----
    md("## 1. 較正表（決着局）: s の帯域ごとの実際の結果")
    md()
    md("| s 帯域 | 件数 | r=+1 | r=0 | r=-1 | 平均 r |")
    md("|---|---|---|---|---|---|")
    for lo, hi, name in [(0.9, 9, "s ≥ 0.9"), (0.5, 0.9, "0.5〜0.9"),
                         (0.2, 0.5, "0.2〜0.5"), (-0.2, 0.2, "-0.2〜0.2"),
                         (-0.5, -0.2, "-0.5〜-0.2"), (-0.9, -0.5, "-0.9〜-0.5"),
                         (-9, -0.9, "s ≤ -0.9")]:
        cell = [r for r in decisive if lo <= r["s"] < hi] \
            if name != "s ≥ 0.9" else [r for r in decisive if r["s"] >= 0.9]
        if name == "s ≤ -0.9":
            cell = [r for r in decisive if r["s"] <= -0.9]
        if not cell:
            continue
        n = len(cell)
        p1 = pct(sum(1 for r in cell if r["r"] == 1), n)
        p0 = pct(sum(1 for r in cell if r["r"] == 0), n)
        pm = pct(sum(1 for r in cell if r["r"] == -1), n)
        mr = sum(r["r"] for r in cell) / n
        md(f"| {name} | {n} | {p1:.1f}% | {p0:.1f}% | {pm:.1f}% | {mr:+.3f} |")
    md()
    md("読み: 平均 r が s の水準から系統的にずれる帯域は、較正の構造的な欠陥。")
    md()

    # ---- 2. 食い違いの分布 ----
    md("## 2. 食い違いの分布（決着局、|s|>0.2 で s のカテゴリ ≠ r）")
    md()
    md("運の露出 = その局面の後に ε ランダム手が少なくとも1回出る確率 1-0.9^k"
       "（k は帯域の代表残り手数）。食い違い率が露出と同程度に消えるなら運、"
       "残り手数が小さく露出が低いのに残るなら構造。")
    md()
    md("| 残り手数 | 運の露出 | 全体 | |s| 0.2-0.5 | |s| 0.5-0.9 | |s| 0.9~ |")
    md("|---|---|---|---|---|---|")
    for pb in PLY_BANDS:
        expo = 1 - 0.9 ** PLY_REP[pb]
        line = [f"| {pb} | {expo * 100:.0f}% |"]
        for band in [None] + S_BANDS[1:]:
            cell = [r for r in decisive
                    if ply_band(r["plies_to_end"]) == pb and abs(r["s"]) > 0.2
                    and (band is None or s_band(r["s"]) == band)]
            if not cell:
                line.append(" - |")
                continue
            dis = sum(1 for r in cell if cat_of_value(r["s"]) != r["r"])
            line.append(f" {pct(dis, len(cell)):.1f}%（{len(cell)}） |")
        md("".join(line))
    md()

    # ---- 3. oracle 条件付け ----
    md("## 3. oracle 条件付け")
    md()
    for r in rows:
        r["oracle"] = oracle_label(oracle, r["key"])
    lab = [r for r in rows if r["oracle"] is not None]
    md(f"- oracle ラベルを用意できたレコード: {len(lab)} / {len(rows)}")
    md()
    for oc, oc_name in [(1, "oracle win（手番側勝ち）"), (-1, "oracle lose（手番側負け）")]:
        cell = [r for r in lab if r["oracle"] == oc and not r["drawn"]]
        if not cell:
            continue
        contra = sum(1 for r in cell if r["r"] == -oc)
        md(f"### {oc_name}（決着局、{len(cell)} 件）: oracle に反する結果 {pct(contra, len(cell)):.1f}%")
        md()
        md("| 残り手数 | 件数 | 反する率 |")
        md("|---|---|---|")
        for pb in PLY_BANDS:
            c2 = [r for r in cell if ply_band(r["plies_to_end"]) == pb]
            if not c2:
                continue
            md(f"| {pb} | {len(c2)} | {pct(sum(1 for r in c2 if r['r'] == -oc), len(c2)):.1f}% |")
        md()
    dcell = [r for r in lab if r["oracle"] == 0]
    ddec = [r for r in dcell if not r["drawn"]]
    if dcell:
        md(f"### oracle draw（{len(dcell)} 件）")
        md()
        md(f"- 決着局内: {len(ddec)} 件、うち r=±1（汚染） "
           f"{pct(sum(1 for r in ddec if r['r'] != 0), len(ddec)):.1f}%"
           f"（r=+1 {sum(1 for r in ddec if r['r'] == 1)} / r=-1 {sum(1 for r in ddec if r['r'] == -1)}）")
        md(f"- 引き分け対局内: {len(dcell) - len(ddec)} 件（r=0 のみのはず）")
        md()
    dec_lab = [r for r in lab if not r["drawn"] and r["oracle"] != 0]
    if dec_lab:
        as_ = sum(1 for r in dec_lab if cat_of_value(r["s"]) == r["oracle"])
        ar_ = sum(1 for r in dec_lab if r["r"] == r["oracle"])
        md(f"- oracle（決着カテゴリ）との一致: s のカテゴリ {pct(as_, len(dec_lab)):.1f}%"
           f" / r {pct(ar_, len(dec_lab)):.1f}%（{len(dec_lab)} 件）")
        md()
    md()

    # ---- 4. split-half 再現性 ----
    md("## 4. split-half 再現性（oracle に反する結果の率、決着局・oracle 決着局面）")
    md()
    game_ids = sorted(set(r["game"] for r in dec_lab))
    half_a = set(game_ids[0::2])
    cells = []
    for oc in (1, -1):
        for pb in PLY_BANDS:
            ca = [r for r in dec_lab if r["oracle"] == oc
                  and ply_band(r["plies_to_end"]) == pb and r["game"] in half_a]
            cb = [r for r in dec_lab if r["oracle"] == oc
                  and ply_band(r["plies_to_end"]) == pb and r["game"] not in half_a]
            if len(ca) >= 30 and len(cb) >= 30:
                ra = sum(1 for r in ca if r["r"] == -oc) / len(ca)
                rb = sum(1 for r in cb if r["r"] == -oc) / len(cb)
                cells.append((oc, pb, len(ca), ra, len(cb), rb))
    md("| oracle | 残り手数 | A 半分 n | A 反する率 | B 半分 n | B 反する率 |")
    md("|---|---|---|---|---|---|")
    for oc, pb, na, ra, nb, rb in cells:
        md(f"| {'win' if oc == 1 else 'lose'} | {pb} | {na} | {ra * 100:.1f}% | {nb} | {rb * 100:.1f}% |")
    if len(cells) >= 3:
        xs = [c[3] for c in cells]
        ys = [c[5] for c in cells]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        corr = cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0
        md()
        md(f"- セル間の相関（A 半分 vs B 半分）: **{corr:+.2f}**"
           "（1 に近いほどセルの偏りが再現性のある構造、0 付近なら運）")
    md()
    md("## 読み方の目安")
    md()
    md("- §2 で残り手数 1-2・|s| 大の食い違いが運の露出を大きく上回る"
       " → 終盤の取りこぼしという構造があり、r の信号は学習可能かもしれない")
    md("- §4 の相関が高い → セル単位の偏りは実在する（arm 化の根拠になる）")
    md("- 食い違いが残り手数と運の露出に比例して消え、相関も低い"
       " → r の食い違いはほぼ運。結果錨付けは見送るのが妥当")


def main():
    parser = argparse.ArgumentParser(description="結果錨付けの事前診断")
    parser.add_argument("--glob", action="append", default=[],
                        help="対象グループ（ディレクトリ名か glob）。複数指定可")
    parser.add_argument("--out", default="diagnose_r_result.md")
    parser.add_argument("--max-games", type=int, default=None)
    args = parser.parse_args()

    print(f"diagnose_r.py {VERSION}（結果錨付けの事前診断）")
    if not self_test():
        print("self-test 失敗のため中止")
        return
    patterns = args.glob if args.glob else ["data/arm0_gen_*"]
    rows, games, total_records, skipped = load_rows(patterns, args.max_games)
    if not rows:
        print("対象レコードがありません（--glob を確認してください）")
        return
    oracle = LabelOracle()
    analyze(rows, games, total_records, skipped, oracle)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"結果を {args.out} に書きました")


if __name__ == "__main__":
    main()
