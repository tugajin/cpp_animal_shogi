# CN 選択の生成実験: 分析（analyze_cn.py）
# ========================================================
# 問い1（生成比較）: 同じ po=100 で、CN 選択で生成した教師 s は
#   UBFM 選択（gen_po100）より正確か。測定は investigate45 の C と同じ方式
#   （正規化+鏡像照合、カテゴリ別正解率、seed=123 サンプル）。
# 問い2（注釈の妥当性）: レコードの "c"（根のカテゴリ共謀数）で層別したとき、
#   s のオラクル正解率は CN が大きいほど上がるか。あわせて CN と
#   維持手数（オラクル定義、phase0_fragility と同じ数え方）の関係を見る。
# 問い3（実装の相互検証）: c=-1（CN ∞ = 証明済み級）のレコードは、
#   |s|>0.5 では格子一致（k≠200）になっているはず。
#
# 実行（learn/ ディレクトリで）:
#   ../.venv/bin/python analyze_cn.py
#   ../.venv/bin/python analyze_cn.py --cn-group gen_cn100 --base-group gen_po100
# 出力: analyze_cn_result.md
#
# 注意:
#   - オラクルは評価専用。学習には使わない
#   - gamelibs の MoveList は len()+インデックスでアクセスすること

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from check_unregistered import decode_key, self_test as cu_self_test
from check_unregistered2 import LabelOracle, lookup_variants

THRESHOLD = 0.2
BASE = 0.9999
STEP = 1.0 / 2000.0
TOL = 1.0e-5

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


def lattice_k(abs_s):
    """|s| が証明値の格子 0.9999 - k/2000 に一致すれば k、しなければ None
    （proven_label_check.py と同じ定義）"""
    if abs_s <= 0.5:
        return None
    k = round((BASE - abs_s) / STEP)
    if k < 0:
        return None
    return k if abs(BASE - STEP * k - abs_s) <= TOL else None


def cn_bucket(c):
    if c is None:
        return "なし"
    if c < 0:
        return "∞"
    if c <= 4:
        return str(c)
    return "5+"


def load_group(group):
    files = sorted(glob.glob("data/**/*.json", recursive=True))
    entries = []
    used = 0
    for jf in files:
        if Path(jf).parent.name != group:
            continue
        with open(jf) as fh:
            try:
                entries.extend(json.load(fh))
                used += 1
            except json.JSONDecodeError:
                continue
    return entries, used


def oracle_label_turn(oracle, key):
    """手番側視点のオラクルラベル（正規化+鏡像）。未登録なら None"""
    _d, _m, normalized, normalized_mirror = lookup_variants(oracle, key)
    return normalized if normalized is not None else normalized_mirror


def teacher_rows(entries, oracle, sample_size, seed=123):
    """investigate45 の C と同じ条件で (label, s_pred, turn_is_white, c) を作る。
    終局局面は from_hash で判定して除外する"""
    from game import from_hash
    rng = np.random.default_rng(seed)
    if len(entries) > sample_size:
        idx = rng.choice(len(entries), sample_size, replace=False)
        entries = [entries[i] for i in idx]
        sampled = True
    else:
        sampled = False
    rows = []
    skipped_terminal = 0
    unknown = 0
    for e in entries:
        key = int(e["p"])
        state = from_hash(key)
        if state.pos.is_win() or state.pos.is_lose():
            skipped_terminal += 1
            continue
        label = oracle_label_turn(oracle, key)
        if label is None:
            unknown += 1
            continue
        turn_is_white, _, _ = decode_key(key)
        rows.append((label, classify(float(e["s"])), turn_is_white, e.get("c")))
    return rows, skipped_terminal, unknown, sampled


def accuracy_line(rows, title):
    md(f"### {title}（{len(rows):,} 件）")
    md()
    md("| カテゴリ | 件数 | s 正解率 |")
    md("|---|---|---|")
    for lv, name in ((1, "win"), (-1, "lose"), (0, "draw")):
        sub = [r for r in rows if r[0] == lv]
        if not sub:
            continue
        acc = pct(sum(1 for r in sub if r[1] == lv), len(sub))
        md(f"| {name} | {len(sub):,} | {acc:.1f}% |")
    tot = pct(sum(1 for r in rows if r[1] == r[0]), len(rows)) if rows else 0.0
    md(f"| **全体** | {len(rows):,} | **{tot:.1f}%** |")
    md()
    return tot


def sente_losing_acc(rows):
    """先手が負けている局面域（先手番 label=-1 / 後手番 label=+1）での s 正解率"""
    sub = [r for r in rows if (not r[2] and r[0] == -1) or (r[2] and r[0] == 1)]
    if not sub:
        return 0, 0.0
    return len(sub), pct(sum(1 for r in sub if r[1] == r[0]), len(sub))


def legal_moves_list(state):
    ml = state.legal_actions()
    return [ml[i] for i in range(len(ml))]


def child_outcome(oracle, child_state):
    """phase0_fragility と同じ: 子局面の結果を親の手番側視点で返す"""
    key = child_state.hash_key()
    label = oracle_label_turn(oracle, key)
    if label is not None:
        return -label
    if child_state.is_lose():
        return 1
    if child_state.is_win():
        return -1
    return None


def spearman(x, y):
    def rank(a):
        order = np.argsort(a)
        r = np.empty(len(a), dtype=float)
        r[order] = np.arange(len(a), dtype=float)
        return r
    rx, ry = rank(np.asarray(x, dtype=float)), rank(np.asarray(y, dtype=float))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def self_test():
    assert classify(0.21) == 1 and classify(-0.21) == -1 and classify(0.2) == 0
    assert lattice_k(0.9999) == 0
    assert lattice_k(0.8999) == 200
    assert lattice_k(0.5) is None
    assert cn_bucket(-1) == "∞" and cn_bucket(3) == "3" and cn_bucket(9) == "5+"
    print("self-test: OK")
    return True


def main():
    parser = argparse.ArgumentParser(description="CN 選択の生成実験 分析")
    parser.add_argument("--cn-group", default="gen_cn100")
    parser.add_argument("--base-group", default="gen_po100")
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--preserving-sample", type=int, default=1500)
    parser.add_argument("--out", default="analyze_cn_result.md")
    args = parser.parse_args()

    md("# CN 選択の生成実験 分析（analyze_cn.py）")
    md()
    if not (self_test() and cu_self_test()):
        md("**self-test 失敗のため中止**")
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md_lines) + "\n")
        return

    oracle = LabelOracle()
    cn_entries, cn_files = load_group(args.cn_group)
    base_entries, base_files = load_group(args.base_group)
    md(f"- CN グループ `{args.cn_group}`: {cn_files} ファイル、{len(cn_entries):,} レコード")
    md(f"- 比較グループ `{args.base_group}`: {base_files} ファイル、{len(base_entries):,} レコード")
    n_with_c = sum(1 for e in cn_entries if "c" in e)
    md(f"- CN グループで c 注釈のあるレコード: {n_with_c:,} 件")
    md()
    if not cn_entries or not base_entries:
        md("**どちらかのグループが空のため中止**（生成とグループ名を確認）")
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md_lines) + "\n")
        return

    # ---- 問い1: 教師の正確さの比較 ----
    md("## 問い1. 教師 s の正確さ（CN 選択 vs UBFM 選択、ともに po=100）")
    md()
    cn_rows, cn_skip, cn_unk, cn_smp = teacher_rows(cn_entries, oracle, args.sample)
    base_rows, base_skip, base_unk, base_smp = teacher_rows(base_entries, oracle, args.sample)
    md(f"- CN: 終局除外 {cn_skip} / 未登録 {cn_unk} / 集計 {len(cn_rows):,} 件" + ("（サンプル）" if cn_smp else ""))
    md(f"- UBFM: 終局除外 {base_skip} / 未登録 {base_unk} / 集計 {len(base_rows):,} 件" + ("（サンプル）" if base_smp else ""))
    md()
    cn_tot = accuracy_line(cn_rows, f"CN 選択（{args.cn_group}）")
    base_tot = accuracy_line(base_rows, f"UBFM 選択（{args.base_group}）")
    cn_sl_n, cn_sl = sente_losing_acc(cn_rows)
    base_sl_n, base_sl = sente_losing_acc(base_rows)
    md(f"- 先手劣勢セル: CN {cn_sl:.1f}%（{cn_sl_n:,} 件） / UBFM {base_sl:.1f}%（{base_sl_n:,} 件）")
    md(f"- 全体差: CN {cn_tot:.1f}% − UBFM {base_tot:.1f}% = {cn_tot - base_tot:+.1f} pt")
    md()
    md("読み方: 生成条件（モデル・po・局数）が揃っている前提で、CN 側の win/lose 正解率が"
       "明確に高ければ「葉選択を共謀に寄せると教師が正確になる」。draw は千日手が特徴量に"
       "見えない問題があるため、判定は主に win/lose と先手劣勢セルで見る。")
    md()

    # ---- 問い2: CN 層別 ----
    md("## 問い2. CN 層別の s 正解率（注釈の妥当性）")
    md()
    by_bucket = defaultdict(list)
    for r in cn_rows:
        by_bucket[cn_bucket(r[3])].append(r)
    md("| CN | 件数 | 割合 | s 正解率 | win | lose | draw |")
    md("|---|---|---|---|---|---|---|")
    order = ["1", "2", "3", "4", "5+", "∞", "なし"]
    accs = {}
    for b in order:
        sub = by_bucket.get(b)
        if not sub:
            continue
        acc = pct(sum(1 for r in sub if r[1] == r[0]), len(sub))
        accs[b] = acc
        cat_strs = []
        for lv in (1, -1, 0):
            csub = [r for r in sub if r[0] == lv]
            cat_strs.append(f"{pct(sum(1 for r in csub if r[1] == lv), len(csub)):.1f}%（{len(csub):,}）" if csub else "—")
        md(f"| {b} | {len(sub):,} | {pct(len(sub), len(cn_rows)):.1f}% | {acc:.1f}% | {cat_strs[0]} | {cat_strs[1]} | {cat_strs[2]} |")
    md()
    md("カテゴリ別の列はバケット内の構成比の交絡を切り分けるためのもの"
       "（例: 特定の CN に draw 局面が寄っていれば、その段の正解率は構造的に下がる）。")
    md()
    # 正解率の低いセルについて、s が実際には何と予測したかの内訳
    low_cells = []
    for b in order:
        sub = by_bucket.get(b)
        if not sub:
            continue
        for lv, lname in ((1, "win"), (-1, "lose"), (0, "draw")):
            csub = [r for r in sub if r[0] == lv]
            if len(csub) < 100:
                continue
            acc_c = pct(sum(1 for r in csub if r[1] == lv), len(csub))
            if acc_c < 90.0:
                dist = Counter(r[1] for r in csub)
                low_cells.append((b, lname, len(csub), acc_c, dist))
    if low_cells:
        md("### 正解率 90% 未満のセルの s 予測内訳")
        md()
        for b, lname, n, acc_c, dist in low_cells:
            md(f"- CN={b} × {lname}（{n:,} 件、正解率 {acc_c:.1f}%）: "
               f"s=win {dist.get(1, 0):,} / s=draw {dist.get(0, 0):,} / s=lose {dist.get(-1, 0):,}")
        md()
    finite = [accs[b] for b in ("1", "2", "3", "4", "5+") if b in accs]
    if len(finite) >= 3:
        mono = all(finite[i] <= finite[i + 1] + 3.0 for i in range(len(finite) - 1))
        md(f"- 有限 CN の正解率の並び: {' → '.join(f'{a:.1f}%' for a in finite)}"
           f"（単調に近い: {'はい' if mono else 'いいえ'}、±3pt の揺れは許容）")
    if "∞" in accs:
        md(f"- CN ∞ の正解率: {accs['∞']:.1f}%（証明済み級なので最高水準のはず）")
    md()
    md("読み方: CN が大きいほど s が当たるなら、CN は教師の信頼度として機能しており、"
       "学習の重み付けや生成時の選別に使える。差が無ければ CN は木の形を測っているだけ。")
    md()

    # ---- 問い2b: CN と維持手数 ----
    md("## 問い2b. CN と維持手数（オラクル定義）の関係")
    md()
    from game import from_hash
    rng = np.random.default_rng(123)
    cand = [e for e in cn_entries if "c" in e]
    if len(cand) > args.preserving_sample:
        idx = rng.choice(len(cand), args.preserving_sample, replace=False)
        cand = [cand[i] for i in idx]
    pairs = []  # (cn, preserving)
    skipped = 0
    for e in cand:
        state = from_hash(int(e["p"]))
        if state.pos.is_win() or state.pos.is_lose():
            skipped += 1
            continue
        actions = legal_moves_list(state)
        if not actions:
            skipped += 1
            continue
        outcomes = [child_outcome(oracle, state.next(a)) for a in actions]
        if any(o is None for o in outcomes):
            skipped += 1
            continue
        best = max(outcomes)
        preserving = int(sum(1 for o in outcomes if o == best))
        pairs.append((int(e["c"]), preserving))
    md(f"- 調査できた局面: {len(pairs):,}（スキップ {skipped:,}）")
    if pairs:
        by_b = defaultdict(list)
        for c, p in pairs:
            by_b[cn_bucket(c)].append(p)
        md("| CN | 件数 | 維持手数の平均 |")
        md("|---|---|---|")
        for b in order:
            if b in by_b:
                md(f"| {b} | {len(by_b[b]):,} | {float(np.mean(by_b[b])):.2f} |")
        finite_pairs = [(c, p) for c, p in pairs if c > 0]
        if len(finite_pairs) > 30:
            rho = spearman([c for c, _ in finite_pairs], [p for _, p in finite_pairs])
            md(f"- 有限 CN と維持手数の Spearman 相関: {rho:+.3f}")
        md()
        md("読み方: 相関が正で、∞ の維持手数が最大なら、CN はオラクル定義の脆弱性を"
           "木内部の量で再現できている。")
    md()

    # ---- 問い3: 相互検証 ----
    md("## 問い3. 相互検証（c=-1 は格子一致か）")
    md()
    inf_rows = [e for e in cn_entries if e.get("c") == -1]
    big = [e for e in inf_rows if abs(float(e["s"])) > 0.5]
    if big:
        lat = sum(1 for e in big if (lambda k: k is not None and k != 200)(lattice_k(abs(float(e["s"])))))
        md(f"- c=-1 かつ |s|>0.5: {len(big):,} 件中、格子一致（k≠200）は {lat:,} 件（{pct(lat, len(big)):.2f}%）")
        md("- 100% 近くなければ、C++ 側の CN 計算か注釈の受け渡しに不整合がある")
    zeros = [e for e in inf_rows if float(e["s"]) == 0.0]
    if zeros:
        agree0 = 0
        checked0 = 0
        for e in zeros[:5000]:
            lab = oracle_label_turn(oracle, int(e["p"]))
            if lab is not None:
                checked0 += 1
                agree0 += 1 if lab == 0 else 0
        if checked0:
            md(f"- c=-1 かつ s=0: {len(zeros):,} 件（うち照合 {checked0:,} 件でオラクル draw 一致 {pct(agree0, checked0):.2f}%）")
    md()

    # ---- 判定 ----
    md("## 判定")
    md()
    md("- 問い1で CN 側が win/lose ともに明確に高い（+2pt 以上）→ 葉選択の CN 化は生成レバーになる。段階B の候補として B1 と比較する")
    md("- 問い1が互角でも、問い2で CN 層別の勾配があれば、CN は注釈（信頼度）として採用できる")
    md("- 問い1で CN 側が低く、問い2にも勾配が無ければ、この路線は下げる")
    md("- 問い3が崩れていたら、まず C++ 実装を疑う（Python ブルートフォースで再帰自体は検証済み）")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"\n結果を {args.out} に書きました")


if __name__ == "__main__":
    main()
