# 未登録原因の第2調査: DBの正規化（手番・鏡像・回転）を確定する
# ============================================================
# 第1調査の結果:
#   - 後手番局面の登録率は 0.0%（本体・対照Aとも）。先手番は 55〜62%
#   - oracle/main.cpp の conv_state() は手番を BLACK2 に固定して win.dat 等を生成
#   - oracle/dobutsu.h の State は pack 表現が常に先手番前提
#     （State(p, WHITE) は makeBlackFromUint64(p).rotateChangeTurn() で復元する）
#   → DBのキーは「先手番に正規化された局面」だけのはず。本スクリプトは
#     後手番局面を 180度回転+色交換して先手番化したキーで照合できるか、
#     先手番の残り約4割が鏡像（左右反転）の正規化で説明できるかを検証する。
#
# さらに、回転照合で得たラベルを使って実験4.5-C の教師精度表を
# 手番別に再計算し、正規化の解釈が正しければ精度が回復するかを見る。
#
# 実行（learn/ ディレクトリで）:
#   ../.venv/bin/python check_unregistered2.py
# 出力: check_unregistered2_result.md

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from check_unregistered import (Oracle, decode_key, encode_key, self_test,
                                B_H, B_K, B_Z, B_L, B_N, W_H, W_K, W_Z, W_L, W_N, EMPTY)

SWAP = {B_H: W_H, W_H: B_H, B_K: W_K, W_K: B_K, B_Z: W_Z, W_Z: B_Z,
        B_L: W_L, W_L: B_L, B_N: W_N, W_N: B_N, EMPTY: EMPTY}

md_lines = []


def md(line=""):
    md_lines.append(line)
    print(line)


def pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


def classify(score, threshold=0.2):
    if score > threshold:
        return 1
    if score < -threshold:
        return -1
    return 0


def mirror_key(key):
    """左右鏡像: 各段の中で file を反転（index = row*3 + f → row*3 + (2-f)）。
    色・持駒・手番は不変"""
    turn_is_white, pieces, hand = decode_key(key)
    mirrored = [EMPTY] * 12
    for idx, code in enumerate(pieces):
        row, f = divmod(idx, 3)
        mirrored[row * 3 + (2 - f)] = code
    return encode_key(turn_is_white, mirrored, hand)


def rotate_to_black_key(key):
    """後手番局面を、先手番に正規化した等価局面のキーに変換する。
    180度回転（index → 11-index）+ 色交換 + 持駒の色交換 + 手番を先手に。
    ゲーム状況としては等価で、「手番側の勝ち/負け/引き分け」は保存される"""
    turn_is_white, pieces, hand = decode_key(key)
    rotated = [EMPTY] * 12
    for idx, code in enumerate(pieces):
        rotated[11 - idx] = SWAP[code]
    new_hand = {}
    for (color, kind), n in hand.items():
        new_hand[("W" if color == "B" else "B", kind)] = n
    return encode_key(False, rotated, new_hand)


class LabelOracle(Oracle):
    def label(self, key):
        k = np.uint64(key)
        for name, value in (("win", 1), ("lose", -1), ("draw", 0)):
            arr = self.arrays[name]
            idx = np.searchsorted(arr, k)
            if idx < len(arr) and arr[idx] == k:
                return value
        return None


def lookup_variants(oracle, key):
    """(直接ラベル, 鏡像ラベル, 正規化ラベル) を返す。ラベルは手番側視点。
    正規化ラベル: 先手番は直接と同じ、後手番は回転正規化キーで引く"""
    turn_is_white, _, _ = decode_key(key)
    direct = oracle.label(key)
    mirror = oracle.label(mirror_key(key))
    if turn_is_white:
        normalized = oracle.label(rotate_to_black_key(key))
        normalized_mirror = oracle.label(mirror_key(rotate_to_black_key(key)))
    else:
        normalized = direct
        normalized_mirror = mirror
    return direct, mirror, normalized, normalized_mirror


def main():
    parser = argparse.ArgumentParser(description="未登録原因の第2調査")
    parser.add_argument("--out", default="check_unregistered2_result.md")
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--group", default="data")
    args = parser.parse_args()

    md("# 未登録原因の第2調査（check_unregistered2.py）")
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
    rng = np.random.default_rng(123)  # C・第1調査と同一サンプル
    if len(entries) > args.sample:
        idx = rng.choice(len(entries), args.sample, replace=False)
        entries = [entries[i] for i in idx]
    md(f"対象: グループ `{args.group}` から {len(entries):,} 件（Cと同一サンプル）")
    md()

    # ---- カバレッジ: 手番別 × 照合方法別 ----
    stats = {}
    teacher_rows = []  # (label, model対象外) C再計算用: (label, search_pred, result_pred, turn_is_white)
    from game import from_hash
    for e in entries:
        key = int(e["p"])
        turn_is_white, _, _ = decode_key(key)
        state = from_hash(key)
        if state.pos.is_win() or state.pos.is_lose():
            continue
        direct, mirror, normalized, normalized_mirror = lookup_variants(oracle, key)
        bucket = stats.setdefault(turn_is_white, {"n": 0, "direct": 0, "mirror_any": 0,
                                                  "norm": 0, "norm_mirror_any": 0})
        bucket["n"] += 1
        if direct is not None:
            bucket["direct"] += 1
        if direct is not None or mirror is not None:
            bucket["mirror_any"] += 1
        label = normalized if normalized is not None else normalized_mirror
        if normalized is not None:
            bucket["norm"] += 1
        if label is not None:
            bucket["norm_mirror_any"] += 1
            r = e["r"]
            teacher_rows.append((label, classify(float(e["s"])),
                                 int(r) if r in (-1, 0, 1) else None, turn_is_white))

    md("## 照合方法別の登録率")
    md()
    md("| 手番 | 件数 | 直接 | 直接+鏡像 | 正規化 | 正規化+鏡像 |")
    md("|---|---|---|---|---|---|")
    for tw, name in ((False, "先手番"), (True, "後手番")):
        b = stats.get(tw, {"n": 0, "direct": 0, "mirror_any": 0, "norm": 0, "norm_mirror_any": 0})
        n = b["n"]
        md(f"| {name} | {n:,} | {pct(b['direct'], n):.1f}% | {pct(b['mirror_any'], n):.1f}% | "
           f"{pct(b['norm'], n):.1f}% | {pct(b['norm_mirror_any'], n):.1f}% |")
    tot_n = sum(b["n"] for b in stats.values())
    tot_nm = sum(b["norm_mirror_any"] for b in stats.values())
    md()
    md(f"**正規化+鏡像での全体カバレッジ: {tot_nm:,} / {tot_n:,}（{pct(tot_nm, tot_n):.1f}%）**")
    md()

    # ---- 教師精度の再計算（正規化ラベル使用、手番別） ----
    md("## 教師精度の再計算（正規化ラベル、手番別）")
    md()
    md("実験4.5-C の表は直接照合のみ（=先手番局面だけが集計対象）だった。ここでは後手番も回転正規化で取り込み、手番別に search_score / game_result の正解率を出す。ラベルは手番側視点（win=+1）。")
    md()
    for tw, name in ((False, "先手番"), (True, "後手番"), (None, "合算")):
        sub = [r for r in teacher_rows if tw is None or r[3] == tw]
        if not sub:
            continue
        md(f"### {name}（{len(sub):,} 件）")
        md()
        md("| カテゴリ | 件数 | search_score正解率 | game_result正解率 |")
        md("|---|---|---|---|")
        for lv, lname in ((1, "win"), (-1, "lose"), (0, "draw")):
            ss = [r for r in sub if r[0] == lv]
            if not ss:
                continue
            s_acc = pct(sum(1 for r in ss if r[1] == lv), len(ss))
            gr = [r for r in ss if r[2] is not None]
            g_acc = pct(sum(1 for r in gr if r[2] == lv), len(gr)) if gr else float("nan")
            md(f"| {lname} | {len(ss):,} | {s_acc:.1f}% | {g_acc:.1f}% |")
        md()

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"\n結果を {args.out} に書きました")


if __name__ == "__main__":
    main()
