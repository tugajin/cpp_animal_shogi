# 未登録68%の原因調査スクリプト
# ==============================
# 実験4.5-C で、自己対局局面サンプルの 68.1% がオラクルDB（win/lose/draw.dat）に
# 登録されていないことが分かった。その理由を切り分ける。
#
# 実行（learn/ ディレクトリで）:
#   cd ~/Documents/cpp_animal_shogi/learn
#   ../.venv/bin/python check_unregistered.py
#
# 出力: check_unregistered_result.md
#
# 調査の柱:
#   0. ハッシュ decode/encode の自己検証（平手キーで照合）
#   1. 本体: C と同じサンプル（data グループ、seed=123、20,000件）について
#      登録/未登録を特徴量で層別する
#   2. 対照A: 平手からランダム合法プレイアウトで作った局面の登録率
#      （DBとハッシュが正常ならほぼ100%になるはず）
#   3. 対照B: C++ の generate_random_position() を Python で近似再現した
#      ランダム配置局面の登録率（>1 attacker の棄却は gamelibs で再現できないため省略し、
#      王手特徴量で層別して補う）
#   4. 変形キーの照合: 未登録キーについて手番ビット反転キー（key ^ 1）が登録されていないか
#
# 特徴量（decode は ai/hash.hpp の from_hash / hash_key の仕様に基づく）:
#   - 手番（先手/後手）
#   - 手番側ライオンに王手がかかっているか（合法であり得る）
#   - 非手番側ライオンに王手がかかっているか（静的に違法: 相手の王手を放置して手番が回ることはない）
#   - 両ライオン同時王手（静的に違法）
#   - 先手ライオンが4段目（トライゾーン）/ 後手ライオンが1段目（トライゾーン）
#   - 先手ひよこが4段目 / 後手ひよこが1段目（成らずに到達できないマス）
#   - 駒種ごとの総数（盤上+持駒）が保存則（ライオン2・きりん2・ぞう2・ひよこ系2）から外れているか

import argparse
import glob
import json
import os
import random
from pathlib import Path

import numpy as np

ORACLE_DIR = "../data"

# hash の駒コード（ai/common.hpp の color_piece_no 割当: EMPTY=1 から連番）
EMPTY, B_H, B_K, B_Z, B_L, B_N, W_H, W_K, W_Z, W_L, W_N = range(1, 12)
PIECE_NAME = {EMPTY: "空", B_H: "先ひ", B_K: "先き", B_Z: "先ぞ", B_L: "先ら", B_N: "先に",
              W_H: "後ひ", W_K: "後き", W_Z: "後ぞ", W_L: "後ら", W_N: "後に"}
START_HASH_KEY = 1248110771415179264

md_lines = []


def md(line=""):
    md_lines.append(line)
    print(line)


def pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


# ----------------------------------------------------------------------
# ハッシュ decode / encode（ai/hash.hpp と対）
# ----------------------------------------------------------------------

def decode_key(key):
    """key -> (turn_is_white, pieces[12], hand) hand は {(color, kind): n}
    kind は 'H'（ひよこ）/'K'（きりん）/'Z'（ぞう）。手番ビットが最下位、
    続いて持駒カウント（2bitずつ）が積まれ、最後に盤面12マス（4bitずつ）。
    from_hash の読み順: 手番 → 後手ぞう/きりん/ひよこ → 先手ぞう/きりん/ひよこ → 盤面（index 11→0）"""
    k = int(key)
    turn_is_white = bool(k & 1)
    k >>= 1
    hand = {}
    for color in ("W", "B"):
        for kind in ("Z", "K", "H"):
            hand[(color, kind)] = k & 0x3
            k >>= 2
    # 注: from_hash は 後手(Z,K,H) → 先手(Z,K,H) の順に読む（hash_key の書き順の逆積み）
    pieces = [EMPTY] * 12
    for i in range(11, -1, -1):
        pieces[i] = k & 0xF
        k >>= 4
    return turn_is_white, pieces, hand


def encode_key(turn_is_white, pieces, hand):
    """hash_key の再現。hand は decode_key と同じ dict 形式"""
    key = 0
    for code in pieces:  # SQUARE_INDEX 順（index 0..11）
        key += code
        key <<= 4
    key >>= 4
    # hash_key の書き順: 先手 H,K,Z → 後手 H,K,Z → 手番
    for color in ("B", "W"):
        for kind in ("H", "K", "Z"):
            key = (key << 2) + hand[(color, kind)]
    key = (key << 1) + (1 if turn_is_white else 0)
    return key


def self_test():
    """平手キーで decode/encode を検証する。失敗したら以降の段・駒特徴量は信用できない"""
    turn_is_white, pieces, hand = decode_key(START_HASH_KEY)
    ok = True
    # 平手: 先手番、持駒なし。decode 実測（2026-11-19 検証）では
    #   index 0..2 が後手陣（最上段）、9..11 が先手陣（最下段）で、
    #   後手ライオン index1・先手ライオン index10、後手ひよこ index4・先手ひよこ index7
    expect = {10: B_L, 1: W_L, 7: B_H, 4: W_H}
    for idx, code in expect.items():
        if pieces[idx] != code:
            print(f"self-test NG: pieces[{idx}]={pieces[idx]} != {code}")
            ok = False
    if turn_is_white or any(hand.values()):
        print("self-test NG: 平手は先手番・持駒なしのはず")
        ok = False
    if encode_key(turn_is_white, pieces, hand) != START_HASH_KEY:
        print("self-test NG: encode(decode(平手)) != 平手キー")
        ok = False
    print(f"self-test: {'OK' if ok else 'NG（特徴量の解釈を中止すること）'}")
    return ok


# ----------------------------------------------------------------------
# オラクル照合
# ----------------------------------------------------------------------

class Oracle:
    def __init__(self, base_dir=ORACLE_DIR):
        self.arrays = {}
        for name in ("win", "lose", "draw"):
            arr = np.fromfile(os.path.join(base_dir, f"{name}.dat"), dtype=np.uint64)
            arr.sort()
            self.arrays[name] = arr

    def registered(self, key):
        k = np.uint64(key)
        for arr in self.arrays.values():
            idx = np.searchsorted(arr, k)
            if idx < len(arr) and arr[idx] == k:
                return True
        return False


# ----------------------------------------------------------------------
# 特徴量
# ----------------------------------------------------------------------

def features_of(key):
    """decode ベースの静的特徴量。王手判定は呼び出し側（gamelibs）で付ける"""
    turn_is_white, pieces, hand = decode_key(key)
    row = lambda idx: idx // 3  # 0=後手陣（最上段）… 3=先手陣（最下段）。平手 decode 実測で確認
    f = {}
    f["turn_is_white"] = turn_is_white
    f["black_lion_try"] = any(pieces[i] == B_L and row(i) == 0 for i in range(12))
    f["white_lion_try"] = any(pieces[i] == W_L and row(i) == 3 for i in range(12))
    f["black_hiyoko_far"] = any(pieces[i] == B_H and row(i) == 0 for i in range(12))
    f["white_hiyoko_far"] = any(pieces[i] == W_H and row(i) == 3 for i in range(12))
    # 駒種ごとの総数（盤上 + 持駒）。ひよこ系は ひよこ+にわとり で数える
    totals = {
        "lion": sum(1 for p in pieces if p in (B_L, W_L)),
        "kirin": sum(1 for p in pieces if p in (B_K, W_K)) + hand[("B", "K")] + hand[("W", "K")],
        "zou": sum(1 for p in pieces if p in (B_Z, W_Z)) + hand[("B", "Z")] + hand[("W", "Z")],
        "hiyoko_line": sum(1 for p in pieces if p in (B_H, W_H, B_N, W_N)) + hand[("B", "H")] + hand[("W", "H")],
    }
    f["conservation_broken"] = any(totals[k] != 2 for k in totals)
    f["hand_total"] = sum(hand.values())
    f["board_pieces"] = sum(1 for p in pieces if p != EMPTY)
    return f


def add_check_features(key, f):
    """gamelibs で王手判定を付ける。手番反転キーで非手番側の王手を見る"""
    from game import from_hash
    state = from_hash(int(key))
    f["turn_in_check"] = bool(state.in_checked())
    flipped = from_hash(int(key) ^ 1)
    f["other_in_check"] = bool(flipped.in_checked())
    f["both_in_check"] = f["turn_in_check"] and f["other_in_check"]
    return f


ILLEGAL_FEATURES = ["other_in_check", "both_in_check", "black_hiyoko_far", "white_hiyoko_far",
                    "conservation_broken"]


def rate_table(rows, title):
    """rows: (registered, features) のリスト。特徴量ごとの登録率を出す"""
    md(f"### {title}")
    md()
    n = len(rows)
    reg = sum(1 for r, _ in rows if r)
    md(f"- 全体: {n:,} 件、登録 {reg:,} 件（{pct(reg, n):.1f}%）")
    md()
    md("| 特徴量 | 該当件数 | 該当の登録率 | 非該当の登録率 |")
    md("|---|---|---|---|")
    keys = ["turn_is_white", "turn_in_check", "other_in_check", "both_in_check",
            "black_lion_try", "white_lion_try", "black_hiyoko_far", "white_hiyoko_far",
            "conservation_broken"]
    for k in keys:
        yes = [r for r, f in rows if f.get(k)]
        no = [r for r, f in rows if not f.get(k)]
        if not yes:
            continue
        yr = pct(sum(yes), len(yes))
        nr = pct(sum(no), len(no)) if no else float("nan")
        md(f"| {k} | {len(yes):,} | {yr:.1f}% | {nr:.1f}% |")
    illegal = [r for r, f in rows if any(f.get(k) for k in ILLEGAL_FEATURES)]
    legal = [r for r, f in rows if not any(f.get(k) for k in ILLEGAL_FEATURES)]
    md(f"| **違法系特徴のいずれか** | {len(illegal):,} | {pct(sum(illegal), len(illegal)):.1f}% | {pct(sum(legal), len(legal)):.1f}%（非該当側） |")
    md()
    return reg, n


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="未登録68%の原因調査")
    parser.add_argument("--out", default="check_unregistered_result.md")
    parser.add_argument("--sample", type=int, default=20000, help="本体サンプル上限（Cと同条件）")
    parser.add_argument("--group", default="data", help="本体の対象グループ（Cと同じ data）")
    parser.add_argument("--control", type=int, default=2000, help="対照A/B の局面数")
    args = parser.parse_args()

    md("# 未登録局面の原因調査（check_unregistered.py）")
    md()

    if not self_test():
        md("**self-test が失敗したため中止。decode の対応付けが想定と違う。**")
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md_lines) + "\n")
        return

    oracle = Oracle()

    # ---- 本体: C と同じサンプル ----
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
    md(f"## 本体: 自己対局データ（グループ `{args.group}`、{len(entries):,} 件サンプル、Cと同条件）")
    md()
    rows = []
    flip_hits = 0
    unreg = 0
    for e in entries:
        key = int(e["p"])
        reg = oracle.registered(key)
        f = add_check_features(key, features_of(key))
        rows.append((reg, f))
        if not reg:
            unreg += 1
            if oracle.registered(key ^ 1):
                flip_hits += 1
    rate_table(rows, "自己対局サンプルの層別")
    md(f"- 未登録 {unreg:,} 件のうち、手番反転キー（key ^ 1）が登録されていたもの: {flip_hits} 件")
    md()

    # ---- グループ別の登録率（簡易版: 特徴量なし） ----
    md("## グループ別の登録率")
    md()
    md("| グループ | サンプル数 | 登録率 |")
    md("|---|---|---|")
    by_group = {}
    for jf in files:
        by_group.setdefault(Path(jf).parent.name, []).append(jf)
    rng2 = np.random.default_rng(7)
    for name in sorted(by_group):
        pool = []
        for jf in by_group[name]:
            with open(jf) as fh:
                try:
                    pool.extend(json.load(fh))
                except json.JSONDecodeError:
                    continue
        if len(pool) > 2000:
            idx = rng2.choice(len(pool), 2000, replace=False)
            pool = [pool[i] for i in idx]
        hits = sum(1 for e in pool if oracle.registered(int(e["p"])))
        md(f"| {name} | {len(pool):,} | {pct(hits, len(pool)):.1f}% |")
    md()

    # ---- 対照A: 平手からのランダム合法プレイアウト ----
    md("## 対照A: 平手からのランダム合法プレイアウト")
    md()
    from game import State
    rnd = random.Random(42)
    rows_a = []
    tries = 0
    while len(rows_a) < args.control and tries < args.control * 20:
        tries += 1
        state = State()
        depth = rnd.randint(2, 40)
        for _ in range(depth):
            if state.is_done():
                break
            actions = state.legal_actions()
            if len(actions) == 0:
                break
            state = state.next(actions[rnd.randint(0, len(actions) - 1)])
        if state.is_done() or state.is_win() or state.is_lose():
            continue
        key = state.hash_key()
        f = add_check_features(key, features_of(key))
        rows_a.append((oracle.registered(key), f))
    rate_table(rows_a, "プレイアウト局面の層別")

    # ---- 対照B: generate_random_position の近似再現 ----
    md("## 対照B: ランダム初期局面の近似再現")
    md()
    md("C++ の `generate_random_position()` を再現（ライオン2枚をランダム配置、残り6枚〔ひよこ/きりん/ぞう×2〕は色・配置・持駒をランダム、盤上のひよこは50%でにわとり化、手番は先手固定）。")
    md("再現できていない棄却条件: 「どちらかのライオンへの attacker 数 > 1」の棄却（gamelibs に attacker 数がないため）。代わりに王手特徴量で層別する。is_win/is_lose の棄却は適用。")
    md()
    from game import from_hash
    rows_b = []
    tries = 0
    while len(rows_b) < args.control and tries < args.control * 30:
        tries += 1
        pieces = [EMPTY] * 12
        hand = {(c, k): 0 for c in ("B", "W") for k in ("H", "K", "Z")}
        wl = rnd.randrange(12)
        pieces[wl] = W_L
        while True:
            bl = rnd.randrange(12)
            if pieces[bl] == EMPTY:
                pieces[bl] = B_L
                break
        for base in (B_H, B_K, B_Z, B_H, B_K, B_Z):
            while True:
                index = rnd.randrange(13)  # 12 は持駒
                color_is_white = rnd.random() < 0.5
                if base == B_H:
                    code_board = (W_H if color_is_white else B_H)
                    code_niwa = (W_N if color_is_white else B_N)
                    kind = "H"
                elif base == B_K:
                    code_board = (W_K if color_is_white else B_K)
                    kind = "K"
                else:
                    code_board = (W_Z if color_is_white else B_Z)
                    kind = "Z"
                if index == 12:
                    hand[("W" if color_is_white else "B", kind)] += 1
                    break
                if pieces[index] == EMPTY:
                    if base != B_H:
                        pieces[index] = code_board
                    elif rnd.random() < 0.5:
                        pieces[index] = code_board
                    else:
                        pieces[index] = code_niwa
                    break
        key = encode_key(False, pieces, hand)
        state = from_hash(key)
        if state.is_win() or state.is_lose():
            continue
        f = add_check_features(key, features_of(key))
        rows_b.append((oracle.registered(key), f))
    rate_table(rows_b, "ランダム配置局面の層別")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"\n結果を {args.out} に書きました")


if __name__ == "__main__":
    main()
