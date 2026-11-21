# 順序の調査（audit_order.py）
# ============================
# 問い: モデルの値の「順序」は、実戦で比較される兄弟手同士では正しいのか。
# ans（problem.txt）は値の分類精度しか測らない。圧縮があっても順序が正しければ
# 指し手は変わらないはず、という主張を直接検証する。
#
# 方法: 自己対局レコードから対局順の局面列を復元し、各局面で全合法手の子局面を列挙して
#   - 子局面のオラクルカテゴリ（正規化照合）→ 指した側視点の真の結果（win/draw/lose）
#   - 子局面のモデル値（手番視点なので反転して指した側視点に）
# を並べ、次を測る:
#   A. モデル値の argmax 手が、その局面で達成可能な最良カテゴリに届いている率
#   B. カテゴリの異なる兄弟ペアについて、モデル値の順序が真の順序と一致する率
#   C. 実際に指された手（レコードの次の局面）のカテゴリ維持/低下
#      ※自己対局の手選択には約10%のランダム手が混ざる（ai/selfplay.hpp）ため、
#        C は探索+探索ノイズ込みの数値として読むこと。A/B はモデル値のみで測る
#
# 終局手（ライオン取り等）でDBに無い子局面は、Position の is_lose/is_win で判定する
# （子局面で手番側が is_lose なら指した側の勝ち、is_win なら負け）。
#
# 実行（learn/ ディレクトリで）:
#   ../.venv/bin/python audit_order.py
# 出力: audit_order_result.md
#
# 注意: gamelibs の MoveList は __getitem__ に境界チェックが無く、for で回すと
# 範囲外を読み続けて segfault する。必ず len() + インデックスでアクセスすること
# （game.py の利用側も同様）。

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


CAT_NAME = {1: "win", 0: "draw", -1: "lose"}


def legal_moves_list(state):
    """MoveList を安全にリスト化する（イテレーションは segfault するのでインデックスで）"""
    ml = state.legal_actions()
    return [ml[i] for i in range(len(ml))]


def child_outcome(oracle, child_state):
    """子局面について、指した側視点の真の結果（+1/0/-1）。不明なら None"""
    key = child_state.hash_key()
    direct, mirror, normalized, normalized_mirror = lookup_variants(oracle, key)
    label = normalized if normalized is not None else normalized_mirror
    if label is not None:
        return -label  # ラベルは子局面の手番側（=相手）視点
    if child_state.is_lose():
        return 1
    if child_state.is_win():
        return -1
    return None


def main():
    parser = argparse.ArgumentParser(description="順序の調査")
    parser.add_argument("--model", default="./model/best_single.h5")
    parser.add_argument("--out", default="audit_order_result.md")
    parser.add_argument("--sample", type=int, default=3000, help="調査する局面数")
    parser.add_argument("--group", default="data")
    args = parser.parse_args()

    md("# 順序の調査（audit_order.py）")
    md()
    if not self_test():
        md("**self-test 失敗のため中止**")
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md_lines) + "\n")
        return

    oracle = LabelOracle()

    # ---- 対局列の復元（連続エントリが1手でつながる間を同一対局とする） ----
    from game import from_hash

    files = sorted(glob.glob("data/**/*.json", recursive=True))
    positions = []  # (key, played_child_key or None)
    games = 0
    transitions = 0
    for jf in files:
        if Path(jf).parent.name != args.group:
            continue
        with open(jf) as fh:
            try:
                entries = json.load(fh)
            except json.JSONDecodeError:
                continue
        prev_state = None
        prev_key = None
        for e in entries:
            if len(positions) >= 60000:  # 復元は重いので上限を設ける（サンプルは後で抽く）
                break
            key = int(e["p"])
            if prev_state is not None:
                child_keys = set()
                for a in legal_moves_list(prev_state):
                    child_keys.add(prev_state.next(a).hash_key())
                if key in child_keys:
                    positions[-1] = (prev_key, key)
                    transitions += 1
                else:
                    games += 1
            else:
                games += 1
            positions.append((key, None))
            prev_state = from_hash(key)
            prev_key = key
    md(f"- 復元した対局数（推定）: {games:,} / 局面数: {len(positions):,} / 手の接続を確認できた遷移: {transitions:,}")
    md()

    rng = np.random.default_rng(123)
    if len(positions) > args.sample:
        idx = rng.choice(len(positions), args.sample, replace=False)
        positions = [positions[i] for i in idx]

    # ---- モデル準備 ----
    import torch
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
            feats = np.array([np.array(s.feature()).reshape(channel, file, rank) for s in batch])
            x = torch.from_numpy(feats).to(device).float()
            with torch.no_grad():
                out.append(model(x).squeeze(-1).cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    # ---- 局面ごとの兄弟評価 ----
    records = []  # dict: turn_white, pos_cat, best, argmax_ok, pairs(list), played_outcome
    skipped = 0
    all_children = []
    metas = []
    for key, played_key in positions:
        state = from_hash(key)
        if state.is_win() or state.is_lose():
            skipped += 1
            continue
        actions = legal_moves_list(state)
        if len(actions) == 0:
            skipped += 1
            continue
        children = [state.next(a) for a in actions]
        outcomes = [child_outcome(oracle, c) for c in children]
        if any(o is None for o in outcomes):
            skipped += 1
            continue
        tw, _, _ = decode_key(key)
        metas.append((key, played_key, tw, children, outcomes))
        all_children.extend(children)
    values_flat = batch_predict(all_children) if all_children else np.array([])
    ofs = 0
    for key, played_key, tw, children, outcomes in metas:
        n = len(children)
        child_vals = values_flat[ofs:ofs + n]
        ofs += n
        mover_vals = -child_vals  # 手番視点 → 指す側視点
        outcomes = np.array(outcomes)
        best = outcomes.max()
        argmax_idx = int(np.argmax(mover_vals))
        pairs = []
        for i in range(n):
            for j in range(n):
                if outcomes[i] > outcomes[j]:
                    if mover_vals[i] > mover_vals[j]:
                        pairs.append(1.0)
                    elif mover_vals[i] == mover_vals[j]:
                        pairs.append(0.5)
                    else:
                        pairs.append(0.0)
        played_outcome = None
        if played_key is not None:
            for c, o in zip(children, outcomes):
                if c.hash_key() == played_key:
                    played_outcome = int(o)
                    break
        # 先手視点のカテゴリ（手番側カテゴリを先手基準に直す）
        pos_cat_turn = int(best)  # 局面のカテゴリは最良手の結果と一致するはず（手番側視点）
        records.append({
            "turn_white": tw,
            "pos_cat": pos_cat_turn,
            "sente_losing": (pos_cat_turn == -1 and not tw) or (pos_cat_turn == 1 and tw),
            "argmax_ok": outcomes[argmax_idx] == best,
            "pairs": pairs,
            "played_outcome": played_outcome,
            "best": int(best),
        })
    md(f"- 調査できた局面: {len(records):,}（スキップ {skipped:,}）")
    md()

    def summarize(sub, title):
        if not sub:
            return
        md(f"### {title}（{len(sub):,} 局面）")
        md()
        ok = sum(1 for r in sub if r["argmax_ok"])
        md(f"- A. モデル値 argmax が最良カテゴリに到達: **{pct(ok, len(sub)):.1f}%**")
        all_pairs = [p for r in sub for p in r["pairs"]]
        if all_pairs:
            md(f"- B. 兄弟ペアの順序一致率: **{100.0 * np.mean(all_pairs):.1f}%**（{len(all_pairs):,} ペア）")
        played = [r for r in sub if r["played_outcome"] is not None]
        if played:
            keep = sum(1 for r in played if r["played_outcome"] == r["best"])
            drop1 = sum(1 for r in played if r["best"] - r["played_outcome"] == 1)
            drop2 = sum(1 for r in played if r["best"] - r["played_outcome"] == 2)
            md(f"- C. 実際に指された手: 最良カテゴリ維持 **{pct(keep, len(played)):.1f}%** / "
               f"1段階低下 {pct(drop1, len(played)):.1f}% / 2段階低下 {pct(drop2, len(played)):.1f}%（{len(played):,} 手、約10%のランダム手を含む）")
        md()

    md("## 全体")
    md()
    summarize(records, "全局面")

    md("## 局面カテゴリ別（手番側視点）")
    md()
    for cat in (1, 0, -1):
        summarize([r for r in records if r["pos_cat"] == cat], f"{CAT_NAME[cat]} 局面")

    md("## 先手劣勢 vs それ以外")
    md()
    summarize([r for r in records if r["sente_losing"]], "先手が負けている局面")
    summarize([r for r in records if not r["sente_losing"]], "それ以外")

    md("## 手番別")
    md()
    summarize([r for r in records if not r["turn_white"]], "先手番")
    summarize([r for r in records if r["turn_white"]], "後手番")

    md("## 読み方")
    md()
    md("- A/B が先手劣勢でも他と同水準なら、圧縮は指し手に無関係（順序は保たれている）")
    md("- A/B が先手劣勢で明確に落ちるなら、圧縮は境界比較を通じて手を悪くしている")
    md("- C は A より悪くて当然（探索の有限予算＋約10%のランダム手）。C と A の差が探索側の損失の目安")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"\n結果を {args.out} に書きました")


if __name__ == "__main__":
    main()
