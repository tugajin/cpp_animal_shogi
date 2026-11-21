# フェーズ0 再集計（phase0_analysis.py）
# ======================================
# 新規の生成・学習は一切しない。既存の自己対局レコードと現行モデルから、
# 実験5の設計判断に必要な3点を切り分ける。
#
#   P1. 圧縮の分解: 「先手劣勢で値が 0 側に縮む」のは、混合教師 t=(s+r)/2 の
#       条件付き平均そのものが小さいからか（教師の算術）、それともモデルが
#       教師の平均よりさらに縮んでいるのか（学習側の shrinkage）。
#       重複レコードのある局面について、局面ごとの t の実測平均とモデルの
#       出力を、先手視点カテゴリ別に比較する。
#   P2. 教師ノイズ地図: 同一局面の重複レコード間で s がどれだけばらつくかを
#       カテゴリ別に出す。ノイズの大きいセルと audit の漏れセルの対応を見る。
#   P4. 引き分け汚染: 引き分け対局由来のレコードの割合と、draw 局面に付いた
#       r の矛盾（draw なのに ±1）の大きさを測る。
#
# 実行（learn/ ディレクトリで）:
#   ../.venv/bin/python phase0_analysis.py
# 出力: phase0_result.md
#
# 注意:
#   - オラクル照合は正規化照合（鏡像+回転）を使う（check_unregistered2 の方式）。
#     ラベルは手番側視点で返るので、先手視点への変換を明示的に行う。
#   - P4 の対局復元は audit_order.py と同じ「連続エントリが1手でつながる間を
#     同一対局とする」方式。gamelibs の MoveList は len()+インデックスで扱うこと。

import argparse
import glob
import json
from collections import defaultdict
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


def oracle_label_sente(oracle, key):
    """正規化照合でラベルを引き、先手視点 (+1 先手勝ち / 0 引分 / -1 先手負け) に直す。
    lookup_variants のラベルは手番側視点なので、後手番なら符号を反転する"""
    direct, mirror, normalized, normalized_mirror = lookup_variants(oracle, key)
    label = normalized if normalized is not None else normalized_mirror
    if label is None:
        return None
    turn_is_white, _, _ = decode_key(key)
    return -label if turn_is_white else label


def load_group_entries(group):
    files = sorted(glob.glob("data/**/*.json", recursive=True))
    entries = []
    used_files = []
    for jf in files:
        if Path(jf).parent.name != group:
            continue
        with open(jf) as fh:
            try:
                entries.extend(json.load(fh))
                used_files.append(jf)
            except json.JSONDecodeError:
                continue
    return entries, used_files


def legal_moves_list(state):
    ml = state.legal_actions()
    return [ml[i] for i in range(len(ml))]


def main():
    parser = argparse.ArgumentParser(description="フェーズ0 再集計（P1/P2/P4）")
    parser.add_argument("--model", default="./model/best_single.h5")
    parser.add_argument("--out", default="phase0_result.md")
    parser.add_argument("--group", default="data")
    parser.add_argument("--min-dup", type=int, default=3, help="条件付き平均に使う最小重複数")
    parser.add_argument("--model-sample", type=int, default=20000, help="P1 でモデル推論する局面数の上限")
    args = parser.parse_args()

    md("# フェーズ0 再集計（phase0_analysis.py）")
    md()
    if not self_test():
        md("**self-test 失敗のため中止**")
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md_lines) + "\n")
        return

    oracle = LabelOracle()
    entries, used_files = load_group_entries(args.group)
    md(f"- 対象グループ: `{args.group}`（{len(used_files)} ファイル、{len(entries):,} レコード）")
    md()

    # ---- ハッシュごとに集約 ----
    by_key = defaultdict(list)  # key -> [(s, r)]
    invalid_r = 0
    for e in entries:
        r = e["r"]
        if r not in (-1, 0, 1):
            invalid_r += 1
            continue
        by_key[int(e["p"])].append((float(e["s"]), int(r)))
    md(f"- r が有効なレコードで集約: ユニーク {len(by_key):,} 局面（r 無効 {invalid_r:,} 件を除外）")
    md()

    dup_keys = [k for k, v in by_key.items() if len(v) >= args.min_dup]
    md(f"- 重複 {args.min_dup} 件以上の局面: {len(dup_keys):,}（P1/P2 の対象）")
    md()

    # ---- ラベル付け（先手視点）と終局除外 ----
    from game import from_hash
    labeled = {}  # key -> sente_label
    for k in dup_keys:
        state = from_hash(k)
        if state.pos.is_win() or state.pos.is_lose():
            continue
        lab = oracle_label_sente(oracle, k)
        if lab is not None:
            labeled[k] = lab
    md(f"- ラベル取得できた局面: {len(labeled):,}")
    md()

    # ================= P2. 教師ノイズ地図 =================
    md("## P2. 教師ノイズ地図（同一局面の重複レコード間での s のばらつき）")
    md()
    noise = defaultdict(list)  # sente_label -> [std of s]
    for k, lab in labeled.items():
        s_vals = np.array([s for s, _ in by_key[k]], dtype=float)
        noise[lab].append(float(s_vals.std()))
    md("| 先手視点カテゴリ | 局面数 | s の標準偏差（平均） | 中央値 | std>0.3 の割合 |")
    md("|---|---|---|---|---|")
    for lab, name in ((1, "先手勝ち"), (0, "引き分け"), (-1, "先手負け")):
        vals = noise.get(lab, [])
        if not vals:
            continue
        arr = np.array(vals)
        md(f"| {name} | {len(vals):,} | {arr.mean():.3f} | {np.median(arr):.3f} | {pct(int((arr > 0.3).sum()), len(arr)):.1f}% |")
    md()
    md("読み方: ノイズ（標準偏差）が大きいカテゴリほど、同じ局面でも探索のたびに教師が揺れている。audit の漏れセル（win 変換・draw 保持・先手負け）と重なるかを見る。")
    md()

    # ================= P1. 圧縮の分解 =================
    md("## P1. 圧縮の分解（混合教師の条件付き平均 vs モデルの出力）")
    md()
    keys = list(labeled.keys())
    rng = np.random.default_rng(123)
    if len(keys) > args.model_sample:
        idx = rng.choice(len(keys), args.model_sample, replace=False)
        keys = [keys[i] for i in idx]
        md(f"- モデル推論は {args.model_sample:,} 局面にサンプル（seed=123）")
        md()
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

    states = [from_hash(k) for k in keys]
    model_vals = batch_predict(states)
    rows = defaultdict(lambda: {"t": [], "v": []})  # sente_label -> target means / model values (先手視点)
    for k, v in zip(keys, model_vals):
        turn_is_white, _, _ = decode_key(k)
        t_vals = [(s + r) / 2.0 for s, r in by_key[k]]
        t_mean = float(np.mean(t_vals))
        t_sente = -t_mean if turn_is_white else t_mean
        v_sente = -float(v) if turn_is_white else float(v)
        rows[labeled[k]]["t"].append(t_sente)
        rows[labeled[k]]["v"].append(v_sente)
    md("| 先手視点カテゴリ | 局面数 | 混合教師の平均 | モデルの出力平均 | 差（モデル − 教師） |")
    md("|---|---|---|---|---|")
    for lab, name in ((1, "先手勝ち"), (0, "引き分け"), (-1, "先手負け")):
        d = rows.get(lab)
        if not d:
            continue
        tm = float(np.mean(d["t"]))
        vm = float(np.mean(d["v"]))
        md(f"| {name} | {len(d['t']):,} | {tm:+.3f} | {vm:+.3f} | {vm - tm:+.3f} |")
    md()
    md("読み方: 差がほぼ 0 なら、モデルは混合教師の平均を正しく再現しており、圧縮は教師の算術（対策は教師設計）。モデルが教師より有意に 0 側（勝ちで負の差、負けで正の差）なら、学習側の shrinkage が残っている。")
    md()

    # ================= P4. 引き分け汚染 =================
    md("## P4. 引き分け汚染")
    md()
    # ファイルごとに、連続エントリの接続で対局を復元する
    games = []  # 各対局: [(key, s, r), ...]
    for jf in used_files:
        with open(jf) as fh:
            try:
                file_entries = json.load(fh)
            except json.JSONDecodeError:
                continue
        prev_state = None
        current = []
        for e in file_entries:
            key = int(e["p"])
            connected = False
            if prev_state is not None:
                child_keys = {prev_state.next(a).hash_key() for a in legal_moves_list(prev_state)}
                connected = key in child_keys
            if not connected:
                if current:
                    games.append(current)
                current = []
            current.append((key, float(e["s"]), e["r"]))
            prev_state = from_hash(key)
        if current:
            games.append(current)
    drawn_games, decisive_games, invalid_games = [], [], []
    for g in games:
        rs = [r for _, _, r in g]
        if all(r == 0 for r in rs):
            drawn_games.append(g)
        elif any(r not in (-1, 0, 1) for r in rs):
            invalid_games.append(g)
        else:
            decisive_games.append(g)
    n_rec = sum(len(g) for g in games)
    n_drawn_rec = sum(len(g) for g in drawn_games)
    md(f"- 復元した対局: {len(games):,}（引き分け {len(drawn_games):,} / 決着 {len(decisive_games):,} / r 無効を含む {len(invalid_games):,}）")
    md(f"- 引き分け対局由来のレコード: {n_drawn_rec:,} / {n_rec:,}（{pct(n_drawn_rec, n_rec):.1f}%）")
    md()
    # 引き分け対局内の局面のオラクルカテゴリ分布
    cat_in_drawn = defaultdict(int)
    total_in_drawn = 0
    for g in drawn_games:
        for key, _, _ in g:
            state = from_hash(key)
            if state.pos.is_win() or state.pos.is_lose():
                continue
            lab = oracle_label_sente(oracle, key)
            if lab is None:
                continue
            cat_in_drawn[lab] += 1
            total_in_drawn += 1
    md("引き分け対局に含まれる局面のオラクルカテゴリ（先手視点）:")
    md()
    md("| カテゴリ | 件数 | 割合 |")
    md("|---|---|---|")
    for lab, name in ((1, "先手勝ち"), (0, "引き分け"), (-1, "先手負け")):
        n = cat_in_drawn.get(lab, 0)
        md(f"| {name} | {n:,} | {pct(n, total_in_drawn):.1f}% |")
    md()
    # 決着した対局内で、オラクル draw の局面に付いた r
    draw_r = defaultdict(int)
    for g in decisive_games:
        for key, _, r in g:
            if r not in (-1, 1):
                continue
            state = from_hash(key)
            if state.pos.is_win() or state.pos.is_lose():
                continue
            lab = oracle_label_sente(oracle, key)
            if lab == 0:
                draw_r[int(r)] += 1
    n_draw_r = sum(draw_r.values())
    md(f"- 決着した対局内でオラクル「引き分け」の局面に付いた r: +1 が {draw_r.get(1, 0):,} 件 / −1 が {draw_r.get(-1, 0):,} 件（合計 {n_draw_r:,} 件）")
    md("  これらのレコードは「引き分け局面に ±1」を教えており、draw セルの教師として矛盾している。合計件数が P4 の規模感になる。")
    md()

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"\n結果を {args.out} に書きました")


if __name__ == "__main__":
    main()
