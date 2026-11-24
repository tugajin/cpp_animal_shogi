# 実験5 段階A用 予算プールの抽出（make_budget.py）
# ============================================================
# 問い: 段階Aの全 arm が共通で使う学習プール（5,000局分・約100,000局面）を、
# 実験4の最新世代データから対局単位で切り出す。局面単位の抽出はしない
# （短期ターゲットの計算と r の整合のため、EXP5_DESIGN.md の制約）。
#
# 前提（コード読解で確認）:
#   - ai/selfplay.hpp の ReplayBuffer は1対局ごとに1ファイル
#     （data/selfplay_<timestamp>_<rand>.json）を書き出す。1ファイル ≒ 1対局。
#   - ただし merge 系で結合されたファイルがあり得るため、ファイル内では
#     audit_order.py と同一の接続判定（連続エントリが1手でつながる間を
#     同一対局とする）で対局を復元し、つながらなければ分割する。
#   - レコードは {"p": hash, "s": 教師値, "r": 対局結果, ("c": CN注釈)}。
#     r は対局終了時に overwrite_result() で埋められる（USE_DESCENT=false時）。
#     r=-99 が残っているレコードは結果未記入なので件数を報告する。
#   - 千日手（引き分け）対局はプールから除外する（2026-11-21 ユーザー決定:
#     千日手関係は実験・評価の対象外）。判定は最終レコードの r == 0
#     （overwrite_result の埋め方により最終手番視点の値で、引き分けは 0）。
#     含めたい場合は --include-drawn を付ける。
#   - 各レコードに証明ラベルの注釈 "pl" を付ける（proven_label_check.py と
#     同一の格子同定）。pl=1 は証明ラベル（格子一致かつ k≠200）、pl=0 はそれ
#     以外。k=200（s=±0.8999）は predict のクランプ飽和と紛れるため証明に
#     数えず、マニフェストで件数を分離して報告する。学習側は pl=1 の
#     レコードで r 項をマスクする（実験5の共通衛生層、EXP5_DESIGN.md）。
#
# 実行（ホスト側 learn/ ディレクトリで、プロジェクトの .venv を使う。
# data/ と gamelibs が必要。対局復元に game.py を使う）:
#   cd ~/Documents/cpp_animal_shogi/learn
#   ../.venv/bin/python make_budget.py --group data --out-group exp5_budget
#   ../.venv/bin/python make_budget.py --group gen_16 --out-group exp5_budget --seed 123
#
# 出力:
#   - data/<out-group>/ に抽出対局を1対局1ファイルで書き出す（中身は元レコードに
#     証明注釈 "pl" を加えたもの。ファイル名は元ファイル名を維持し、衝突時のみ
#     _2, _3 を付ける）。
#   - make_budget_result.md にマニフェスト（元グループ、対局数・局面数、seed、
#     どちらかの上限に当たったか、r=-99 件数、対局結果の分布、再読込による検証）。
#   - 既に中身のある出力先がある場合は --force がない限り中止する。
#
# 注意: gamelibs の MoveList は __getitem__ に境界チェックが無く、for で回すと
# 範囲外を読み続けて segfault する。必ず len() + インデックスでアクセスすること
# （audit_order.py の legal_moves_list と同一）。

import argparse
import glob
import json
import random
from pathlib import Path


def self_test():
    """抽出ロジックの自己テスト（gamelibs・データ不要の純粋関数だけを検証）"""
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'OK' if cond else 'NG'}: {name}")
        ok = ok and cond

    # 合成対局: 長さの異なる対局のリスト（要素はレコード数だけのダミー）
    games = [[None] * n for n in (21, 19, 25, 1, 30, 21, 21, 18, 22, 20)]

    # 1. 対局単位: サンプルされた対局は分割されず、局面数の合計が一致する
    idx, pos, bound = sample_games(games, max_games=5, max_positions=100000, seed=123)
    sel = [games[i] for i in idx]
    check("5局ちょうどで止まる", len(sel) == 5)
    check("局面数 = 選んだ対局の長さの合計",
          pos == sum(len(g) for g in sel))
    check("上限は対局数側", bound == "games")

    # 2. 局面数上限: 100局面の上限で、入る対局はシャッフル順にすべて拾う
    idx, pos, bound = sample_games(games, max_games=5000, max_positions=100, seed=123)
    check("局面数が上限以下", pos <= 100)
    check("上限は局面数側", bound == "positions")
    check("選ばれなかった対局は残り枠に入らない（拾い残しがない）",
          all(pos + len(games[i]) > 100 for i in range(len(games)) if i not in idx))

    # 3. 決定性: 同じ seed では同じ選択、違う seed では（ほぼ確実に）違う選択
    a, _, _ = sample_games(games, max_games=5, max_positions=100000, seed=123)
    b, _, _ = sample_games(games, max_games=5, max_positions=100000, seed=123)
    c, _, _ = sample_games(games, max_games=5, max_positions=100000, seed=999)
    check("同じ seed で同じ選択", a == b)
    check("違う seed で違う選択", a != c)

    # 4. 空入力・上限0でも壊れない
    idx, pos, bound = sample_games([], max_games=5000, max_positions=100000, seed=1)
    check("空入力は空出力", idx == [] and pos == 0)
    idx, pos, bound = sample_games(games, max_games=0, max_positions=100000, seed=1)
    check("上限0局は空出力", idx == [] and pos == 0)

    # 5. 引き分け対局の判定: 最終レコードの r で判定し、不明なものは除外しない
    check("最終 r=0 は引き分け", is_drawn_game([{"r": 1}, {"r": 0}]))
    check("最終 r=-1 は決着", not is_drawn_game([{"r": 0}, {"r": -1}]))
    check("r=-99（未記入）は引き分け扱いしない", not is_drawn_game([{"r": -99}]))
    check("r 欠落は引き分け扱いしない", not is_drawn_game([{"p": 1}]))
    check("空対局は引き分け扱いしない", not is_drawn_game([]))

    # 6. 証明ラベルの格子同定: k=200 は証明に数えず分離する
    check("格子 k=0 を同定", lattice_k(0.9999) == 0)
    check("格子 k=37 を同定", lattice_k(0.9999 - 37 * LATTICE_STEP) == 37)
    check("格子外（0.5）は None", lattice_k(0.5) is None)
    check("k=200 は格子一致する", lattice_k(0.8999) == CLAMP_K)
    check("証明 s は pl=1", proven_flag(0.9999 - 3 * LATTICE_STEP) == (1, False))
    check("負の証明 s も pl=1", proven_flag(-(0.9999 - 3 * LATTICE_STEP)) == (1, False))
    check("k=200 は pl=0 かつクランプ分離", proven_flag(0.8999) == (0, True))
    check("推定 s は pl=0", proven_flag(0.35) == (0, False))
    check("s 欠落は pl=0", proven_flag(None) == (0, False))

    print("self-test:", "OK" if ok else "NG")
    return ok


def is_drawn_game(game):
    """引き分け対局かどうか。最終レコードの r が 0 なら引き分け
    （r は最終手番視点で埋められるが、引き分けは視点によらず 0）。
    r 未記入（-99 や欠落）は結果不明として引き分け扱いしない（除外しない）。"""
    if not game or not isinstance(game[-1], dict):
        return False
    return game[-1].get("r", -99) == 0


# 証明ラベルの格子同定（proven_label_check.py と同一の定数・判定）
LATTICE_BASE = 0.9999
LATTICE_STEP = 1.0 / 2000.0
LATTICE_TOL = 1.0e-5
CLAMP_K = 200  # s=±0.8999 は格子点 k=200 と predict のクランプ飽和が衝突する


def lattice_k(s_abs):
    """|s| が格子点 BASE - k*STEP に一致すれば k を、しなければ None を返す"""
    d = (LATTICE_BASE - s_abs) / LATTICE_STEP
    k = int(round(d))
    if k < 0:
        return None
    if abs(LATTICE_BASE - k * LATTICE_STEP - s_abs) < LATTICE_TOL:
        return k
    return None


def proven_flag(s):
    """レコードの s に対する証明注釈を返す。
    (pl, is_clamp_k200): pl=1 は証明ラベル（格子一致かつ k≠200）。
    k=200 は証明に数えず、is_clamp_k200=True で分離して数える。"""
    if not isinstance(s, (int, float)):
        return 0, False
    k = lattice_k(abs(s))
    if k is None:
        return 0, False
    if k == CLAMP_K:
        return 0, True
    return 1, False


def sample_games(games, max_games, max_positions, seed):
    """対局単位でサンプリングする。

    シャッフルした順に対局を眺め、対局数上限・局面数上限のどちらにも
    収まる対局だけを拾う（収まらない対局は飛ばして次を見るので、上限近く
    まで詰める）。戻り値: (選んだ対局のインデックス, 局面数合計, 当たった上限)。
    上限の判定: 全対局を拾い切れば "all"、対局数上限に達すれば "games"、
    それ以外（残りは局面数上限に入らない対局だけ）なら "positions"。"""
    order = list(range(len(games)))
    random.Random(seed).shuffle(order)
    selected = []
    positions = 0
    for i in order:
        if len(selected) >= max_games:
            continue
        if positions + len(games[i]) > max_positions:
            continue
        selected.append(i)
        positions += len(games[i])
    if len(selected) == len(games):
        bound = "all"
    elif len(selected) >= max_games:
        bound = "games"
    else:
        bound = "positions"
    return selected, positions, bound


def legal_moves_list(state):
    """MoveList を安全にリスト化する（イテレーションは segfault するのでインデックスで）"""
    ml = state.legal_actions()
    return [ml[i] for i in range(len(ml))]


def reconstruct_games(entries, from_hash):
    """1ファイル分のエントリ列を、audit_order.py と同一の接続判定で対局に分割する。

    連続する2エントリについて、前の局面の合法手の子局面に次の局面が
    含まれれば同一対局、含まれなければ新しい対局とする。"""
    games = []
    cur = []
    prev_state = None
    for e in entries:
        try:
            key = int(e["p"])
        except (KeyError, TypeError, ValueError):
            continue
        if prev_state is not None:
            child_keys = set()
            for a in legal_moves_list(prev_state):
                child_keys.add(prev_state.next(a).hash_key())
            if key not in child_keys:
                if cur:
                    games.append(cur)
                cur = []
        cur.append(e)
        prev_state = from_hash(key)
    if cur:
        games.append(cur)
    return games


def main():
    parser = argparse.ArgumentParser(description="実験5 段階A用 予算プールの抽出")
    parser.add_argument("--group", default="data",
                        help="抽出元のグループ（data/<group>/。既定は data 直下の data）")
    parser.add_argument("--out-group", default="exp5_budget",
                        help="出力先グループ（data/<out-group>/ を作る）")
    parser.add_argument("--games", type=int, default=5000, help="対局数の上限")
    parser.add_argument("--positions", type=int, default=100000, help="局面数の上限")
    parser.add_argument("--seed", type=int, default=123, help="シャッフルの seed")
    parser.add_argument("--out", default="make_budget_result.md")
    parser.add_argument("--force", action="store_true",
                        help="出力先が既に中身を持っていても上書きする")
    parser.add_argument("--include-drawn", action="store_true",
                        help="引き分け対局もプールに含める（既定は千日手対象外化のため除外）")
    args = parser.parse_args()

    md_lines = []

    def md(line=""):
        md_lines.append(line)
        print(line)

    md("# 予算プールの抽出（make_budget.py）")
    md()
    if not self_test():
        md("**self-test 失敗のため中止**")
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md_lines) + "\n")
        return

    from game import from_hash

    files = sorted(glob.glob("data/**/*.json", recursive=True))
    games = []  # (source_basename, [records])
    n_files = 0
    n_skipped_files = 0
    n_split_files = 0
    for jf in files:
        if Path(jf).parent.name != args.group:
            continue
        n_files += 1
        try:
            with open(jf) as fh:
                entries = json.load(fh)
        except (json.JSONDecodeError, OSError):
            n_skipped_files += 1
            continue
        if not entries:
            n_skipped_files += 1
            continue
        split = reconstruct_games(entries, from_hash)
        if len(split) > 1:
            n_split_files += 1
        for g in split:
            games.append((Path(jf).name, g))
    total_positions = sum(len(g) for _, g in games)
    md(f"- 元グループ: `{args.group}` / 読んだファイル: {n_files:,} "
       f"（読めなかった・空: {n_skipped_files:,}、1ファイルに複数対局あり: {n_split_files:,}）")
    md(f"- 復元した対局数: {len(games):,} / 局面数: {total_positions:,}")

    # 千日手（引き分け）対局の除外（2026-11-21 ユーザー決定）
    n_drawn_games = 0
    n_drawn_positions = 0
    if not args.include_drawn:
        kept = []
        for src_name, g in games:
            if is_drawn_game(g):
                n_drawn_games += 1
                n_drawn_positions += len(g)
            else:
                kept.append((src_name, g))
        games = kept
        md(f"- 引き分け対局の除外: {n_drawn_games:,} 局 / {n_drawn_positions:,} 局面"
           f"（残り {len(games):,} 局 / {sum(len(g) for _, g in games):,} 局面から抽出する）")
    else:
        md("- 引き分け対局の除外: なし（--include-drawn）")
    md()

    if not games:
        md("**対局が0件のため中止（--group を確認してください）**")
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md_lines) + "\n")
        return

    chosen_idx, sel_positions, bound = sample_games(
        [g for _, g in games], args.games, args.positions, args.seed)
    selected = [games[i][1] for i in chosen_idx]
    assert sel_positions == sum(len(g) for g in selected)
    bound_name = {"games": "対局数上限", "positions": "局面数上限",
                  "all": "母集団を使い切った"}[bound]
    md(f"- 抽出: {len(selected):,} 局 / {sel_positions:,} 局面"
       f"（seed={args.seed}、上限に当たったのは {bound_name}）")
    md()

    out_dir = Path("data") / args.out_group
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        md(f"**出力先 `{out_dir}/` に既にファイルがあるため中止（上書きする場合は --force）**")
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md_lines) + "\n")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # r=-99（結果未記入）と対局結果の分布を数える。対局結果は最終レコードの r
    # （overwrite_result は手番を交互に反転させながら埋めるため、最終手番視点の値）。
    n_r99 = 0
    n_proven = 0
    n_clamp = 0
    result_dist = {}
    used_names = set()
    for i in chosen_idx:
        src_name, g = games[i]
        annotated = []
        for e in g:
            if e.get("r", -99) == -99:
                n_r99 += 1
            pl, is_clamp = proven_flag(e.get("s"))
            n_proven += pl
            n_clamp += 1 if is_clamp else 0
            e2 = dict(e)
            e2["pl"] = pl
            annotated.append(e2)
        last_r = g[-1].get("r", -99)
        result_dist[last_r] = result_dist.get(last_r, 0) + 1
        name = src_name
        k = 2
        while name in used_names:
            stem = Path(src_name).stem
            name = f"{stem}_{k}.json"
            k += 1
        used_names.add(name)
        with open(out_dir / name, "w", encoding="utf-8") as fh:
            json.dump(annotated, fh)
    md(f"- 出力先: `{out_dir}/`（{len(chosen_idx):,} ファイル、元ファイル名を維持）")
    md(f"- 証明ラベル注釈 `pl`: pl=1 が {n_proven:,} 件"
       f"（{100.0 * n_proven / max(sel_positions, 1):.1f}%、格子一致かつ k≠200）。"
       f" k=200（±0.8999、クランプ飽和と分離）: {n_clamp:,} 件は証明に数えていない")
    md(f"- r=-99 のレコード: {n_r99:,} 件（{100.0 * n_r99 / max(sel_positions, 1):.2f}%）"
       f" — 0件でない場合は元データの結果記入を確認すること")
    md(f"- 対局結果の分布（最終レコードの r）: "
       + ", ".join(f"r={k}: {v:,}局" for k, v in sorted(result_dist.items())))
    md()

    # ---- 検証: 出力先を再読込して数え直す ----
    verify_files = sorted(out_dir.glob("*.json"))
    v_games = 0
    v_positions = 0
    for vf in verify_files:
        with open(vf) as fh:
            v_games += 1
            v_positions += len(json.load(fh))
    md("## 検証（再読込）")
    md()
    md(f"- 出力ファイル数: {len(verify_files):,}（期待 {len(chosen_idx):,}）")
    md(f"- 出力の局面数: {v_positions:,}（期待 {sel_positions:,}）")
    check_ok = (len(verify_files) == len(chosen_idx) and v_positions == sel_positions)
    md(f"- 一致: **{'OK' if check_ok else 'NG — 出力を確認してください'}**")
    md()
    md("## 次の手順")
    md()
    md("- このプール（`data/" + args.out_group + "/`）が全 arm 共通の学習データになる。"
       " train 側は HistoryDataset にこのディレクトリだけを渡すこと"
       "（train_network.py の path_list / train.py の呼び出し側で指定）")
    md("- A9（順序損失）の arm では、このプールの各対局に実着手を注釈する別ステップ"
       "（audit_order.py の対局復元と同じ接続判定で子局面を列挙）が要る。"
       "このスクリプトは局面の抽出までで、注釈は含まない")
    md("- r 項マスクは学習側の変更: pl=1 のレコードでは損失の r 項を無効化する"
       "（EXP5_DESIGN.md の共通衛生層）。このプール自体は s・r を元のまま保持する")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"\n結果を {args.out} に書きました")


if __name__ == "__main__":
    main()
