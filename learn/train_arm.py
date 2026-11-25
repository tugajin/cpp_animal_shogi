# 実験5 4アーム factorial 用 学習ハーネス（train_arm.py）
# ============================================================
# 問い: 生成方法（B5: CN 選択）× 記録方法（B1: 木学習）の 2×2 factorial を、
# 全 arm 同一条件（未学習初期値・累計 5,000局 / 100k 局面予算・約20
# イテレーション）で回すための学習側。設計は EXP5_DESIGN.md の
# 「実施計画（2026-11-21 改訂）」、生成側の手順は ARM_RUNBOOK.md。
#
# train_network.py との違い（意図的なもの）:
#   - チェックポイントの自動再開をしない。train_network.py は model/*.save の
#     最新からモデルとオプティマイザを黙って引き継ぐが、本ハーネスは
#     --init で指定した重みだけを読み、オプティマイザは毎回新規に作る。
#   - データは --data-dir で渡した1イテレーション分だけを使う（全世代の
#     蓄積を毎回学習し直す実験4方式は、再学習と新規学習が分離できないため
#     採らない）。累計の予算管理は model-dir 内の arm_state.json で行う。
#   - 損失に共通衛生層を入れる: r 項のマスク（下記 r_mask_for）。
#   - モデル入出力は arm 別の --model-dir に閉じる（best_single.h5 /
#     best_single.onnx / best_single_jit.pt / arm_state.json / train_log.md）。
#
# r 項マスクの規則（EXP5_DESIGN.md 共通衛生層）:
#   - 木学習 arm（2, 3）: 木レコードは r=-99（結果なし）なので全レコードで
#     r 項をマスクし、s（バックアップ値）のみを教師とする。
#   - 根のみ arm（0, 1）: 証明レコード（pl=1）では r 項をマスクする。
#     pl はレコードにあればそれを使い、無ければ s から格子同定で計算する
#     （proven_flag、本ファイルに内蔵。k=200 は証明に数えない）。r=-99 の
#     レコードも安全のためマスクする。
#
# 引き分け対局の除外:
#   - 根のみ arm では、ファイル（=1対局）の最終レコード r=0 を引き分け
#     として除外できる（is_drawn_game、本ファイルに内蔵）。
#   - 木学習 arm ではレコードに結果が残らないため学習側では判定できない。
#     除外は生成側の課題として ARM_RUNBOOK.md に明記し、ここでは
#     「判定不能」とログに残す（実験4では引き分け対局はレコードの 1.8%）。
#
# 予算の間引き:
#   - 累計が上限（既定 5,000局 / 100,000局面）を超えないよう、対局（ファイル）
#     単位で間引く（sample_games、本ファイルに内蔵）。局面単位の
#     切り捨てはしない。
#
# 前提:
#   - --data-dir 内の *.json は1ファイル=1対局（C++ 自己対局の出力契約）。
#   - 特徴量の作り方は history_dataset.py と同一（from_hash → feature()）。
#   - 学習条件は train_network.py に合わせる（AdamW lr=1e-4, wd=1e-5、
#     15 epoch、batch 128、LAMBDA=0.5、損失はバッチ内 sum）。
#
# 実行（ホスト側 learn/ ディレクトリで、プロジェクトの .venv を使う）:
#   # 初期化（全 arm 共通の seed で乱数初期モデルを作り ONNX まで出す）
#   ../.venv/bin/python train_arm.py --arm 0 --model-dir model/arm0 --init-only --seed 123
#   # イテレーションの学習（gen_0 は初期モデルの自己対局データ）
#   ../.venv/bin/python train_arm.py --arm 0 --model-dir model/arm0 \
#       --data-dir data/arm0/gen_0 --init model/arm0/best_single.h5
#   ../.venv/bin/python train_arm.py --arm 1 --model-dir model/arm1 \
#       --data-dir data/arm1/gen_1 --init model/arm1/best_single.h5

import argparse
import json
import random
import sys
import time
from pathlib import Path

VERSION = "2026-11-24 v7（B1 対照実験: --relabel-raw / --max-depth を追加）"

# v7 の追加（B1 対照実験用。設計は EXPERIMENT.md「B1 対照実験の設計」）:
#   --relabel-raw: サンプルしたレコードのラベル s を、学習開始時点のモデル
#     （= --init の重み = その世代の生成モデル）の生評価値（0 探索）に
#     付け替える。局面の集合は木レコードのまま、ラベルの形だけを変える
#     対照（RAW arm）。r 項は触らない（木 arm では元々全マスク）。
#   --max-depth D: 木レコードを、探索の根からの深さ d <= D のものに
#     限定してからサンプルする（DEPTH arm）。レコードに "d" が無い場合は
#     静かに素通りせず中止する（深さ記録対応バイナリで生成したデータが
#     前提。C++ 側は selfplay.hpp の add_replay_buffer が "d" を書く）。


# ============================================================
# 純粋ロジック（torch / gamelibs 不要、self-test の対象）
# ============================================================

# ---- make_budget.py から複製した3関数（train_arm.py を単体で動かすため） ----
# 格子定数や判定を変えるときは make_budget.py / proven_label_check.py と
# 揃えること。

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

def r_mask_for(arm, pl, r):
    """r 項のマスク（1=損失に使う、0=使わない）。規則はヘッダ参照。"""
    if arm in (2, 3):
        return 0
    if pl == 1:
        return 0
    if r == -99:
        return 0
    return 1


def record_pl(record):
    """レコードの証明フラグ。pl キーがあればそれ、無ければ s から格子同定。"""
    if "pl" in record:
        return record["pl"]
    pl, _is_clamp = proven_flag(record.get("s"))
    return pl


def filter_drawn(games, arm):
    """引き分け対局の除外。games は (名前, レコード列) のリスト。
    戻り値: (残した games, 除外した対局数, 判定できたか)。
    木学習 arm（2, 3）はレコードに結果が無いため判定不能で全て残す。"""
    if arm in (2, 3):
        return games, 0, False
    kept = []
    dropped = 0
    for name, records in games:
        if is_drawn_game(records):
            dropped += 1
        else:
            kept.append((name, records))
    return kept, dropped, True


def select_within_budget(games, state, budget_games, budget_positions, seed,
                         iter_positions=0):
    """累計予算に収まるよう対局単位で間引く。
    iter_positions > 0 のときは、累計予算に加えて1イテレーション分の
    局面数上限も適用する（木学習 arm のように records/game が大きい場合に、
    予算を複数イテレーションへ分散させるため）。
    戻り値: (選んだ games, 使う局面数, 当たった上限)。"""
    remain_games = budget_games - state["games"]
    remain_positions = budget_positions - state["positions"]
    if remain_games <= 0 or remain_positions <= 0:
        return [], 0, "exhausted"
    quota_binds = 0 < iter_positions < remain_positions
    cap = iter_positions if quota_binds else remain_positions
    idx, pos, bound = sample_games(
        [records for _, records in games], remain_games, cap, seed)
    if quota_binds and bound == "positions":
        bound = "iter"
    return [games[i] for i in idx], pos, bound


def select_tree_records(games, state, budget_games, budget_positions, seed,
                        iter_positions=0):
    """木学習 arm 用: 対局単位でなくレコード単位で間引く。

    木レコードは生成時にほぼ無償で増える一方、対局単位で間引くと
    学習局面が数十局分に偏って多様性を失う。そのため生成した全対局の
    木レコードから一様にサンプルし、選ばれたレコードを対局ごとに
    まとめ直して返す（以降の処理は根 arm と共通）。上限の考え方は
    select_within_budget と同じ（累計予算＋イテレーション上限）。
    戻り値: (選んだ games, 使う局面数, 当たった上限)。"""
    remain_games = budget_games - state["games"]
    remain_positions = budget_positions - state["positions"]
    if remain_games <= 0 or remain_positions <= 0:
        return [], 0, "exhausted"
    quota_binds = 0 < iter_positions < remain_positions
    cap = iter_positions if quota_binds else remain_positions
    flat = [(gi, rec) for gi, (_name, records) in enumerate(games)
            for rec in records]
    if len(flat) <= cap:
        chosen = flat
        bound = "all"
    else:
        chosen = random.Random(seed).sample(flat, cap)
        bound = "iter" if quota_binds else "positions"
    grouped = {}
    for gi, rec in chosen:
        grouped.setdefault(gi, []).append(rec)
    selected = [(games[gi][0], grouped[gi]) for gi in sorted(grouped)]
    return selected, len(chosen), bound


def filter_by_depth(games, max_depth):
    """木レコードを根からの深さ d <= max_depth に限定する（DEPTH arm 用）。

    games は (名前, レコード列) のリスト。戻り値:
    (絞った games, 残したレコード数, 全レコード数, "d" 欠落レコード数)。
    空になった対局は落とす。"d" の欠落は呼び出し側で中止判断に使う
    （ここでは落とさず数だけ返す）。"""
    filtered = []
    kept = 0
    total = 0
    missing = 0
    for name, records in games:
        total += len(records)
        rows = []
        for rec in records:
            if "d" not in rec:
                missing += 1
                continue
            if rec["d"] <= max_depth:
                rows.append(rec)
        kept += len(rows)
        if rows:
            filtered.append((name, rows))
    return filtered, kept, total, missing


def apply_relabel(rows, values):
    """(p, s, r, mask) の列の s を values（同じ順の生評価値）で置き換える。
    戻り値: (新しい rows, |新 s − 旧 s| の平均)。"""
    assert len(rows) == len(values)
    new_rows = []
    diff_sum = 0.0
    for (p, s, r, mask), v in zip(rows, values):
        new_rows.append((p, float(v), r, mask))
        diff_sum += abs(float(v) - s)
    mean_diff = diff_sum / len(rows) if rows else 0.0
    return new_rows, mean_diff


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'OK' if cond else 'NG'}: {name}")
        ok = ok and cond

    # r マスク: 根のみ arm は証明と r 未記入だけをマスクする
    check("arm0 通常レコードは r を使う", r_mask_for(0, 0, 1) == 1)
    check("arm0 証明レコードは r をマスク", r_mask_for(0, 1, 1) == 0)
    check("arm0 r=-99 はマスク", r_mask_for(0, 0, -99) == 0)
    check("arm1 も根のみ規則", r_mask_for(1, 0, -1) == 1 and r_mask_for(1, 1, 0) == 0)
    # r マスク: 木学習 arm は全マスク
    check("arm2 は全マスク", r_mask_for(2, 0, 1) == 0 and r_mask_for(2, 1, 1) == 0)
    check("arm3 は全マスク", r_mask_for(3, 0, -1) == 0)

    # pl の取得: キーがあればそれ、無ければ格子同定（k=200 は証明にしない）
    check("pl キーを優先", record_pl({"p": 1, "s": 0.3, "pl": 1}) == 1)
    check("pl 無しは格子同定（証明）", record_pl({"p": 1, "s": 0.9999 - 3 / 2000.0}) == 1)
    check("pl 無しは格子同定（推定）", record_pl({"p": 1, "s": 0.35}) == 0)
    check("pl 無しは格子同定（k=200 は証明でない）", record_pl({"p": 1, "s": 0.8999}) == 0)

    # 引き分け除外: 根のみ arm だけ判定できる
    g_win = ("a", [{"p": 1, "s": 0.1, "r": 1}, {"p": 2, "s": 0.2, "r": -1}])
    g_draw = ("b", [{"p": 3, "s": 0.0, "r": 0}, {"p": 4, "s": 0.0, "r": 0}])
    kept, dropped, detectable = filter_drawn([g_win, g_draw], 0)
    check("arm0 は引き分け対局を除外", kept == [g_win] and dropped == 1 and detectable)
    kept, dropped, detectable = filter_drawn([g_win, g_draw], 2)
    check("arm2 は判定不能で全て残す", kept == [g_win, g_draw] and dropped == 0 and not detectable)

    # 予算の間引き: 累計に収め、使い切ったら何も選ばない
    games = [("g1", [None] * 30), ("g2", [None] * 25), ("g3", [None] * 40)]
    state = {"games": 4990, "positions": 99950}
    sel, pos, _bound = select_within_budget(games, state, 5000, 100000, seed=123)
    check("残り局面予算 50 に収まる", pos <= 50 and len(sel) >= 1)
    check("残り対局予算 10 に収まる", len(sel) <= 10)
    sel2, pos2, bound2 = select_within_budget(
        games, {"games": 5000, "positions": 100000}, 5000, 100000, seed=123)
    check("予算を使い切ったら空", sel2 == [] and pos2 == 0 and bound2 == "exhausted")
    a, _, _ = select_within_budget(games, {"games": 0, "positions": 0}, 5000, 100000, seed=7)
    b, _, _ = select_within_budget(games, {"games": 0, "positions": 0}, 5000, 100000, seed=7)
    check("同じ seed で同じ選択", a == b)

    # イテレーション上限: 累計予算より先に効き、上限として区別される
    sel3, pos3, bound3 = select_within_budget(
        games, {"games": 0, "positions": 0}, 5000, 100000, seed=123,
        iter_positions=50)
    check("イテレーション上限 50 に収まる", 0 < pos3 <= 50 and bound3 == "iter")
    sel4, pos4, bound4 = select_within_budget(
        games, {"games": 0, "positions": 99990}, 5000, 100000, seed=123,
        iter_positions=5000)
    check("残り累計予算が小さいときは累計側が効く",
          sel4 == [] and pos4 == 0 and bound4 == "positions")

    # 木学習 arm のレコード単位サンプル: 上限を守り、全対局に分散する
    tree_games = [(f"t{i}", [{"p": i * 1000 + j} for j in range(40)])
                  for i in range(3)]
    tsel, tpos, tbound = select_tree_records(
        tree_games, {"games": 0, "positions": 0}, 5000, 100000, seed=123,
        iter_positions=50)
    check("木レコードはイテレーション上限 50 に収まる",
          tpos == 50 and tbound == "iter")
    check("木レコードは複数対局に分散する", len(tsel) >= 2)
    check("木レコードの合計が局面数と一致",
          sum(len(recs) for _, recs in tsel) == 50)
    tsel2, tpos2, tbound2 = select_tree_records(
        tree_games, {"games": 0, "positions": 0}, 5000, 100000, seed=123)
    check("母集団が上限以下なら全部使う", tpos2 == 120 and tbound2 == "all")
    tsel3, _, tbound3 = select_tree_records(
        tree_games, {"games": 0, "positions": 100000}, 5000, 100000, seed=123)
    check("木レコードも予算を使い切ったら空",
          tsel3 == [] and tbound3 == "exhausted")

    # 深さフィルタ: d <= max_depth だけ残し、空の対局は落とし、欠落は数える
    d_games = [("x", [{"p": 1, "d": 0}, {"p": 2, "d": 1},
                      {"p": 3, "d": 2}, {"p": 4, "d": 1}]),
               ("y", [{"p": 5, "d": 3}]),
               ("z", [{"p": 6}])]
    f_games, kept, total, missing = filter_by_depth(d_games, 1)
    check("深さフィルタは d<=1 を残す",
          kept == 3 and total == 6 and [n for n, _ in f_games] == ["x"])
    check("深さフィルタは d 欠落を数える", missing == 1)

    # ラベル付け替え: s だけが置き換わり、差分の平均が返る
    rows = [(1, 0.5, -99, 0), (2, -0.2, -99, 0)]
    new_rows, mean_diff = apply_relabel(rows, [0.1, 0.3])
    check("付け替えで s が置き換わる",
          new_rows[0][1] == 0.1 and new_rows[1][1] == 0.3
          and new_rows[0][0] == 1 and new_rows[0][2] == -99)
    check("付け替え差分の平均", abs(mean_diff - 0.45) < 1e-12)

    print("self-test:", "OK" if ok else "NG")
    return ok


# ============================================================
# データ読み込み
# ============================================================

def load_games(data_dir):
    """data_dir の *.json を (ファイル名, レコード列) で読む。
    壊れたファイル・空ファイルは件数を数えて飛ばす。"""
    games = []
    n_skipped = 0
    for path in sorted(Path(data_dir).glob("*.json")):
        try:
            with path.open() as fh:
                records = json.load(fh)
        except (json.JSONDecodeError, OSError):
            n_skipped += 1
            continue
        if not records:
            n_skipped += 1
            continue
        games.append((path.name, records))
    return games, n_skipped


def load_state(model_dir):
    path = Path(model_dir) / "arm_state.json"
    if path.exists():
        with path.open() as fh:
            return json.load(fh)
    return {"arm": None, "games": 0, "positions": 0, "iterations": 0}


def save_state(model_dir, state):
    path = Path(model_dir) / "arm_state.json"
    with path.open("w") as fh:
        json.dump(state, fh, indent=2)


# ============================================================
# 学習（torch / gamelibs が要る部分はここに閉じる）
# ============================================================

def build_model(init, seed):
    """SingleNet を作る。init='random' なら seed を固定して乱数初期化し、
    それ以外は state_dict のパスとして重みだけを読む（オプティマイザは
    引き継がない）。"""
    import torch
    from single_network import SingleNet

    if init == "random":
        torch.manual_seed(seed)
        return SingleNet()
    model = SingleNet()
    state = torch.load(init, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def save_model_files(model_dir, model):
    """best_single.h5（state_dict）と、生成側が使う best_single.onnx /
    best_single_jit.pt を model_dir に書く。ONNX の契約は export_onnx.py と
    同一（input "input" / output "value" / opset 17 / batch 動的軸）。"""
    import torch
    from single_network import SingleNet

    model_dir = Path(model_dir)
    h5_path = model_dir / "best_single.h5"
    torch.save(model.state_dict(), h5_path)

    fresh = SingleNet()
    fresh.load_state_dict(torch.load(h5_path, map_location="cpu", weights_only=True))
    fresh.eval()

    jit_path = model_dir / "best_single_jit.pt"
    torch.jit.script(fresh).save(str(jit_path))

    onnx_path = model_dir / "best_single.onnx"
    dummy = torch.zeros(1, 42, 3, 4, dtype=torch.float32)
    torch.onnx.export(
        fresh, dummy, str(onnx_path),
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["input"], output_names=["value"],
        dynamic_axes={"input": {0: "batch"}, "value": {0: "batch"}})
    return h5_path, onnx_path, jit_path


def evaluate_ans(model_dir):
    """../oracle/problem.txt に対する ans（±0.2 分類）を返す。
    全体と、draw ラベルを除いた win/lose のみ（現行の主指標はこちら）。"""
    import numpy as np
    import torch
    from game import from_hash
    from single_network import DN_INPUT_SHAPE, SingleNet

    problem_path = Path("../oracle/problem.txt")
    if not problem_path.exists():
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingleNet()
    model.load_state_dict(torch.load(
        Path(model_dir) / "best_single.h5", map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    file, rank, channel = DN_INPUT_SHAPE
    n_all = n_all_ok = n_wl = n_wl_ok = 0
    with problem_path.open() as fh:
        lines = [line.strip() for line in fh if line.strip()]
    with torch.no_grad():
        for line in lines:
            key_str, result_str = line.split(",")
            result = int(result_str)
            state = from_hash(int(key_str))
            feature = np.array(state.feature()).reshape(1, channel, file, rank)
            score = model(torch.from_numpy(feature).to(device).float()).item()
            pred = 1 if score > 0.2 else (-1 if score < -0.2 else 0)
            n_all += 1
            n_all_ok += 1 if pred == result else 0
            if result != 0:
                n_wl += 1
                n_wl_ok += 1 if pred == result else 0
    return {
        "ans_all": n_all_ok / n_all if n_all else 0.0,
        "n_all": n_all,
        "ans_winlose": n_wl_ok / n_wl if n_wl else 0.0,
        "n_winlose": n_wl,
    }


def train(args):
    import numpy as np
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    from game import from_hash
    from single_network import DN_INPUT_SHAPE

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(model_dir)
    if state["arm"] is not None and state["arm"] != args.arm:
        print(f"** arm_state.json の arm={state['arm']} と --arm {args.arm} が"
              "食い違うため中止（model-dir の取り違え防止）**")
        return False
    state["arm"] = args.arm

    log_lines = []

    def log(line=""):
        log_lines.append(line)
        print(line)

    log(f"## train_arm: arm {args.arm} / iteration {state['iterations']}"
        f"（{time.strftime('%Y-%m-%d %H:%M:%S')}）")
    log()

    # ---- 初期化だけの実行（生成に使う未学習モデルを出す） ----
    if args.init_only:
        model = build_model("random", args.seed)
        h5, onnx, jit = save_model_files(model_dir, model)
        save_state(model_dir, state)
        log(f"- 初期モデル（seed={args.seed} の乱数初期化）を保存: {h5} / {onnx} / {jit}")
        log(f"- 累計: {state['games']:,} 局 / {state['positions']:,} 局面")
        with (model_dir / "train_log.md").open("a", encoding="utf-8") as fh:
            fh.write("\n".join(log_lines) + "\n\n")
        return True

    if args.data_dir is None or args.init is None:
        print("** --data-dir と --init が必要です（初期化だけなら --init-only）**")
        return False

    # ---- データの準備: 読み込み → 引き分け除外 → 予算で間引き ----
    games, n_skipped = load_games(args.data_dir)
    total_positions = sum(len(records) for _, records in games)
    log(f"- データ: `{args.data_dir}` から {len(games):,} 局 / {total_positions:,} 局面"
        f"（読めなかった・空: {n_skipped:,} ファイル）")
    if not games:
        log("** 対局が0件のため中止 **")
        with (model_dir / "train_log.md").open("a", encoding="utf-8") as fh:
            fh.write("\n".join(log_lines) + "\n\n")
        return False

    games, n_drawn, detectable = filter_drawn(games, args.arm)
    if detectable:
        log(f"- 引き分け対局の除外: {n_drawn:,} 局（残り {len(games):,} 局）")
    else:
        log("- 引き分け対局の除外: 木学習 arm のため学習側では判定不能"
            "（レコードに結果なし。生成側の課題として ARM_RUNBOOK.md に記載）")

    # ---- 深さの限定（DEPTH arm）: サンプル前に母集団を d <= max_depth に絞る ----
    if args.max_depth >= 0:
        games, kept, total, missing = filter_by_depth(games, args.max_depth)
        if missing:
            log(f"** レコードに深さ \"d\" が無いものが {missing:,} 件あります。"
                "深さ記録対応バイナリで生成したデータではないため中止 **")
            with (model_dir / "train_log.md").open("a", encoding="utf-8") as fh:
                fh.write("\n".join(log_lines) + "\n\n")
            return False
        log(f"- 深さフィルタ（--max-depth {args.max_depth}）: "
            f"{total:,} 局面中 {kept:,} 局面が残った（残り {len(games):,} 局）")
        if not games:
            log("** 深さフィルタ後にデータが空のため中止 **")
            with (model_dir / "train_log.md").open("a", encoding="utf-8") as fh:
                fh.write("\n".join(log_lines) + "\n\n")
            return False

    if args.arm in (2, 3) and not args.iter_positions:
        log("** 警告: 木学習 arm で --iter-positions が未指定です。"
            "この回は累計予算の残り全部まで学習します"
            "（意図した実行か確認してください）**")
    if args.arm in (2, 3):
        selected, sel_positions, bound = select_tree_records(
            games, state, args.budget_games, args.budget_positions, args.seed,
            iter_positions=args.iter_positions)
    else:
        selected, sel_positions, bound = select_within_budget(
            games, state, args.budget_games, args.budget_positions, args.seed,
            iter_positions=args.iter_positions)
    if not selected:
        log("** 累計予算を使い切っているため中止（arm_state.json を確認）**")
        with (model_dir / "train_log.md").open("a", encoding="utf-8") as fh:
            fh.write("\n".join(log_lines) + "\n\n")
        return False
    bound_name = {"games": "対局数上限", "positions": "局面数上限",
                  "iter": "イテレーション上限",
                  "all": "母集団を使い切った", "exhausted": "予算切れ"}[bound]
    log(f"- 予算で間引き: {len(selected):,} 局 / {sel_positions:,} 局面を学習する"
        f"（上限に当たったのは {bound_name}）")
    if args.iter_positions:
        log(f"- イテレーション上限: {args.iter_positions:,} 局面/回"
            "（累計予算とは別に適用）")
    if args.arm in (2, 3):
        log(f"- 木学習 arm: 全 {len(games):,} 局の木レコードからレコード単位で"
            f"サンプル（触れた対局 {len(selected):,} 局）")

    records = []
    n_masked = 0
    for _name, game_records in selected:
        for rec in game_records:
            mask = r_mask_for(args.arm, record_pl(rec), rec.get("r", -99))
            n_masked += 1 - mask
            records.append((rec["p"], rec["s"], rec.get("r", -99), mask))
    log(f"- r 項マスク: {n_masked:,} / {len(records):,} レコード"
        f"（{100.0 * n_masked / len(records):.1f}%）")

    class ArmDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, index):
            p, s, r, mask = self.rows[index]
            state_ = from_hash(p)
            file, rank, channel = DN_INPUT_SHAPE
            feature = np.array(state_.feature()).reshape(channel, file, rank)
            return feature, s, r, mask

        def __len__(self):
            return len(self.rows)

    # ---- 学習（train_network.py と同じ条件。オプティマイザは毎回新規） ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.init, args.seed).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.00001)

    # ---- ラベルの付け替え（RAW arm）: 学習前のモデル（= 生成モデル）の
    # 生評価値を新しい s とする。局面の集合と r / mask は変えない ----
    if args.relabel_raw:
        model.eval()
        raw_values = []
        keys = [p for p, _s, _r, _m in records]
        with torch.no_grad():
            for i in range(0, len(keys), 512):
                feats = []
                for p in keys[i:i + 512]:
                    st = from_hash(p)
                    file, rank, channel = DN_INPUT_SHAPE
                    feats.append(np.array(st.feature()).reshape(channel, file, rank))
                x = torch.from_numpy(np.stack(feats)).float().to(device)
                raw_values.extend(model(x).reshape(-1).tolist())
        records, mean_diff = apply_relabel(records, raw_values)
        log(f"- ラベル付け替え（--relabel-raw）: {len(records):,} レコードの s を"
            f"生成モデル（--init の重み）の生評価値に置き換えた"
            f"（|新 s − 旧 s| の平均 {mean_diff:.4f}）")
    dataloader = DataLoader(
        dataset=ArmDataset(records), batch_size=args.batch_size, shuffle=True)

    start = time.time()
    avg_loss = None
    for epoch in range(args.epochs):
        sum_loss = 0.0
        sum_num = 0
        for x, y_s, y_r, mask in dataloader:
            model.train()
            x = x.float().to(device)
            y_s = y_s.float().to(device)
            y_r = y_r.float().to(device)
            mask = mask.float().to(device)
            optimizer.zero_grad()
            outputs = torch.squeeze(model(x))
            loss = (args.lam * torch.sum((outputs - y_s) ** 2)
                    + (1 - args.lam) * torch.sum(mask * (outputs - y_r) ** 2))
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            sum_num += 1
        avg_loss = sum_loss / max(sum_num, 1)
        print(f"epoch:{epoch} avg loss:{avg_loss:.4f} sec:{int(time.time() - start)}")

    # ---- 保存・評価・状態の更新 ----
    h5, onnx, jit = save_model_files(model_dir, model)
    log(f"- 保存: {h5} / {onnx} / {jit}（最終 epoch の平均損失 {avg_loss:.4f}）")

    ans = evaluate_ans(model_dir)
    if ans is not None:
        log(f"- ans（参考）: 全体 {ans['ans_all']:.4f}（{ans['n_all']:,} 局面） / "
            f"win/lose のみ {ans['ans_winlose']:.4f}（{ans['n_winlose']:,} 局面）")
    else:
        log("- ans: ../oracle/problem.txt が無いため未実施")

    state["games"] += len(selected)
    state["positions"] += sel_positions
    state["iterations"] += 1
    save_state(model_dir, state)
    log(f"- 累計: {state['games']:,} 局 / {state['positions']:,} 局面"
        f"（予算 {args.budget_games:,} 局 / {args.budget_positions:,} 局面）")
    log()

    with (model_dir / "train_log.md").open("a", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="実験5 4アーム factorial 用 学習ハーネス")
    parser.add_argument("--arm", type=int, required=True, choices=[0, 1, 2, 3],
                        help="0=Baseline / 1=B5(CN) / 2=B1(木学習) / 3=B5+B1")
    parser.add_argument("--model-dir", required=True,
                        help="arm 専用のモデル出力先（例: model/arm1）")
    parser.add_argument("--data-dir", default=None,
                        help="このイテレーションの対局データ（*.json のあるディレクトリ）")
    parser.add_argument("--init", default=None,
                        help="初期重み: state_dict のパス、または random")
    parser.add_argument("--init-only", action="store_true",
                        help="乱数初期モデルを作って保存するだけ（学習しない）")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lam", type=float, default=0.5, help="損失の s 項の重み（LAMBDA）")
    parser.add_argument("--budget-games", type=int, default=5000)
    parser.add_argument("--budget-positions", type=int, default=100000)
    parser.add_argument("--iter-positions", type=int, default=0,
                        help="1イテレーションで学習する局面数の上限"
                             "（0=制限なし。木学習 arm 用）")
    parser.add_argument("--relabel-raw", action="store_true",
                        help="ラベル s を学習開始時点のモデルの生評価値に"
                             "付け替える（B1 対照実験の RAW arm）")
    parser.add_argument("--max-depth", type=int, default=-1,
                        help="木レコードを根からの深さ d <= この値に限定する"
                             "（-1=無効。B1 対照実験の DEPTH arm）")
    args = parser.parse_args()

    print(f"train_arm.py version {VERSION}")
    if not self_test():
        print("** self-test 失敗のため中止 **")
        sys.exit(1)
    if not train(args):
        sys.exit(1)


if __name__ == "__main__":
    main()
