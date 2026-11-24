# 実験5 段階A arm A9L: 木レコードの兄弟ランキング学習（train_rank.py）
# ============================================================
# 問い: B1（木学習）の記録には、探索木の backup が作った「兄弟局面の完全な
# 順序」が含まれている。train_arm.py は各レコードの絶対値 s への MSE しか
# 使わないが、A9L は兄弟グループ内の順序そのものを listwise 損失で学習に
# 加える。factorial の結論（B1 採用・B5 不採用、Stage A の土台は arm 2 の
# 方式）を受けた段階Aの arm。設計は EXP5_DESIGN.md の A9L の項。
#
# A9（実着手ラベル）との違い: A9 のラベルは「実際に指された手」で、損失は
# 実着手 vs 他の全手の pairwise。本 arm のラベルは探索の backup 値そのもの
# で、兄弟全員の順序を教師分布として合わせる（ListNet 型）。両者は併用
# 可能だが、まずは単独で測る。
#
# データと符号の規約（ai/selfplay.hpp / learn/audit_order.py で確認済み）:
#   - レコードは {"p": ハッシュキー(int), "s": 値, "r": -99}。木レコードの
#     s はそのノードの手番側視点の backup 値 w（add_replay_buffer が
#     node->w をそのまま記録する）。モデルの出力も手番側視点。
#   - 兄弟の比較は「指す側視点」で行う: 子局面 c の指す側価値は -s(c)。
#     教師分布 q = softmax(-s(子) / tau)、モデルの分布 p = softmax(-v(子)
#     / tau) とし、損失は交差エントロピー -sum(q * log p)。
#   - 兄弟グループは学習側で再構成する: 記録された局面を親として合法手を
#     列挙し、同じ対局ファイル内に記録がある子だけをグループにする
#     （add_replay_buffer は「展開されたノード＋未展開だが解決済みの子」を
#     記録するので、記録のある子同士は同じ木の兄弟である）。
#   - 同一ファイル内に同じ局面が複数回記録された場合、s は平均する
#     （dlshogi の重複ラベル平均と同じ扱い）。
#
# 予算の会計（arm 2 と揃える）:
#   - 学習局面数 = グループの親と子＋補充レコードの、局面キーの相異なる数。
#     1イテレーション上限（既定 5,000）と累計上限（既定 100,000）の両方を
#     守る。グループで上限に届かない場合は、残りを一様サンプルのレコード
#     （MSE のみ）で補充するので、グループが乏しい世代でも arm 2 相当に
#     劣化するだけで止まらない。
#   - 子の二重使用はしない: ある局面が子として採用されたら、同じ
#     イテレーションの他のグループでは子に使わない（親にはなれる）。
#
# 損失: train_arm.py の木 arm と同じ MSE 項（lam * sum((v - s)^2)、r は
# 木レコードなので全マスクで項自体なし）に、順序項
# rank_weight * sum_groups( 子の数 * CE ) を加える。順序項を子の数で
# 重み付けするのは、MSE 項が局面単位の和であることに揃えるため。
#
# 実行（ホスト側 learn/ ディレクトリで、プロジェクトの .venv を使う）:
#   # 初期化（factorial と同じ seed の乱数初期モデル）
#   ../.venv/bin/python train_rank.py --model-dir model/a9l --init-only --seed 123
#   # グループ再構成の事前確認（学習しない。arm 2 の既存データで測れる）
#   ../.venv/bin/python train_rank.py --analyze --data-dir data/arm2_gen_19
#   # イテレーションの学習（生成は arm 2 と同一設定。run_rank.sh が回す）
#   ../.venv/bin/python train_rank.py --model-dir model/a9l \
#       --data-dir data/a9l_gen_0 --init model/a9l/best_single.h5 \
#       --iter-positions 5000

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

VERSION = "2026-11-22 v1（A9L: 木レコードの兄弟グループに対する listwise 順序損失）"
ARM_TAG = "A9L"


# ============================================================
# 純粋ロジック（torch / gamelibs 不要、self-test の対象）
# ============================================================

def build_file_index(records):
    """1対局分のレコード列から {局面キー: s の平均} を作る。
    戻り値: (index, 重複レコード数)。"""
    sums = {}
    counts = {}
    for rec in records:
        key = rec["p"]
        s = rec["s"]
        if key in sums:
            sums[key] += s
            counts[key] += 1
        else:
            sums[key] = s
            counts[key] = 1
    index = {k: sums[k] / counts[k] for k in sums}
    return index, len(records) - len(index)


def teacher_distribution(t_values, tau):
    """教師値（指す側視点）のリストから softmax(t / tau) を返す（numpy）。"""
    t = np.asarray(t_values, dtype=np.float64) / tau
    t = t - t.max()
    e = np.exp(t)
    return e / e.sum()


def listwise_ce(t_values, u_values, tau):
    """教師分布（t から作る）とモデルスコア u の交差エントロピー（参照実装）。
    学習側は torch で同じ式を計算する。self-test と analyze の検証用。"""
    q = teacher_distribution(t_values, tau)
    u = np.asarray(u_values, dtype=np.float64) / tau
    u = u - u.max()
    log_p = u - np.log(np.exp(u).sum())
    return float(-(q * log_p).sum())


def build_groups(file_indexes, children_of, seed, quota, scan_cap):
    """兄弟グループを再構成する。

    file_indexes: 対局ファイルごとの {局面キー: s}。
    children_of(key): 局面の全合法手の子局面キーのリストを返す関数
        （gamelibs 実装は make_children_of、self-test では合成関数）。
    候補となる親を seed でシャッフルして走査し、同じファイル内に記録の
    ある子が2つ以上ある親をグループにする。学習局面数（親∪子の相異なる
    キー数）が quota に達したら止める。scan_cap は走査する親の上限。
    戻り値: (groups, stats)。group = {"file", "parent", "parent_s",
    "children": [(key, s), ...]}。"""
    candidates = [(fi, key) for fi, index in enumerate(file_indexes)
                  for key in index]
    random.Random(seed).shuffle(candidates)
    groups = []
    stats = {"parents_scanned": 0, "groups": 0, "children": 0,
             "positions": 0, "sizes": [], "files": set()}
    claimed_children = set()
    counted = set()
    for fi, parent in candidates:
        if stats["positions"] >= quota or stats["parents_scanned"] >= scan_cap:
            break
        stats["parents_scanned"] += 1
        index = file_indexes[fi]
        kids = []
        seen = set()
        for ck in children_of(parent):
            if ck in seen or ck in claimed_children or ck not in index:
                continue
            seen.add(ck)
            kids.append((ck, index[ck]))
        if len(kids) < 2:
            continue
        claimed_children.update(seen)
        groups.append({"file": fi, "parent": parent,
                       "parent_s": index[parent], "children": kids})
        stats["groups"] += 1
        stats["children"] += len(kids)
        stats["sizes"].append(len(kids))
        stats["files"].add(fi)
        counted.add(parent)
        counted.update(seen)
        stats["positions"] = len(counted)
    return groups, stats


def select_topup(file_indexes, used_keys, need, seed):
    """グループで不足した分を、全ファイルの記録済み局面から一様に補充する。
    戻り値: [(key, s), ...]（used_keys と重ならないもの）。"""
    if need <= 0:
        return []
    pool = {}
    for index in file_indexes:
        for key, s in index.items():
            if key not in used_keys and key not in pool:
                pool[key] = s
    keys = sorted(pool)
    random.Random(seed).shuffle(keys)
    return [(k, pool[k]) for k in keys[:need]]


def plan_iteration(file_indexes, children_of, state, budget_positions,
                   iter_positions, seed, scan_cap):
    """1イテレーション分の学習対象（グループ＋補充）を予算内で決める。
    戻り値: dict(groups, topup, positions, bound, stats)。
    bound は当たった上限: "iter" / "positions" / "exhausted" / "data"。"""
    remain = budget_positions - state["positions"]
    if remain <= 0:
        return {"groups": [], "topup": [], "positions": 0,
                "bound": "exhausted", "stats": None}
    quota_binds = 0 < iter_positions < remain
    cap = iter_positions if quota_binds else remain
    groups, stats = build_groups(file_indexes, children_of, seed, cap,
                                 scan_cap)
    used = set()
    for g in groups:
        used.add(g["parent"])
        for ck, _s in g["children"]:
            used.add(ck)
    topup = select_topup(file_indexes, used, cap - len(used), seed + 1)
    positions = len(used) + len(topup)
    if positions >= cap:
        bound = "iter" if quota_binds else "positions"
    else:
        bound = "data"
    files = set(stats["files"]) if stats else set()
    return {"groups": groups, "topup": topup, "positions": positions,
            "bound": bound, "stats": stats, "files": files}


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'OK' if cond else 'NG'}: {name}")
        ok = ok and cond

    # 重複レコードは s を平均する
    idx, ndup = build_file_index(
        [{"p": 1, "s": 0.2}, {"p": 1, "s": 0.4}, {"p": 2, "s": -0.1}])
    check("同一局面の重複は s を平均", abs(idx[1] - 0.3) < 1e-12 and ndup == 1)

    # 教師分布: 合計1で、値の開いた教師ほど鋭い
    q_flat = teacher_distribution([0.0, 0.0, 0.0], 0.1)
    q_sharp = teacher_distribution([0.5, 0.0, -0.5], 0.1)
    check("教師分布の合計は1", abs(q_flat.sum() - 1.0) < 1e-9
          and abs(q_sharp.sum() - 1.0) < 1e-9)
    check("開いた教師ほど分布が鋭い", q_sharp.max() > q_flat.max())

    # CE: 教師と一致する順序は逆順より小さい
    t = [0.5, 0.1, -0.3]
    ce_good = listwise_ce(t, t, 0.1)
    ce_bad = listwise_ce(t, list(reversed(t)), 0.1)
    check("CE は正しい順序で小さい", 0.0 <= ce_good < ce_bad)

    # グループ再構成: 子が2つ以上記録にある親だけがグループになる
    files = [{1: 0.1, 2: 0.5, 3: -0.2, 4: 0.3, 6: 0.0, 7: 0.2}]
    cmap = {1: [2, 3, 4], 2: [3], 6: [3, 4], 7: [2, 99]}

    def children_of(k):
        return cmap.get(k, [])

    groups, stats = build_groups(files, children_of, seed=0, quota=100,
                                 scan_cap=1000)
    parents = {g["parent"] for g in groups}
    check("子1つだけの親はグループにならない", 2 not in parents)
    check("記録に無い子は除かれる", all(
        ck in files[0] for g in groups for ck, _ in g["children"]))
    all_kids = [ck for g in groups for ck, _ in g["children"]]
    check("子は二重に使われない", len(all_kids) == len(set(all_kids)))

    # quota: 局面数（親∪子）が quota に達したら止まる
    groups2, stats2 = build_groups(files, children_of, seed=0, quota=3,
                                   scan_cap=1000)
    check("quota で止まる", stats2["positions"] >= 3
          and sum(1 + len(g["children"]) for g in groups2) >= stats2["positions"])

    # 同じ seed で同じ結果
    ga, _ = build_groups(files, children_of, seed=42, quota=100, scan_cap=1000)
    gb, _ = build_groups(files, children_of, seed=42, quota=100, scan_cap=1000)
    check("同じ seed で同じグループ", [g["parent"] for g in ga]
          == [g["parent"] for g in gb])

    # 補充: 使用済みキーを避けて必要数を返す
    top = select_topup(files, {2, 3}, 2, seed=0)
    check("補充は使用済みを避ける", len(top) == 2
          and all(k not in (2, 3) for k, _ in top))

    # 予算: 使い切っていたら exhausted、iter 上限が先に効く
    plan = plan_iteration(files, children_of, {"positions": 100000},
                          100000, 5000, seed=0, scan_cap=1000)
    check("予算を使い切ったら exhausted", plan["bound"] == "exhausted"
          and plan["positions"] == 0)
    plan2 = plan_iteration(files, children_of, {"positions": 0},
                           100000, 4, seed=0, scan_cap=1000)
    check("iter 上限に収まる", plan2["positions"] <= 4 + 3
          and plan2["bound"] in ("iter", "data"))

    print("self-test:", "OK" if ok else "NG")
    return ok


# ============================================================
# データ読み込み・状態（train_arm.py と同じ契約）
# ============================================================

def load_games(data_dir):
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
# gamelibs による子局面の列挙
# ============================================================

def make_children_of():
    """局面キー → 全合法手の子局面キーのリストを返す関数を作る。
    MoveList は len() + インデックスでアクセスする（イテレーションは
    segfault する。audit_order.py の注意と同じ）。"""
    from game import from_hash

    def children_of(key):
        try:
            state = from_hash(key)
            if state.is_win() or state.is_lose():
                return []
            ml = state.legal_actions()
            out = []
            for i in range(len(ml)):
                out.append(state.next(ml[i]).hash_key())
            return out
        except Exception:
            return []

    return children_of


# ============================================================
# モデルの保存・評価（train_arm.py と同一の契約）
# ============================================================

def build_model(init, seed):
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


# ============================================================
# グループ再構成の事前確認（学習しない）
# ============================================================

def analyze(args):
    games, n_skipped = load_games(args.data_dir)
    if not games:
        print(f"** {args.data_dir} に対局データがありません **")
        return False
    file_indexes = []
    total_records = 0
    total_dup = 0
    for _name, records in games:
        index, ndup = build_file_index(records)
        file_indexes.append(index)
        total_records += len(records)
        total_dup += ndup
    distinct = sum(len(ix) for ix in file_indexes)
    print(f"# A9L グループ再構成の事前確認（{args.data_dir}）")
    print(f"- 対局ファイル: {len(games):,}（読めなかった・空: {n_skipped:,}）")
    print(f"- レコード: {total_records:,} / 相異なる局面（ファイル内）: {distinct:,}"
          f" / ファイル内重複: {total_dup:,}（{100.0 * total_dup / max(total_records, 1):.1f}%）")

    children_of = make_children_of()
    cap = args.analyze_parents
    # analyze では file 順に決定的に走査する（build_groups は seed で
    # シャッフルするので、ここでは quota を大きくして走査上限で止める）
    groups, stats = build_groups(file_indexes, children_of, seed=0,
                                 quota=10 ** 12, scan_cap=cap)
    scanned = stats["parents_scanned"]
    sizes = stats["sizes"]
    print(f"- 走査した親: {scanned:,} / グループ: {stats['groups']:,}"
          f"（親あたり {stats['groups'] / max(scanned, 1):.2f}）")
    if sizes:
        arr = np.array(sizes)
        print(f"- グループの子数: 平均 {arr.mean():.2f} / 最大 {arr.max()}"
              f" / 2子 {int((arr == 2).sum()):,} / 3子 {int((arr == 3).sum()):,}"
              f" / 4子以上 {int((arr >= 4).sum()):,}")
        per_group_pos = 1.0 + arr.mean()
        print(f"- グループあたり学習局面数（親＋子）: {per_group_pos:.2f}")
        need_groups = 5000 / per_group_pos
        need_parents = need_groups / max(stats["groups"] / max(scanned, 1), 1e-9)
        print(f"- 5,000 局面をグループで満たすのに必要な親の走査数（推定）: "
              f"{need_parents:,.0f}")
    print(f"- 走査範囲で子としてカバーされた局面: {stats['children']:,}")
    return True


# ============================================================
# 学習
# ============================================================

def train(args):
    import numpy as np
    import torch
    import torch.optim as optim

    from game import from_hash
    from single_network import DN_INPUT_SHAPE

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(model_dir)
    if state["arm"] is not None and state["arm"] != ARM_TAG:
        print(f"** arm_state.json の arm={state['arm']} と {ARM_TAG} が"
              "食い違うため中止（model-dir の取り違え防止）**")
        return False
    state["arm"] = ARM_TAG

    log_lines = []

    def log(line=""):
        log_lines.append(line)
        print(line)

    log(f"## train_rank ({ARM_TAG}): iteration {state['iterations']}"
        f"（{time.strftime('%Y-%m-%d %H:%M:%S')}）")
    log()

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

    # ---- データ準備: 読み込み → 局面索引 → グループ再構成 → 補充 ----
    games, n_skipped = load_games(args.data_dir)
    total_positions = sum(len(records) for _, records in games)
    log(f"- データ: `{args.data_dir}` から {len(games):,} 局 / {total_positions:,} 局面"
        f"（読めなかった・空: {n_skipped:,} ファイル）")
    if not games:
        log("** 対局が0件のため中止 **")
        with (model_dir / "train_log.md").open("a", encoding="utf-8") as fh:
            fh.write("\n".join(log_lines) + "\n\n")
        return False
    log("- 引き分け対局の除外: 木学習のため学習側では判定不能（arm 2 と同じ）")

    file_indexes = []
    total_dup = 0
    for _name, records in games:
        index, ndup = build_file_index(records)
        file_indexes.append(index)
        total_dup += ndup
    log(f"- ファイル内重複レコード: {total_dup:,}（s は平均して1局面1ラベルにした）")

    children_of = make_children_of()
    plan = plan_iteration(file_indexes, children_of, state,
                          args.budget_positions, args.iter_positions,
                          args.seed + state["iterations"], args.scan_cap)
    if plan["bound"] == "exhausted":
        log("** 累計予算を使い切っているため中止（arm_state.json を確認）**")
        with (model_dir / "train_log.md").open("a", encoding="utf-8") as fh:
            fh.write("\n".join(log_lines) + "\n\n")
        return False
    stats = plan["stats"]
    bound_name = {"iter": "イテレーション上限", "positions": "局面数上限",
                  "data": "グループ＋補充でも上限に届かなかった"}[plan["bound"]]
    sizes = np.array(stats["sizes"]) if stats and stats["sizes"] else np.array([0])
    log(f"- グループ再構成: 親 {stats['parents_scanned']:,} を走査して "
        f"{stats['groups']:,} グループ（子数 平均 {sizes.mean():.2f} / 最大 {sizes.max()}）")
    log(f"- 学習局面: {plan['positions']:,}（グループ由来 "
        f"{plan['positions'] - len(plan['topup']):,} ＋ 補充 {len(plan['topup']):,}）"
        f"（上限に当たったのは {bound_name}）")
    log(f"- 順序損失: rank_weight={args.rank_weight} / tau={args.tau}"
        f" / lam(MSE)={args.lam}")

    # ---- 学習行の構築（局面キー → 行番号、特徴量は1回だけ作る） ----
    row_key = []
    row_s = []
    row_of = {}

    def add_row(key, s):
        if key not in row_of:
            row_of[key] = len(row_key)
            row_key.append(key)
            row_s.append(s)
        return row_of[key]

    items = []  # (親の行, [子の行]) — 補充は子の無い item として混ぜる
    for g in plan["groups"]:
        prow = add_row(g["parent"], g["parent_s"])
        crows = [add_row(ck, cs) for ck, cs in g["children"]]
        items.append((prow, crows))
    for key, s in plan["topup"]:
        items.append((add_row(key, s), []))
    n_rows = len(row_key)
    log(f"- 学習行: {n_rows:,} 局面 / アイテム {len(items):,} 件"
        f"（うち順序グループ {len(plan['groups']):,}）")

    file_dim, rank_dim, channel = DN_INPUT_SHAPE
    feats = np.zeros((n_rows, channel, file_dim, rank_dim), dtype=np.float32)
    for i, key in enumerate(row_key):
        st = from_hash(key)
        feats[i] = np.array(st.feature()).reshape(channel, file_dim, rank_dim)
    s_arr = np.array(row_s, dtype=np.float32)

    # ---- 学習ループ（オプティマイザは毎回新規。train_arm.py と同条件） ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.init, args.seed).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.00001)
    x_all = torch.from_numpy(feats).to(device)
    s_all = torch.from_numpy(s_arr).to(device)

    rng = random.Random(args.seed + 1000 + state["iterations"])
    start = time.time()
    avg_loss = None
    for epoch in range(args.epochs):
        order = list(range(len(items)))
        rng.shuffle(order)
        sum_loss = 0.0
        sum_mse = 0.0
        sum_ce = 0.0
        n_batch = 0
        n_groups_seen = 0
        for b0 in range(0, len(order), args.group_batch):
            batch_items = [items[j] for j in order[b0:b0 + args.group_batch]]
            rows = []
            seen = set()
            for prow, crows in batch_items:
                for r in [prow] + crows:
                    if r not in seen:
                        seen.add(r)
                        rows.append(r)
            rows_t = torch.tensor(rows, dtype=torch.long, device=device)
            row_pos = {r: k for k, r in enumerate(rows)}
            model.train()
            optimizer.zero_grad()
            outputs = model(x_all[rows_t]).squeeze(-1)
            loss = args.lam * torch.sum((outputs - s_all[rows_t]) ** 2)
            mse_mean = torch.mean((outputs - s_all[rows_t]) ** 2).item()
            ce_sum = 0.0
            for prow, crows in batch_items:
                if len(crows) < 2:
                    continue
                cidx = torch.tensor([row_pos[r] for r in crows],
                                    dtype=torch.long, device=device)
                u = -outputs[cidx]
                t = -s_all[torch.tensor(crows, dtype=torch.long, device=device)]
                with torch.no_grad():
                    q = torch.softmax(t / args.tau, dim=0)
                log_p = torch.log_softmax(u / args.tau, dim=0)
                ce = -(q * log_p).sum()
                loss = loss + args.rank_weight * len(crows) * ce
                ce_sum += ce.item()
                n_groups_seen += 1
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            sum_mse += mse_mean
            sum_ce += ce_sum
            n_batch += 1
        avg_loss = sum_loss / max(n_batch, 1)
        print(f"epoch:{epoch} avg loss:{avg_loss:.4f} "
              f"mse/pos:{sum_mse / max(n_batch, 1):.4f} "
              f"ce/group:{sum_ce / max(n_groups_seen, 1):.4f} "
              f"sec:{int(time.time() - start)}")

    # ---- 保存・評価・状態の更新 ----
    h5, onnx, jit = save_model_files(model_dir, model)
    log(f"- 保存: {h5} / {onnx} / {jit}（最終 epoch の平均損失 {avg_loss:.4f}）")

    ans = evaluate_ans(model_dir)
    if ans is not None:
        log(f"- ans（参考）: 全体 {ans['ans_all']:.4f}（{ans['n_all']:,} 局面） / "
            f"win/lose のみ {ans['ans_winlose']:.4f}（{ans['n_winlose']:,} 局面）")
    else:
        log("- ans: ../oracle/problem.txt が無いため未実施")

    touched_files = set()
    for g in plan["groups"]:
        touched_files.add(g["file"])
    state["games"] += len(touched_files)
    state["positions"] += plan["positions"]
    state["iterations"] += 1
    save_state(model_dir, state)
    log(f"- 累計: {state['games']:,} 局（触れた対局の累計） / "
        f"{state['positions']:,} 局面"
        f"（予算 {args.budget_games:,} 局 / {args.budget_positions:,} 局面）")
    log()

    with (model_dir / "train_log.md").open("a", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="実験5 段階A A9L: 木レコードの兄弟ランキング学習")
    parser.add_argument("--model-dir", default=None,
                        help="モデル出力先（例: model/a9l）")
    parser.add_argument("--data-dir", default=None,
                        help="このイテレーションの対局データ（*.json のあるディレクトリ）")
    parser.add_argument("--init", default=None,
                        help="初期重み: state_dict のパス、または random")
    parser.add_argument("--init-only", action="store_true",
                        help="乱数初期モデルを作って保存するだけ（学習しない）")
    parser.add_argument("--analyze", action="store_true",
                        help="学習せず、兄弟グループの再構成だけを測定する")
    parser.add_argument("--analyze-parents", type=int, default=30000,
                        help="--analyze で走査する親の上限")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lam", type=float, default=0.5,
                        help="損失の MSE(s) 項の重み（train_arm.py の LAMBDA）")
    parser.add_argument("--rank-weight", type=float, default=0.15,
                        help="順序損失の重み（子の数 × CE の和にかける）")
    parser.add_argument("--tau", type=float, default=0.1,
                        help="教師分布・モデル分布の softmax 温度")
    parser.add_argument("--group-batch", type=int, default=24,
                        help="1バッチのグループ数")
    parser.add_argument("--budget-games", type=int, default=5000)
    parser.add_argument("--budget-positions", type=int, default=100000)
    parser.add_argument("--iter-positions", type=int, default=5000,
                        help="1イテレーションで学習する局面数の上限")
    parser.add_argument("--scan-cap", type=int, default=120000,
                        help="グループ再構成で走査する親の上限")
    args = parser.parse_args()

    print(f"train_rank.py version {VERSION}")
    if not self_test():
        print("** self-test 失敗のため中止 **")
        sys.exit(1)
    if args.analyze:
        if args.data_dir is None:
            print("** --analyze には --data-dir が必要です **")
            sys.exit(1)
        if not analyze(args):
            sys.exit(1)
        return
    if args.model_dir is None:
        print("** --model-dir が必要です **")
        sys.exit(1)
    if not train(args):
        sys.exit(1)


if __name__ == "__main__":
    main()
