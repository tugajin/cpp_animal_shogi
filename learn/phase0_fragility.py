# フェーズ0 脆弱性層別（phase0_fragility.py）
# ============================================
# 問い: audit_order で見つかった指し手の失敗（win の取りこぼし、draw の保持失敗）は、
# 「最良カテゴリを達成できる手が少ない局面」に集中しているのか。
# soft value 論文（GPW2020）の含意では、局面の壊れやすさ＝維持できる手の数が本質で、
# ここが律速なら、生成や重み付けでその局面型を厚くする設計に根拠ができる。
#
# 方法は audit_order.py と同一（自己対局レコードから対局列を復元し、3,000局面を
# seed=123 でサンプル、全合法手の子局面をオラクルカテゴリとモデル値で評価）。
# 各局面について「最良カテゴリを達成できる子の数（維持手数）」を数え、
# 維持手数 1 / 2 / 3+ で A（argmax 到達）と C（実着手の維持）を層別する。
#
# 実行（learn/ ディレクトリで）:
#   ../.venv/bin/python phase0_fragility.py
# 出力: phase0_fragility_result.md
#
# 注意: gamelibs の MoveList は len()+インデックスでアクセスすること（イテレーションは segfault）。

import argparse
import glob
import json
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
    ml = state.legal_actions()
    return [ml[i] for i in range(len(ml))]


def child_outcome(oracle, child_state):
    key = child_state.hash_key()
    direct, mirror, normalized, normalized_mirror = lookup_variants(oracle, key)
    label = normalized if normalized is not None else normalized_mirror
    if label is not None:
        return -label
    if child_state.is_lose():
        return 1
    if child_state.is_win():
        return -1
    return None


def main():
    parser = argparse.ArgumentParser(description="フェーズ0 脆弱性層別")
    parser.add_argument("--model", default="./model/best_single.h5")
    parser.add_argument("--out", default="phase0_fragility_result.md")
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--group", default="data")
    args = parser.parse_args()

    md("# フェーズ0 脆弱性層別（phase0_fragility.py）")
    md()
    if not self_test():
        md("**self-test 失敗のため中止**")
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md_lines) + "\n")
        return

    oracle = LabelOracle()

    from game import from_hash
    files = sorted(glob.glob("data/**/*.json", recursive=True))
    positions = []
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
            if len(positions) >= 60000:
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
    md(f"- 復元した対局数（推定）: {games:,} / 局面数: {len(positions):,} / 遷移: {transitions:,}")
    md()

    rng = np.random.default_rng(123)
    if len(positions) > args.sample:
        idx = rng.choice(len(positions), args.sample, replace=False)
        positions = [positions[i] for i in idx]

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

    records = []  # dict: pos_cat, preserving, argmax_ok, played_outcome, best
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
        metas.append((key, played_key, children, outcomes))
        all_children.extend(children)
    values_flat = batch_predict(all_children) if all_children else np.array([])
    ofs = 0
    for key, played_key, children, outcomes in metas:
        n = len(children)
        child_vals = values_flat[ofs:ofs + n]
        ofs += n
        mover_vals = -child_vals
        outcomes = np.array(outcomes)
        best = int(outcomes.max())
        preserving = int((outcomes == best).sum())
        argmax_idx = int(np.argmax(mover_vals))
        played_outcome = None
        if played_key is not None:
            for c, o in zip(children, outcomes):
                if c.hash_key() == played_key:
                    played_outcome = int(o)
                    break
        records.append({
            "pos_cat": best,
            "preserving": preserving,
            "argmax_ok": bool(outcomes[argmax_idx] == best),
            "played_outcome": played_outcome,
            "best": best,
        })
    md(f"- 調査できた局面: {len(records):,}（スキップ {skipped:,}）")
    md()

    def bucket(p):
        return "1" if p == 1 else ("2" if p == 2 else "3+")

    def summarize(sub, title):
        if not sub:
            return
        md(f"### {title}（{len(sub):,} 局面）")
        md()
        md("| 維持手数 | 局面数 | A. argmax 到達率 | C. 実着手の維持率 | C. 2段階低下率 |")
        md("|---|---|---|---|---|")
        for b in ("1", "2", "3+"):
            ss = [r for r in sub if bucket(r["preserving"]) == b]
            if not ss:
                continue
            a = pct(sum(1 for r in ss if r["argmax_ok"]), len(ss))
            played = [r for r in ss if r["played_outcome"] is not None]
            if played:
                keep = pct(sum(1 for r in played if r["played_outcome"] == r["best"]), len(played))
                drop2 = pct(sum(1 for r in played if r["best"] - r["played_outcome"] == 2), len(played))
                c_str = f"{keep:.1f}%"
                d_str = f"{drop2:.1f}%"
            else:
                c_str = d_str = "—"
            md(f"| {b} | {len(ss):,} | {a:.1f}% | {c_str} | {d_str} |")
        md()

    md("## 全体")
    md()
    summarize(records, "全局面")
    md("## カテゴリ別（手番側視点）")
    md()
    for cat in (1, 0):
        summarize([r for r in records if r["pos_cat"] == cat], f"{CAT_NAME[cat]} 局面")
    md("（lose 局面は全手が最良カテゴリなので層別の意味がなく省略）")
    md()
    md("## 読み方")
    md()
    md("- 失敗（A の不到達、C の低下）が維持手数=1 の層に集中していれば、律速は「壊れやすい局面」であり、生成・重み付けでその局面型を厚くする根拠になる")
    md("- 層によらず失敗率が平坦なら、特定局面型ではなく全体の値の質の問題")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"\n結果を {args.out} に書きました")


if __name__ == "__main__":
    main()
