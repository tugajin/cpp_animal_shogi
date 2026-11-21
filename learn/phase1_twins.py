# フェーズ1 対称ツイン診断（phase1_twins.py）
# ==============================================
# 問い: モデルの値は、対称な局面同士でどれだけ食い違うか。
#   - 回転ツイン（180度回転+色交換+手番反転）: 特徴量は手番で正規化されるため、
#     特徴量の実装が正しければ値は原理的に一致するはず。一致しなければ
#     特徴量か変換のバグであり、対称拡張以前の問題として切り分ける（パリティ検査）。
#   - 鏡像ツイン（左右反転）: 正規化では消えない非対称性がここに出る。
#     この不一致が兄弟手間の値差と同程度以上なら、鏡像拡張に根拠ができる。
#
# 実行（learn/ ディレクトリで）:
#   ../.venv/bin/python phase1_twins.py
# 出力: phase1_twins_result.md

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from check_unregistered import decode_key, encode_key, self_test
from check_unregistered2 import mirror_key, SWAP

md_lines = []


def md(line=""):
    md_lines.append(line)
    print(line)


def pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


def rotate_twin_key(key):
    """180度回転+色交換+手番反転した等価局面のキー（手番側視点の値は保存される）"""
    turn_is_white, pieces, hand = decode_key(key)
    rotated = [0] * 12
    for idx, code in enumerate(pieces):
        rotated[11 - idx] = SWAP[code]
    new_hand = {}
    for (color, kind), n in hand.items():
        new_hand[("W" if color == "B" else "B", kind)] = n
    return encode_key(not turn_is_white, rotated, new_hand)


def legal_moves_list(state):
    ml = state.legal_actions()
    return [ml[i] for i in range(len(ml))]


def main():
    parser = argparse.ArgumentParser(description="フェーズ1 対称ツイン診断")
    parser.add_argument("--model", default="./model/best_single.h5")
    parser.add_argument("--out", default="phase1_twins_result.md")
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--group", default="data")
    args = parser.parse_args()

    md("# フェーズ1 対称ツイン診断（phase1_twins.py）")
    md()
    if not self_test():
        md("**self-test 失敗のため中止**")
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md_lines) + "\n")
        return

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
    if len(entries) > args.sample * 2:
        idx = rng.choice(len(entries), args.sample * 2, replace=False)
        entries = [entries[i] for i in idx]

    from game import from_hash
    keys = []
    for e in entries:
        if len(keys) >= args.sample:
            break
        k = int(e["p"])
        st = from_hash(k)
        if st.pos.is_win() or st.pos.is_lose():
            continue
        keys.append(k)
    md(f"- 対象: {len(keys):,} 局面（グループ `{args.group}`、seed=123）")
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

    orig_states = [from_hash(k) for k in keys]
    mirror_states = [from_hash(mirror_key(k)) for k in keys]
    rotate_states = [from_hash(rotate_twin_key(k)) for k in keys]
    v_orig = batch_predict(orig_states)
    v_mirror = batch_predict(mirror_states)
    v_rotate = batch_predict(rotate_states)

    d_rot = np.abs(v_orig - v_rotate)
    d_mir = np.abs(v_orig - v_mirror)
    md("## 回転ツイン（パリティ検査）")
    md()
    md(f"- |v(原形) − v(回転形)|: 平均 {d_rot.mean():.6f} / 最大 {d_rot.max():.6f}")
    md("- ほぼ 0（1e-3 以下）なら特徴量の手番正規化は正しく、回転形は学習データとして安全に使える。大きな値が出る場合は特徴量か変換のバグを先に直すこと")
    md()
    md("## 鏡像ツイン（非対称性の大きさ）")
    md()
    md(f"- |v(原形) − v(鏡像)|: 平均 {d_mir.mean():.4f} / 中央値 {np.median(d_mir):.4f} / 0.2 超の割合 {pct(int((d_mir > 0.2).sum()), len(d_mir)):.1f}%")
    md()

    # 兄弟手間の値差（物差し）
    all_children = []
    metas = []
    for st in orig_states:
        actions = legal_moves_list(st)
        children = [st.next(a) for a in actions]
        metas.append(children)
        all_children.extend(children)
    child_vals = batch_predict(all_children) if all_children else np.array([])
    ofs = 0
    pair_gaps = []
    top_gaps = []
    for children in metas:
        n = len(children)
        vals = -child_vals[ofs:ofs + n]  # 指す側視点
        ofs += n
        if n >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    pair_gaps.append(abs(float(vals[i]) - float(vals[j])))
            srt = np.sort(vals)[::-1]
            top_gaps.append(float(srt[0] - srt[1]))
    md("## 物差し: 兄弟手間の値差")
    md()
    md(f"- 兄弟ペアの |Δv| 平均: {np.mean(pair_gaps):.4f}（{len(pair_gaps):,} ペア）")
    md(f"- 最善手と2番手の差の平均: {np.mean(top_gaps):.4f}")
    md()
    md("読み方: 鏡像の不一致が兄弟ペアの |Δv| と同程度以上なら、鏡像の非対称性は手の選択を左右し得る大きさであり、鏡像拡張 arm に根拠ができる。不一致が十分小さければ、非対称性は実害のない範囲で、拡張は棚上げのままにする。")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"\n結果を {args.out} に書きました")


if __name__ == "__main__":
    main()
