import numpy as np
import json
import glob
import torch
from single_network import *
from game import *

DATA_DIR = "../data"

def load_oracle_keys(path, count):
    with open(path, "rb") as f:
        return np.fromfile(f, dtype=np.uint64, count=count)

def batch_predict(model, states, device, batch_size=4096):
    file, rank, channel = DN_INPUT_SHAPE
    all_scores = []
    for i in range(0, len(states), batch_size):
        batch = states[i:i+batch_size]
        features = []
        for state in batch:
            feat = np.array(state.feature()).reshape(channel, file, rank)
            features.append(feat)
        x = torch.from_numpy(np.array(features)).to(device).float()
        with torch.no_grad():
            out = model(x).squeeze(-1)
        all_scores.append(out.cpu().numpy())
    return np.concatenate(all_scores)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SingleNet()
    model.load_state_dict(torch.load('./model/best_single.h5', weights_only=True))
    model = model.to(device)
    model.eval()

    # 1. オラクルwin局面のうち、逆判定されるものを特定
    WIN_COUNT = 196773087
    print("win.datを読み込み中...")
    keys = load_oracle_keys(f"{DATA_DIR}/win.dat", WIN_COUNT)
    rng = np.random.default_rng(42)
    indices = rng.choice(WIN_COUNT, 100000, replace=False)
    sampled_keys = keys[indices]

    # 有効局面のハッシュとスコアを取得
    valid_keys = []
    valid_states = []
    for k in sampled_keys:
        state = from_hash(int(k))
        if state.pos.is_win() or state.pos.is_lose():
            continue
        valid_keys.append(int(k))
        valid_states.append(state)

    scores = batch_predict(model, valid_states, device)
    reverse_mask = scores < -0.2
    reverse_keys = set(valid_keys[i] for i in range(len(valid_keys)) if reverse_mask[i])
    print(f"逆判定局面: {len(reverse_keys):,}局面")

    # 正解局面のキーもサンプル
    correct_mask = scores > 0.2
    correct_keys_list = [valid_keys[i] for i in range(len(valid_keys)) if correct_mask[i]]
    correct_keys = set(correct_keys_list[:len(reverse_keys)])
    print(f"正解局面(比較用): {len(correct_keys):,}局面")

    # 2. 自己対局データを読んで、これらの局面が含まれるか確認
    print("\n自己対局データを読み込み中...")
    json_files = sorted(glob.glob('data/**/*.json', recursive=True))  # gen_N/ 配下も対象（実験4の世代別蓄積対応）
    print(f"ファイル数: {len(json_files)}")

    total_entries = 0
    reverse_found = []  # (key, search_score, game_result)
    correct_found = []

    for jf in json_files:
        with open(jf) as fh:
            try:
                data = json.load(fh)
            except:
                continue
        for entry in data:
            total_entries += 1
            p = entry["p"]
            if p in reverse_keys:
                reverse_found.append({
                    "key": p,
                    "search_score": entry["s"],
                    "game_result": entry["r"],
                })
            elif p in correct_keys:
                correct_found.append({
                    "key": p,
                    "search_score": entry["s"],
                    "game_result": entry["r"],
                })

    print(f"総エントリ数: {total_entries:,}")
    print(f"\n逆判定局面のヒット: {len(reverse_found)} / {len(reverse_keys)}")
    print(f"正解局面のヒット: {len(correct_found)} / {len(correct_keys)}")

    # 3. ヒットした局面の教師信号を分析
    if reverse_found:
        print(f"\n{'='*60}")
        print(f"逆判定局面の教師信号 ({len(reverse_found)}件)")
        print(f"これらは正解=winだがモデルが「負け」と評価する局面")
        print(f"{'='*60}")
        ss = [e["search_score"] for e in reverse_found]
        gr = [e["game_result"] for e in reverse_found]
        print(f"  search_score: 平均={np.mean(ss):.4f} 中央値={np.median(ss):.4f}")
        print(f"  game_result:  平均={np.mean(gr):.4f} 中央値={np.median(gr):.4f}")
        print(f"  game_result分布:")
        for val in [-1.0, 0.0, 1.0]:
            cnt = sum(1 for g in gr if g == val)
            print(f"    {val:+.1f}: {cnt} ({100*cnt/len(gr):.1f}%)")

        print(f"\n  search_score分布:")
        for lo, hi in [(-1.0,-0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0)]:
            cnt = sum(1 for s in ss if lo <= s < hi)
            print(f"    [{lo:+.1f}, {hi:+.1f}): {cnt}")

    if correct_found:
        print(f"\n{'='*60}")
        print(f"正解局面の教師信号 ({len(correct_found)}件, 比較用)")
        print(f"{'='*60}")
        ss = [e["search_score"] for e in correct_found]
        gr = [e["game_result"] for e in correct_found]
        print(f"  search_score: 平均={np.mean(ss):.4f} 中央値={np.median(ss):.4f}")
        print(f"  game_result:  平均={np.mean(gr):.4f} 中央値={np.median(gr):.4f}")
        print(f"  game_result分布:")
        for val in [-1.0, 0.0, 1.0]:
            cnt = sum(1 for g in gr if g == val)
            print(f"    {val:+.1f}: {cnt} ({100*cnt/len(gr):.1f}%)")

    if not reverse_found:
        print("\n★ 逆判定局面は自己対局データに1件も含まれていません")
        print("  → モデルがこれらの局面を学習する機会がそもそもない")

if __name__ == "__main__":
    main()
