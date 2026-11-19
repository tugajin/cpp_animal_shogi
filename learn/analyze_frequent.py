import numpy as np
import json
import glob
import torch
from single_network import *
from game import *

DATA_DIR = "../data"

def load_oracle_sorted(path, count):
    with open(path, "rb") as f:
        data = np.fromfile(f, dtype=np.uint64, count=count)
    data.sort()
    return data

def in_sorted(arr, val):
    idx = np.searchsorted(arr, val)
    return idx < len(arr) and arr[idx] == val

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SingleNet()
    model.load_state_dict(torch.load('./model/best_single.h5', weights_only=True))
    model = model.to(device)
    model.eval()

    print("オラクル読み込み中...")
    win_arr = load_oracle_sorted(f"{DATA_DIR}/win.dat", 196773087)
    print(f"  win: {len(win_arr):,}")
    lose_arr = load_oracle_sorted(f"{DATA_DIR}/lose.dat", 47347380)
    print(f"  lose: {len(lose_arr):,}")
    draw_arr = load_oracle_sorted(f"{DATA_DIR}/draw.dat", 2682700)
    print(f"  draw: {len(draw_arr):,}")

    def oracle_result(key_int):
        k = np.uint64(key_int)
        if in_sorted(win_arr, k): return 1
        if in_sorted(lose_arr, k): return -1
        if in_sorted(draw_arr, k): return 0
        return None

    # 自己対局データを読む
    print("\n自己対局データ読み込み中...")
    json_files = sorted(glob.glob('data/*.json'))
    entries = []
    for jf in json_files:
        with open(jf) as fh:
            try:
                data = json.load(fh)
            except:
                continue
            entries.extend(data)
    print(f"エントリ数: {len(entries):,}")

    # 各エントリについてオラクル判定 + モデル評価
    file, rank, channel = DN_INPUT_SHAPE
    batch_states = []
    batch_meta = []

    unknown_count = 0
    for entry in entries:
        k = entry["p"]
        state = from_hash(k)
        if state.pos.is_win() or state.pos.is_lose():
            continue
        oracle = oracle_result(k)
        if oracle is None:
            # ミラーキーも試す（hash_keyで再計算）
            mirror_key = gamelibs.hash_key(state.pos)
            if mirror_key != k:
                oracle = oracle_result(mirror_key)
        batch_states.append(state)
        batch_meta.append((oracle, entry["s"], entry["r"]))

    # バッチ推論
    all_scores = []
    for i in range(0, len(batch_states), 4096):
        batch = batch_states[i:i+4096]
        features = []
        for state in batch:
            feat = np.array(state.feature()).reshape(channel, file, rank)
            features.append(feat)
        x = torch.from_numpy(np.array(features)).to(device).float()
        with torch.no_grad():
            out = model(x).squeeze(-1)
        all_scores.extend(out.cpu().numpy().tolist())

    results = []
    for i, score in enumerate(all_scores):
        oracle, ss, gr = batch_meta[i]
        results.append((score, oracle, ss, gr))

    print(f"評価済み局面: {len(results):,}")

    # オラクル判定別の集計
    oracle_groups = {}
    for model_score, oracle, ss, gr in results:
        if oracle not in oracle_groups:
            oracle_groups[oracle] = []
        oracle_groups[oracle].append((model_score, ss, gr))

    print(f"\n{'='*60}")
    print("自己対局局面のオラクル分類")
    print(f"{'='*60}")
    for label in [1, -1, 0, None]:
        name = {1: "win", -1: "lose", 0: "draw", None: "unknown"}[label]
        if label in oracle_groups:
            n = len(oracle_groups[label])
            print(f"  {name:8s}: {n:6d} ({100*n/len(results):.1f}%)")

    # 各カテゴリでのモデル精度
    for label in [1, -1, 0]:
        if label not in oracle_groups:
            continue
        name = {1: "win", -1: "lose", 0: "draw"}[label]
        data = oracle_groups[label]
        n = len(data)
        if n == 0:
            continue

        scores = np.array([d[0] for d in data])
        search_scores = np.array([d[1] for d in data])
        game_results = np.array([d[2] for d in data])

        if label == 1:
            correct = (scores > 0.2).sum()
            reverse = (scores < -0.2).sum()
        elif label == -1:
            correct = (scores < -0.2).sum()
            reverse = (scores > 0.2).sum()
        else:
            correct = ((scores >= -0.2) & (scores <= 0.2)).sum()
            reverse = ((scores > 0.2) | (scores < -0.2)).sum()
        undecided = n - correct - reverse

        print(f"\n--- {name}局面 ({n}件) ---")
        print(f"  モデル正解率: {correct}/{n} ({100*correct/n:.1f}%)")
        if reverse > 0:
            print(f"  逆判定: {reverse} ({100*reverse/n:.1f}%)")
        print(f"  未決: {undecided} ({100*undecided/n:.1f}%)")

        # search_scoreの精度
        if label == 1:
            ss_correct = (search_scores > 0.2).sum()
        elif label == -1:
            ss_correct = (search_scores < -0.2).sum()
        else:
            ss_correct = ((search_scores >= -0.2) & (search_scores <= 0.2)).sum()
        print(f"  search_score正解率: {ss_correct}/{n} ({100*ss_correct/n:.1f}%)")

        # game_resultの精度
        if label == 1:
            gr_correct = (game_results > 0).sum()
        elif label == -1:
            gr_correct = (game_results < 0).sum()
        else:
            gr_correct = (game_results == 0).sum()
        print(f"  game_result正解率: {gr_correct}/{n} ({100*gr_correct/n:.1f}%)")

        # スコア統計
        print(f"  モデルスコア: 平均={scores.mean():.3f} 中央値={np.median(scores):.3f}")
        print(f"  search_score: 平均={search_scores.mean():.3f} 中央値={np.median(search_scores):.3f}")
        print(f"  game_result:  平均={game_results.mean():.3f}")

    # unknown局面の分析
    if None in oracle_groups:
        data = oracle_groups[None]
        n = len(data)
        scores = np.array([d[0] for d in data])
        search_scores = np.array([d[1] for d in data])
        game_results = np.array([d[2] for d in data])
        print(f"\n--- unknown局面 ({n}件) ---")
        print(f"  モデルスコア: 平均={scores.mean():.3f} 中央値={np.median(scores):.3f}")
        print(f"  search_score: 平均={search_scores.mean():.3f}")
        print(f"  game_result:  平均={game_results.mean():.3f}")


if __name__ == "__main__":
    main()
