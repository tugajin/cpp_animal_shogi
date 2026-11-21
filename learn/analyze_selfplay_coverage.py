import numpy as np
import json
import glob
import torch
from single_network import *
from game import *
from collections import Counter

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

def parse_position(state):
    s = str(state)
    board_str = ""
    lines = s.strip().split('\n')

    sente_hand = 0
    gote_hand = 0
    for line in lines:
        if "先手:持ち駒" in line and "なし" not in line:
            hand_part = line.split("持ち駒")[1] if "持ち駒" in line else ""
            for ch in ["ひ", "き", "ぞ"]:
                sente_hand += hand_part.count(ch)
        elif "後手:持ち駒" in line and "なし" not in line:
            hand_part = line.split("持ち駒")[1] if "持ち駒" in line else ""
            for ch in ["ひ", "き", "ぞ"]:
                gote_hand += hand_part.count(ch)
        elif "手番" not in line and "持ち駒" not in line:
            board_str += line

    piece_map = {
        "^ひ": "s", "^に": "s", "^き": "s", "^ぞ": "s", "^ら": "s",
        "vひ": "g", "vに": "g", "vき": "g", "vぞ": "g", "vら": "g",
    }
    sente_board = sum(board_str.count(k) for k, v in piece_map.items() if v == "s")
    gote_board = sum(board_str.count(k) for k, v in piece_map.items() if v == "g")

    has_niwatori_sente = "^に" in board_str
    has_niwatori_gote = "vに" in board_str

    ml = gamelibs.MoveList()
    gamelibs.legal_moves(state.pos, ml)

    return {
        "sente_board": sente_board,
        "gote_board": gote_board,
        "sente_hand": sente_hand,
        "gote_hand": gote_hand,
        "total_pieces": sente_board + gote_board + sente_hand + gote_hand,
        "legal_moves": ml.len(),
        "in_check": gamelibs.in_checked(state.pos),
        "has_niwatori_sente": has_niwatori_sente,
        "has_niwatori_gote": has_niwatori_gote,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SingleNet()
    model.load_state_dict(torch.load('./model/best_single.h5', weights_only=True))
    model = model.to(device)
    model.eval()

    # オラクルをソート済み配列として読み込み、二分探索（np.searchsorted）で照合する
    # ※ 旧実装は 1.97億件を Python の set 化しており、メモリを大量消費していた。
    #   analyze_frequent.py の load_oracle_sorted / in_sorted と同じ方式に統一。
    print("オラクル読み込み中...")
    win_arr = load_oracle_keys(f"{DATA_DIR}/win.dat", 196773087)
    win_arr.sort()
    lose_arr = load_oracle_keys(f"{DATA_DIR}/lose.dat", 47347380)
    lose_arr.sort()
    draw_arr = load_oracle_keys(f"{DATA_DIR}/draw.dat", 2682700)
    draw_arr.sort()

    def in_sorted(arr, val):
        idx = np.searchsorted(arr, np.uint64(val))
        return idx < len(arr) and arr[idx] == np.uint64(val)

    print(f"  win: {len(win_arr):,}  lose: {len(lose_arr):,}  draw: {len(draw_arr):,}")

    # 自己対局データを読む
    print("\n自己対局データ読み込み中...")
    json_files = sorted(glob.glob('data/**/*.json', recursive=True))  # gen_N/ 配下も対象（実験4の世代別蓄積対応）
    selfplay_entries = []
    for jf in json_files:
        with open(jf) as fh:
            try:
                data = json.load(fh)
            except:
                continue
            for entry in data:
                selfplay_entries.append(entry)
    print(f"  エントリ数: {len(selfplay_entries):,}")

    # 自己対局局面のオラクル判定
    oracle_results = {"win": 0, "lose": 0, "draw": 0, "unknown": 0}
    selfplay_keys = []
    for entry in selfplay_entries:
        k = entry["p"]
        selfplay_keys.append(k)
        if in_sorted(win_arr, k):
            oracle_results["win"] += 1
        elif in_sorted(lose_arr, k):
            oracle_results["lose"] += 1
        elif in_sorted(draw_arr, k):
            oracle_results["draw"] += 1
        else:
            oracle_results["unknown"] += 1

    n = len(selfplay_entries)
    print(f"\n自己対局局面のオラクル分類:")
    for label, cnt in oracle_results.items():
        print(f"  {label:8s}: {cnt:6d} ({100*cnt/n:.1f}%)")

    # 自己対局局面のモデルスコア vs オラクル正解
    print("\n自己対局局面のモデル評価 vs オラクル正解:")
    selfplay_states = []
    selfplay_oracle = []
    selfplay_meta = []
    for entry in selfplay_entries:
        k = entry["p"]
        state = from_hash(k)
        if state.pos.is_win() or state.pos.is_lose():
            continue
        if in_sorted(win_arr, k):
            oracle = 1
        elif in_sorted(lose_arr, k):
            oracle = -1
        elif in_sorted(draw_arr, k):
            oracle = 0
        else:
            oracle = None
        selfplay_states.append(state)
        selfplay_oracle.append(oracle)
        selfplay_meta.append(entry)

    selfplay_scores = batch_predict(model, selfplay_states, device)

    # オラクルwin局面で逆判定されているもの
    sp_win_correct = 0
    sp_win_reverse = 0
    sp_win_undecided = 0
    sp_lose_correct = 0
    sp_lose_reverse = 0
    sp_lose_undecided = 0

    for i, (score, oracle) in enumerate(zip(selfplay_scores, selfplay_oracle)):
        if oracle == 1:
            if score > 0.2: sp_win_correct += 1
            elif score < -0.2: sp_win_reverse += 1
            else: sp_win_undecided += 1
        elif oracle == -1:
            if score < -0.2: sp_lose_correct += 1
            elif score > 0.2: sp_lose_reverse += 1
            else: sp_lose_undecided += 1

    sp_win_total = sp_win_correct + sp_win_reverse + sp_win_undecided
    sp_lose_total = sp_lose_correct + sp_lose_reverse + sp_lose_undecided
    if sp_win_total > 0:
        print(f"\n  自己対局中のオラクルwin局面 ({sp_win_total}件):")
        print(f"    正解: {sp_win_correct} ({100*sp_win_correct/sp_win_total:.1f}%)")
        print(f"    逆判定: {sp_win_reverse} ({100*sp_win_reverse/sp_win_total:.1f}%)")
        print(f"    未決: {sp_win_undecided} ({100*sp_win_undecided/sp_win_total:.1f}%)")
    if sp_lose_total > 0:
        print(f"\n  自己対局中のオラクルlose局面 ({sp_lose_total}件):")
        print(f"    正解: {sp_lose_correct} ({100*sp_lose_correct/sp_lose_total:.1f}%)")
        print(f"    逆判定: {sp_lose_reverse} ({100*sp_lose_reverse/sp_lose_total:.1f}%)")
        print(f"    未決: {sp_lose_undecided} ({100*sp_lose_undecided/sp_lose_total:.1f}%)")

    # 自己対局データ中で逆判定された局面の教師信号
    print(f"\n{'='*60}")
    print("自己対局データ中の逆判定局面の教師信号")
    print(f"{'='*60}")
    for i, (score, oracle) in enumerate(zip(selfplay_scores, selfplay_oracle)):
        if oracle == 1 and score < -0.2:
            entry = selfplay_meta[i]
            ss = entry["s"]
            gr = entry["r"]
            print(f"  key={entry['p']} model={score:.3f} search_score={ss:.3f} game_result={gr:.1f}")

    # 自己対局局面の特徴分析（全体の分布を見る）
    print(f"\n{'='*60}")
    print("自己対局局面の特徴分布")
    print(f"{'='*60}")
    sp_features = []
    for state in selfplay_states[:5000]:
        sp_features.append(parse_position(state))

    # 比較: オラクルwin逆判定局面の特徴分布
    print("\nオラクルwin逆判定局面の特徴を再計算中...")
    WIN_COUNT = 196773087
    all_win_keys = load_oracle_keys(f"{DATA_DIR}/win.dat", WIN_COUNT)
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(WIN_COUNT, 100000, replace=False)
    sampled_win_keys = all_win_keys[sample_indices]

    reverse_states = []
    for k in sampled_win_keys:
        state = from_hash(int(k))
        if state.pos.is_win() or state.pos.is_lose():
            continue
        reverse_states.append(state)

    reverse_scores = batch_predict(model, reverse_states, device)
    reverse_mask = reverse_scores < -0.2
    reverse_pos_states = [reverse_states[i] for i in range(len(reverse_states)) if reverse_mask[i]]

    reverse_features = []
    for state in reverse_pos_states[:3000]:
        reverse_features.append(parse_position(state))

    def print_comparison(label, features):
        n = len(features)
        total = [f["total_pieces"] for f in features]
        sb = [f["sente_board"] for f in features]
        gb = [f["gote_board"] for f in features]
        lm = [f["legal_moves"] for f in features]
        check = sum(1 for f in features if f["in_check"])
        niwatori_s = sum(1 for f in features if f["has_niwatori_sente"])
        niwatori_g = sum(1 for f in features if f["has_niwatori_gote"])
        print(f"\n  {label} ({n}局面):")
        print(f"    総駒数: 平均={np.mean(total):.2f}")
        tc = Counter(total)
        for k in sorted(tc.keys()):
            pct = 100 * tc[k] / n
            print(f"      {k}駒: {tc[k]:5d} ({pct:5.1f}%)")
        print(f"    先手盤上: {np.mean(sb):.2f}  後手盤上: {np.mean(gb):.2f}")
        print(f"    合法手数: 平均={np.mean(lm):.2f}")
        print(f"    王手状態: {check} ({100*check/n:.1f}%)")
        print(f"    にわとり先手: {niwatori_s} ({100*niwatori_s/n:.1f}%)")
        print(f"    にわとり後手: {niwatori_g} ({100*niwatori_g/n:.1f}%)")

    print_comparison("自己対局局面", sp_features)
    print_comparison("逆判定局面 (oracle=win, model=lose)", reverse_features)


if __name__ == "__main__":
    main()
