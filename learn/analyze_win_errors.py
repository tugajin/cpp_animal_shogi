import numpy as np
import torch
import re
from collections import Counter
from single_network import *
from game import *

DATA_DIR = "../data"

def load_oracle_keys(path, count):
    with open(path, "rb") as f:
        data = np.fromfile(f, dtype=np.uint64, count=count)
    return data

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
    lines = s.strip().split('\n')

    # 盤面の駒を数える
    board_pieces = {"先手": [], "後手": []}
    hand_pieces = {"先手": [], "後手": []}

    piece_map = {
        "^ひ": "先手ひよこ", "^に": "先手にわとり", "^き": "先手きりん",
        "^ぞ": "先手ぞう", "^ら": "先手ライオン",
        "vひ": "後手ひよこ", "vに": "後手にわとり", "vき": "後手きりん",
        "vぞ": "後手ぞう", "vら": "後手ライオン",
    }

    board_str = ""
    for line in lines:
        if "手番" in line or "持ち駒" in line:
            continue
        board_str += line

    # count pieces on board
    sente_board = 0
    gote_board = 0
    has_niwatori_sente = False
    has_niwatori_gote = False
    for token, name in piece_map.items():
        count = board_str.count(token)
        if "先手" in name:
            sente_board += count
            if "にわとり" in name and count > 0:
                has_niwatori_sente = True
        else:
            gote_board += count
            if "にわとり" in name and count > 0:
                has_niwatori_gote = True

    # parse hand pieces
    sente_hand = 0
    gote_hand = 0
    for line in lines:
        if "先手:持ち駒" in line and "なし" not in line:
            # count hand pieces
            hand_part = line.split("持ち駒")[1] if "持ち駒" in line else ""
            for ch in ["ひ", "き", "ぞ"]:
                sente_hand += hand_part.count(ch)
        elif "後手:持ち駒" in line and "なし" not in line:
            hand_part = line.split("持ち駒")[1] if "持ち駒" in line else ""
            for ch in ["ひ", "き", "ぞ"]:
                gote_hand += hand_part.count(ch)

    total_pieces = sente_board + gote_board + sente_hand + gote_hand

    # legal moves count
    ml = gamelibs.MoveList()
    gamelibs.legal_moves(state.pos, ml)
    legal_move_count = ml.len()

    # in check?
    in_check = gamelibs.in_checked(state.pos)

    return {
        "sente_board": sente_board,
        "gote_board": gote_board,
        "sente_hand": sente_hand,
        "gote_hand": gote_hand,
        "total_pieces": total_pieces,
        "legal_moves": legal_move_count,
        "in_check": in_check,
        "has_niwatori_sente": has_niwatori_sente,
        "has_niwatori_gote": has_niwatori_gote,
    }


def main():
    SAMPLE_SIZE = 100000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SingleNet()
    model.load_state_dict(torch.load('./model/best_single.h5', weights_only=True))
    model = model.to(device)
    model.eval()

    # win.dat を読み込み
    WIN_COUNT = 196773087
    print(f"win.dat を読み込み中... ({WIN_COUNT:,}局面)")
    keys = load_oracle_keys(f"{DATA_DIR}/win.dat", WIN_COUNT)
    rng = np.random.default_rng(42)
    indices = rng.choice(WIN_COUNT, SAMPLE_SIZE, replace=False)
    keys = keys[indices]

    states = []
    for k in keys:
        state = from_hash(int(k))
        if state.pos.is_win() or state.pos.is_lose():
            continue
        states.append(state)
    print(f"有効局面: {len(states):,}")

    scores = batch_predict(model, states, device)

    # 正解（score > 0.2）と逆判定（score < -0.2）に分類
    correct_mask = scores > 0.2
    reverse_mask = scores < -0.2
    undecided_mask = (~correct_mask) & (~reverse_mask)

    print(f"\n正解: {correct_mask.sum():,}  逆判定: {reverse_mask.sum():,}  未決: {undecided_mask.sum():,}")

    # 逆判定の局面を詳細分析
    print(f"\n{'='*60}")
    print(f"逆判定の局面の詳細分析 ({reverse_mask.sum():,}件)")
    print(f"正解はwin(+1)なのに score < -0.2 と評価")
    print(f"{'='*60}")

    reverse_indices = np.where(reverse_mask)[0]
    correct_indices = np.where(correct_mask)[0]

    # サンプル数を制限して局面解析（全部やると遅いので）
    ANALYSIS_LIMIT = 5000

    print(f"\n--- 逆判定局面の特徴分析 (最大{ANALYSIS_LIMIT}局面) ---")
    reverse_features = []
    for idx in reverse_indices[:ANALYSIS_LIMIT]:
        feat = parse_position(states[idx])
        feat["score"] = float(scores[idx])
        reverse_features.append(feat)

    print(f"\n--- 正解局面の特徴分析 (最大{ANALYSIS_LIMIT}局面, 比較用) ---")
    correct_sample = rng.choice(correct_indices, min(ANALYSIS_LIMIT, len(correct_indices)), replace=False)
    correct_features = []
    for idx in correct_sample:
        feat = parse_position(states[idx])
        feat["score"] = float(scores[idx])
        correct_features.append(feat)

    # 比較分析
    def summarize(features, label):
        if not features:
            print(f"  {label}: データなし")
            return
        n = len(features)
        print(f"\n{'='*60}")
        print(f"{label} ({n}局面)")
        print(f"{'='*60}")

        # スコア分布
        sc = [f["score"] for f in features]
        print(f"  スコア: 平均={np.mean(sc):.3f} 中央値={np.median(sc):.3f} 標準偏差={np.std(sc):.3f}")

        # 駒数
        total = [f["total_pieces"] for f in features]
        print(f"\n  総駒数: 平均={np.mean(total):.2f} 中央値={np.median(total):.1f}")
        total_counter = Counter(total)
        for k in sorted(total_counter.keys()):
            pct = 100 * total_counter[k] / n
            bar = "#" * int(pct / 2)
            print(f"    {k}駒: {total_counter[k]:5d} ({pct:5.1f}%) {bar}")

        # 先手vs後手の盤上駒数
        sb = [f["sente_board"] for f in features]
        gb = [f["gote_board"] for f in features]
        sh = [f["sente_hand"] for f in features]
        gh = [f["gote_hand"] for f in features]
        print(f"\n  先手盤上: 平均={np.mean(sb):.2f}  後手盤上: 平均={np.mean(gb):.2f}")
        print(f"  先手持駒: 平均={np.mean(sh):.2f}  後手持駒: 平均={np.mean(gh):.2f}")

        # 合法手数
        lm = [f["legal_moves"] for f in features]
        print(f"\n  合法手数: 平均={np.mean(lm):.2f} 中央値={np.median(lm):.1f}")

        # 王手
        check = sum(1 for f in features if f["in_check"])
        print(f"  王手状態: {check} ({100*check/n:.1f}%)")

        # にわとり
        niwatori_s = sum(1 for f in features if f["has_niwatori_sente"])
        niwatori_g = sum(1 for f in features if f["has_niwatori_gote"])
        print(f"  にわとり(先手): {niwatori_s} ({100*niwatori_s/n:.1f}%)")
        print(f"  にわとり(後手): {niwatori_g} ({100*niwatori_g/n:.1f}%)")

    summarize(reverse_features, "逆判定 (win→lose)")
    summarize(correct_features, "正解 (win→win)")

    # 逆判定のスコア分布詳細
    print(f"\n{'='*60}")
    print("逆判定のスコア分布詳細")
    print(f"{'='*60}")
    reverse_scores = scores[reverse_mask]
    for lo, hi in [(-1.0, -0.9), (-0.9, -0.8), (-0.8, -0.7), (-0.7, -0.6),
                   (-0.6, -0.5), (-0.5, -0.4), (-0.4, -0.3), (-0.3, -0.2)]:
        mask = (reverse_scores >= lo) & (reverse_scores < hi)
        cnt = mask.sum()
        pct = 100 * cnt / len(reverse_scores)
        bar = "#" * int(pct / 2)
        print(f"  [{lo:+.1f}, {hi:+.1f}): {cnt:5d} ({pct:5.1f}%) {bar}")

    # 局面サンプルの表示
    print(f"\n{'='*60}")
    print("逆判定の局面サンプル (スコアが低い順に5局面)")
    print(f"{'='*60}")
    sorted_idx = np.argsort(scores[reverse_mask])
    for i in sorted_idx[:5]:
        real_idx = reverse_indices[i]
        print(f"\n--- score={scores[real_idx]:.4f} (正解=win) ---")
        print(states[real_idx])


if __name__ == "__main__":
    main()
