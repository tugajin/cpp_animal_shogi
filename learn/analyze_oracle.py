import numpy as np
import struct
import torch
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

def analyze(scores, label, result_value):
    n = len(scores)
    print(f"\n{'='*60}")
    print(f"{label} ({n:,}局面, 正解={result_value:+d})")
    print(f"{'='*60}")

    # スコア統計
    print(f"  平均: {scores.mean():.4f}  中央値: {np.median(scores):.4f}  "
          f"標準偏差: {scores.std():.4f}")
    print(f"  最小: {scores.min():.4f}  最大: {scores.max():.4f}")

    # 閾値0.2での正解率
    if result_value == 1:
        correct = (scores > 0.2).sum()
        wrong_opposite = (scores < -0.2).sum()
        undecided = n - correct - wrong_opposite
    elif result_value == -1:
        correct = (scores < -0.2).sum()
        wrong_opposite = (scores > 0.2).sum()
        undecided = n - correct - wrong_opposite
    else:
        correct = ((scores >= -0.2) & (scores <= 0.2)).sum()
        wrong_opposite = 0
        undecided = n - correct

    print(f"\n  閾値±0.2での判定:")
    print(f"    正解: {correct:,} ({100*correct/n:.2f}%)")
    print(f"    逆判定: {wrong_opposite:,} ({100*wrong_opposite/n:.2f}%)")
    print(f"    未決(|score|<0.2): {undecided:,} ({100*undecided/n:.2f}%)")

    # スコア分布
    print(f"\n  スコア分布:")
    bins = [(-1.1, -0.8), (-0.8, -0.6), (-0.6, -0.4), (-0.4, -0.2),
            (-0.2, 0.0), (0.0, 0.2), (0.2, 0.4), (0.4, 0.6),
            (0.6, 0.8), (0.8, 1.1)]
    for lo, hi in bins:
        mask = (scores >= lo) & (scores < hi)
        cnt = mask.sum()
        pct = 100 * cnt / n
        bar = "#" * int(pct / 2)
        lo_s = f"{max(lo,-1.0):+.1f}" if lo <= -1.0 else f"{lo:+.1f}"
        hi_s = f"{min(hi,1.0):+.1f}" if hi >= 1.0 else f"{hi:+.1f}"
        print(f"    [{lo_s}, {hi_s}): {cnt:8,} ({pct:5.1f}%) {bar}")


def main():
    SAMPLE_SIZE = 100000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = SingleNet()
    model.load_state_dict(torch.load('./model/best_single.h5', weights_only=True))
    model = model.to(device)
    model.eval()

    categories = [
        ("win",  f"{DATA_DIR}/win.dat",  196773087,  1),
        ("lose", f"{DATA_DIR}/lose.dat",  47347380, -1),
        ("draw", f"{DATA_DIR}/draw.dat",   2682700,  0),
    ]

    for name, path, total, result_value in categories:
        print(f"\n{name}.dat を読み込み中... ({total:,}局面)")
        keys = load_oracle_keys(path, total)

        if total > SAMPLE_SIZE:
            rng = np.random.default_rng(42)
            indices = rng.choice(total, SAMPLE_SIZE, replace=False)
            keys = keys[indices]
            print(f"  {SAMPLE_SIZE:,}局面をサンプリング")

        # 終局局面を除外して特徴量生成
        states = []
        skipped = 0
        for k in keys:
            state = from_hash(int(k))
            if state.pos.is_win() or state.pos.is_lose():
                skipped += 1
                continue
            states.append(state)
        print(f"  有効局面: {len(states):,} (終局除外: {skipped:,})")

        scores = batch_predict(model, states, device)
        analyze(scores, name, result_value)

    # 全体の正解率
    print(f"\n{'='*60}")
    print("サンプル全体のサマリ")
    print(f"{'='*60}")
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        print(f"  閾値 ±{threshold:.2f}:", end="")
        # 再計算用に全データ保持が必要なのでここはスキップ
    print("  (上記カテゴリ別の結果を参照)")

if __name__ == "__main__":
    main()
