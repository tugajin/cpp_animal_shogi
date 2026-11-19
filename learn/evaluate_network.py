# ====================
# 新パラメータ評価部
# ====================

# パッケージのインポート
from game import State
from pathlib import Path
from shutil import copy
import numpy as np
from single_network import *
import json
from line import *

# ベストプレイヤーの交代
def update_best_player():
    copy('./model/latest_single.h5', './model/best_single.h5')
    print('Change BestPlayer')

def predict(model, state, device):
    feature = np.array(state.feature())
    file, rank, channel = DN_INPUT_SHAPE
    feature = feature.reshape(1, channel, file, rank)
    f = torch.from_numpy(feature)
    f = f.to(device).float()
    return model(f)

def read_file_to_list(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data_list.append(line.strip())  # 改行文字を削除してリストに追加
    return data_list

def evaluate_problem():
    # ベストプレイヤーのモデルの読み込み
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SingleNet()
    model.load_state_dict(torch.load('./model/best_single.h5', weights_only=True))
    model = model.to(device)
    model.eval()

    problem_list = read_file_to_list('../oracle/problem.txt')

    correct_num = 0
    for problem in problem_list:
        split_text_list = problem.split(",")
        key = int(split_text_list[0])
        result = int(split_text_list[1])
        state = from_hash(key)
        score = predict(model,state,device)
        score = score.item()
        org_score = score
        if score > 0.2:
            score = 1
        elif score < -0.2:
            score = -1
        else:
            score = 0

        if score == result:
            correct_num += 1
    message = "ans:",correct_num/len(problem_list),f"({correct_num})/({len(problem_list)})"
    print(message)
    return correct_num/len(problem_list)


def analyze_errors():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SingleNet()
    model.load_state_dict(torch.load('./model/best_single.h5', weights_only=True))
    model = model.to(device)
    model.eval()

    problem_list = read_file_to_list('../oracle/problem.txt')

    scores = []
    results = []
    for problem in problem_list:
        split_text_list = problem.split(",")
        key = int(split_text_list[0])
        result = int(split_text_list[1])
        state = from_hash(key)
        score = predict(model, state, device).item()
        scores.append(score)
        results.append(result)

    scores = np.array(scores)
    results = np.array(results)

    def classify(s):
        if s > 0.2: return 1
        elif s < -0.2: return -1
        else: return 0

    preds = np.array([classify(s) for s in scores])
    correct = preds == results

    print("=" * 60)
    print("エラー分析")
    print("=" * 60)

    # 1. 全体の正解率
    print(f"\n正解率: {correct.sum()}/{len(results)} ({correct.mean():.4f})")

    # 2. 誤答の内訳
    wrong_mask = ~correct
    wrong_scores = scores[wrong_mask]
    wrong_results = results[wrong_mask]
    wrong_preds = preds[wrong_mask]

    print(f"\n--- 誤答の内訳 ({wrong_mask.sum()}件) ---")
    patterns = {}
    for r, p in zip(wrong_results, wrong_preds):
        key = f"正解={r:+d} → 予測={p:+d}"
        patterns[key] = patterns.get(key, 0) + 1
    for k, v in sorted(patterns.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}件")

    # 3. スコア分布ヒストグラム
    print("\n--- スコア分布 (全体) ---")
    bins = [(-1.0, -0.8), (-0.8, -0.6), (-0.6, -0.4), (-0.4, -0.2),
            (-0.2, 0.0), (0.0, 0.2), (0.2, 0.4), (0.4, 0.6),
            (0.6, 0.8), (0.8, 1.0)]
    for lo, hi in bins:
        mask = (scores >= lo) & (scores < hi)
        n = mask.sum()
        c = correct[mask].sum()
        rate = c / n if n > 0 else 0
        bar = "#" * int(n / 40)
        print(f"  [{lo:+.1f}, {hi:+.1f}): {n:5d}  正解率 {rate:.3f}  {bar}")
    over = (scores >= 1.0).sum() + (scores <= -1.0).sum()
    if over > 0:
        print(f"  |score|>=1.0: {over}件")

    # 4. 確信度 vs 正解率
    print("\n--- 確信度 vs 正解率 ---")
    confidence_bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
                       (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
                       (0.8, 0.9), (0.9, 1.0)]
    for lo, hi in confidence_bins:
        abs_scores = np.abs(scores)
        mask = (abs_scores >= lo) & (abs_scores < hi)
        n = mask.sum()
        c = correct[mask].sum()
        rate = c / n if n > 0 else 0
        print(f"  |score| [{lo:.1f}, {hi:.1f}): {n:5d}  正解率 {rate:.3f}")

    # 5. 閾値感度分析
    print("\n--- 閾値別の正解率 ---")
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        p = np.where(scores > threshold, 1, np.where(scores < -threshold, -1, 0))
        acc = (p == results).mean()
        print(f"  閾値 ±{threshold:.2f}: {acc:.4f}")

    # 6. 誤答のスコア分布
    print("\n--- 誤答のスコア分布 ---")
    abs_wrong = np.abs(wrong_scores)
    for lo, hi in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]:
        mask = (abs_wrong >= lo) & (abs_wrong < hi)
        n = mask.sum()
        print(f"  |score| [{lo:.1f}, {hi:.1f}): {n:4d}件")


# 動作確認
if __name__ == '__main__':
    analyze_errors()
