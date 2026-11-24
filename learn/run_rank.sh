#!/bin/sh
# 実験5 段階A arm A9L（木レコードの兄弟ランキング学習）: イテレーションループ
#
# 生成は factorial の arm 2 と同一設定（UBFM 生成＋木記録:
# use_descent=1 / cn_search=0）。変わるのは学習側だけで、train_arm.py の
# 代わりに train_rank.py（MSE＋listwise 順序損失）を使う。設計は
# EXP5_DESIGN.md の A9L の項、学習器は train_rank.py のヘッダ参照。
#
# 使い方: ./run_rank.sh <games> [total] [iter_positions] [rank_weight]
#   games         : 1イテレーションの生成局数（arm 2 と同じ 250）
#   total         : イテレーション総数（既定 20）
#   iter_positions: 1イテレーションで学習する局面数の上限（既定 5000）
#   rank_weight   : 順序損失の重み（任意。既定は train_rank.py の 0.15）
#
# 中断しても同じコマンドで再実行すればよい。学習の完了数は
# model/a9l/arm_state.json の iterations で管理され、そこから再開する。
# 学習中に落ちた回は、生成済みデータを再利用して学習からやり直す。
#
# 前提: arm 2 と同じ設定のバイナリ learn/cpp_animal_shogi_arm2 と、
#       train_rank.py --init-only 済みの model/a9l/ があること。
set -u
cd "$(dirname "$0")"

echo "run_rank.sh version 2026-11-22 v1（A9L: B1 土台＋listwise 順序損失）"

GAMES=${1:?使い方: ./run_rank.sh <games> [total] [iter_positions] [rank_weight]}
TOTAL=${2:-20}
ITER_POS=${3:-5000}
RANK_W=${4:-}

export LD_LIBRARY_PATH="${HOME}/onnxruntime-1.20.0/lib:${LD_LIBRARY_PATH:-}"

BIN="./cpp_animal_shogi_arm2"
MODEL_DIR="model/a9l"
EXP_UD=1
EXP_CN=0

[ -x "${BIN}" ] || { echo "バイナリがありません: ${BIN}（arm 2 と同じ設定のもの）" >&2; exit 1; }
[ -f "${MODEL_DIR}/best_single.onnx" ] || { echo "初期モデルがありません: ${MODEL_DIR}（train_rank.py --init-only）" >&2; exit 1; }
[ -f "${MODEL_DIR}/best_single.h5" ] || { echo "初期モデルがありません: ${MODEL_DIR}（train_rank.py --init-only）" >&2; exit 1; }

# 再開位置 = 学習完了イテレーション数（arm_state.json）
START=0
USED_GAMES=0
USED_POS=0
if [ -f "${MODEL_DIR}/arm_state.json" ]; then
  START=$(python3 -c "import json;print(json.load(open('${MODEL_DIR}/arm_state.json'))['iterations'])" 2>/dev/null || echo 0)
  USED_GAMES=$(python3 -c "import json;print(json.load(open('${MODEL_DIR}/arm_state.json'))['games'])" 2>/dev/null || echo 0)
  USED_POS=$(python3 -c "import json;print(json.load(open('${MODEL_DIR}/arm_state.json'))['positions'])" 2>/dev/null || echo 0)
fi
if [ "${START}" -ge "${TOTAL}" ]; then
  echo "A9L: ${TOTAL} イテレーション完了済みです（${MODEL_DIR}/train_log.md を確認）"
  exit 0
fi
# 予算を使い切っていたら生成せずに完了として終わる（train_rank.py の既定予算と揃える）
if [ "${USED_POS}" -ge 100000 ] || [ "${USED_GAMES}" -ge 5000 ]; then
  echo "A9L: 予算を使い切りました（${USED_GAMES} 局 / ${USED_POS} 局面）。この arm は完了です"
  exit 0
fi

mkdir -p logs
echo "A9L: イテレーション ${START} から $((TOTAL - 1)) まで、1回 ${GAMES} 局で実行します"
echo "A9L: 1イテレーションの学習局面数上限は ${ITER_POS} 局面です"
if [ -n "${RANK_W}" ]; then
  echo "A9L: 順序損失の重みは ${RANK_W} です"
fi

i=${START}
while [ "${i}" -lt "${TOTAL}" ]; do
  # ループ内でも予算を確認する（run_arm.sh v4 と同じ）
  if [ -f "${MODEL_DIR}/arm_state.json" ]; then
    NOW_POS=$(python3 -c "import json;print(json.load(open('${MODEL_DIR}/arm_state.json'))['positions'])" 2>/dev/null || echo 0)
    NOW_GAMES=$(python3 -c "import json;print(json.load(open('${MODEL_DIR}/arm_state.json'))['games'])" 2>/dev/null || echo 0)
    if [ "${NOW_POS}" -ge 100000 ] || [ "${NOW_GAMES}" -ge 5000 ]; then
      echo "A9L: 予算を使い切りました（${NOW_GAMES} 局 / ${NOW_POS} 局面）。この arm は完了です"
      exit 0
    fi
  fi
  echo "===== A9L gen_${i} ====="
  GEN_DIR="data/a9l_gen_${i}"
  mkdir -p "${GEN_DIR}"
  if ls "${GEN_DIR}"/*.json >/dev/null 2>&1; then
    echo "既存の生成データを再利用します: ${GEN_DIR}（中断した回の再開）"
  else
    cp "${MODEL_DIR}/best_single.onnx" model/best_single.onnx
    rm -f data/selfplay*.tmp
    # 生成前に data/ 直下の既存ファイルを記録し、生成で増えた分だけを移す
    BEFORE_LIST="logs/.before_a9l_gen${i}.txt"
    ls data/selfplay*.json > "${BEFORE_LIST}" 2>/dev/null || true
    if [ -s "${BEFORE_LIST}" ]; then
      echo "注意: data/ 直下に前回以前の selfplay ファイルが $(wc -l < "${BEFORE_LIST}") 件残っています。今回の生成分だけを移すので混ざりません"
    fi
    "${BIN}" "${GAMES}" 2>&1 | tee "logs/a9l_gen${i}.log"
    PARAMS_LINE=$(grep -m1 "^PARAMS " "logs/a9l_gen${i}.log" || true)
    if [ -z "${PARAMS_LINE}" ]; then
      echo "PARAMS 行が生成ログにありません（search_params_log.patch 未適用のバイナリ？）" >&2
      exit 1
    fi
    UD=$(echo "${PARAMS_LINE}" | sed -n 's/.*use_descent=\([0-9]\).*/\1/p')
    CN=$(echo "${PARAMS_LINE}" | sed -n 's/.*cn_search=\([0-9]\).*/\1/p')
    if [ "${UD}" != "${EXP_UD}" ] || [ "${CN}" != "${EXP_CN}" ]; then
      echo "フラグ不一致: A9L の期待は use_descent=${EXP_UD} cn_search=${EXP_CN} だが、実際は use_descent=${UD} cn_search=${CN}（バイナリの取り違え）" >&2
      exit 1
    fi
    NEW_FILES=$(ls data/selfplay*.json 2>/dev/null | grep -vxF -f "${BEFORE_LIST}" || true)
    COUNT=$(printf '%s\n' "${NEW_FILES}" | grep -c . || true)
    if [ "${COUNT}" -eq 0 ]; then
      echo "生成ファイルがありません（生成に失敗した可能性）" >&2
      exit 1
    fi
    for f in ${NEW_FILES}; do mv "${f}" "${GEN_DIR}/"; done
    rm -f "${BEFORE_LIST}"
    echo "生成: ${COUNT} ファイル -> ${GEN_DIR}/"
  fi
  RANK_OPT=""
  if [ -n "${RANK_W}" ]; then
    RANK_OPT="--rank-weight ${RANK_W}"
  fi
  ../.venv/bin/python train_rank.py --model-dir "${MODEL_DIR}" \
      --data-dir "${GEN_DIR}" --init "${MODEL_DIR}/best_single.h5" \
      --iter-positions "${ITER_POS}" ${RANK_OPT} || {
    echo "train_rank.py が失敗しました（A9L gen_${i}）。train_log と出力を確認してください" >&2
    exit 1
  }
  i=$((i + 1))
done
echo "A9L: 完了。結果は ${MODEL_DIR}/train_log.md と ${MODEL_DIR}/arm_state.json を確認してください"
