#!/bin/sh
# 実験5 4アーム factorial: 1 arm 分のイテレーションループ（ARM_RUNBOOK.md §3）
#
# 使い方: ./run_arm.sh <arm> <games> [total] [iter_positions]
#   arm   : 0=Baseline / 1=B5(CN) / 2=B1(木学習) / 3=B5+B1
#   games : 1イテレーションの生成局数（手順書 §5 の records/game 実測から決めた値）
#   total : イテレーション総数（既定 20）
#   iter_positions : 1イテレーションで学習する局面数の上限（任意）。
#                    木学習 arm のように records/game が大きい場合に、
#                    予算を複数イテレーションへ分散させるために指定する
#
# 中断しても同じコマンドで再実行すればよい。学習の完了数は
# model/arm<arm>/arm_state.json の iterations で管理され、そこから再開する。
# 学習中に落ちた回は、生成済みデータを再利用して学習からやり直す。
#
# 各イテレーションでやること:
#   1. arm の ONNX を model/best_single.onnx に差してから生成する
#      （生成側バイナリは固定パスしか読まないため、この差し替えが必須）
#   2. 生成ログの PARAMS 行で use_descent / cn_search を検証し、
#      その arm の期待値と違えばバイナリの取り違えとして止める
#   3. 生成ファイルを data/arm<arm>_gen<i>/ に移す
#   4. train_arm.py で学習する。失敗したらループを止める
#      （古いモデルのまま生成を続けないため）
#
# 前提: arm 用バイナリ learn/cpp_animal_shogi_arm<arm> と、
#       train_arm.py --init-only 済みの model/arm<arm>/ があること。
set -u
cd "$(dirname "$0")"

echo "run_arm.sh version 2026-11-21 v4（ループ内予算チェック／イテレーション上限の受け渡し）"

ARM=${1:?使い方: ./run_arm.sh <arm> <games> [total] [iter_positions]}
GAMES=${2:?使い方: ./run_arm.sh <arm> <games> [total] [iter_positions]}
TOTAL=${3:-20}
ITER_POS=${4:-}

export LD_LIBRARY_PATH="${HOME}/onnxruntime-1.20.0/lib:${LD_LIBRARY_PATH:-}"

BIN="./cpp_animal_shogi"
MODEL_DIR="model/arm${ARM}"

# arm ごとの期待フラグ（PARAMS 行の検証用）
case "${ARM}" in
  0) EXP_UD=0; EXP_CN=0 ;;
  1) EXP_UD=0; EXP_CN=1 ;;
  2) EXP_UD=1; EXP_CN=0 ;;
  3) EXP_UD=1; EXP_CN=1 ;;
  *) echo "arm は 0..3 で指定してください" >&2; exit 1 ;;
esac

[ -x "${BIN}" ] || { echo "バイナリがありません: ${BIN}（手順書 §1）" >&2; exit 1; }
[ -f "${MODEL_DIR}/best_single.onnx" ] || { echo "初期モデルがありません: ${MODEL_DIR}（手順書 §2 の --init-only）" >&2; exit 1; }
[ -f "${MODEL_DIR}/best_single.h5" ] || { echo "初期モデルがありません: ${MODEL_DIR}（手順書 §2 の --init-only）" >&2; exit 1; }

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
  echo "arm${ARM}: ${TOTAL} イテレーション完了済みです（${MODEL_DIR}/train_log.md を確認）"
  exit 0
fi
# 予算を使い切っていたら生成せずに完了として終わる（train_arm.py の既定予算と揃える）
if [ "${USED_POS}" -ge 100000 ] || [ "${USED_GAMES}" -ge 5000 ]; then
  echo "arm${ARM}: 予算を使い切りました（${USED_GAMES} 局 / ${USED_POS} 局面）。この arm は完了です"
  exit 0
fi

mkdir -p logs
echo "arm${ARM}: イテレーション ${START} から $((TOTAL - 1)) まで、1回 ${GAMES} 局で実行します"
if [ -n "${ITER_POS}" ]; then
  echo "arm${ARM}: 1イテレーションの学習局面数上限は ${ITER_POS} 局面です"
fi

i=${START}
while [ "${i}" -lt "${TOTAL}" ]; do
  # ループ内でも予算を確認する（1イテレーションで予算を使い切る arm では、
  # 次の生成を回さずに完了として終わる）
  if [ -f "${MODEL_DIR}/arm_state.json" ]; then
    NOW_POS=$(python3 -c "import json;print(json.load(open('${MODEL_DIR}/arm_state.json'))['positions'])" 2>/dev/null || echo 0)
    NOW_GAMES=$(python3 -c "import json;print(json.load(open('${MODEL_DIR}/arm_state.json'))['games'])" 2>/dev/null || echo 0)
    if [ "${NOW_POS}" -ge 100000 ] || [ "${NOW_GAMES}" -ge 5000 ]; then
      echo "arm${ARM}: 予算を使い切りました（${NOW_GAMES} 局 / ${NOW_POS} 局面）。この arm は完了です"
      exit 0
    fi
  fi
  echo "===== arm${ARM} gen_${i} ====="
  GEN_DIR="data/arm${ARM}_gen_${i}"
  mkdir -p "${GEN_DIR}"
  if ls "${GEN_DIR}"/*.json >/dev/null 2>&1; then
    echo "既存の生成データを再利用します: ${GEN_DIR}（中断した回の再開）"
  else
    cp "${MODEL_DIR}/best_single.onnx" model/best_single.onnx
    rm -f data/selfplay*.tmp
    # 生成前に data/ 直下の既存ファイルを記録し、生成で増えた分だけを移す
    # （残っていた古いファイルが今回の世代に混ざらないようにするため）
    BEFORE_LIST="logs/.before_arm${ARM}_gen${i}.txt"
    ls data/selfplay*.json > "${BEFORE_LIST}" 2>/dev/null || true
    if [ -s "${BEFORE_LIST}" ]; then
      echo "注意: data/ 直下に前回以前の selfplay ファイルが $(wc -l < "${BEFORE_LIST}") 件残っています。今回の生成分だけを移すので混ざりません"
    fi
    "${BIN}" "${GAMES}" 2>&1 | tee "logs/arm${ARM}_gen${i}.log"
    PARAMS_LINE=$(grep -m1 "^PARAMS " "logs/arm${ARM}_gen${i}.log" || true)
    if [ -z "${PARAMS_LINE}" ]; then
      echo "PARAMS 行が生成ログにありません（search_params_log.patch 未適用のバイナリ？）" >&2
      exit 1
    fi
    UD=$(echo "${PARAMS_LINE}" | sed -n 's/.*use_descent=\([0-9]\).*/\1/p')
    CN=$(echo "${PARAMS_LINE}" | sed -n 's/.*cn_search=\([0-9]\).*/\1/p')
    if [ "${UD}" != "${EXP_UD}" ] || [ "${CN}" != "${EXP_CN}" ]; then
      echo "フラグ不一致: arm${ARM} の期待は use_descent=${EXP_UD} cn_search=${EXP_CN} だが、実際は use_descent=${UD} cn_search=${CN}（バイナリの取り違え）" >&2
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
  ITER_OPT=""
  if [ -n "${ITER_POS}" ]; then
    ITER_OPT="--iter-positions ${ITER_POS}"
  fi
  ../.venv/bin/python train_arm.py --arm "${ARM}" --model-dir "${MODEL_DIR}" \
      --data-dir "${GEN_DIR}" --init "${MODEL_DIR}/best_single.h5" ${ITER_OPT} || {
    echo "train_arm.py が失敗しました（arm${ARM} gen_${i}）。train_log と出力を確認してください" >&2
    exit 1
  }
  i=$((i + 1))
done
echo "arm${ARM}: 完了。結果は ${MODEL_DIR}/train_log.md と ${MODEL_DIR}/arm_state.json を確認してください"
