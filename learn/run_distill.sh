#!/bin/sh
# 予算の蒸留（実験候補 A）イテレーションループ（run_distill.sh）
# ============================================================
# 使い方（learn/ ディレクトリで）:
#   ./run_distill.sh [games=250] [total=20] [iter_positions=5000] [student_po=10]
#
# 1イテレーションの流れ:
#   1. model/distill/best_single.onnx を model/best_single.onnx にコピー
#   2. 生成（arm 0 と同じ po=100 自己対局）し、PARAMS 行で
#      use_descent=0 / cn_search=0 を確認（食い違えば中止）
#   3. 新しくできた data/selfplay*.json だけを data/distill_gen<i>/ へ移動
#   4. train_distill.py --make-keys で root キー一覧を作る
#   5. dump モードで学生 po の探索木を data/distill_gen<i>/trees.txt に出す
#      （DUMP_PARAMS の po_num を確認）
#   6. train_distill.py で学習（backup 損失で教師 s に合わせる）
#
# 再開: model/distill/arm_state.json の iterations から自動で続きを回す。
# 生成済み（gen ディレクトリに *.json がある）場合は生成と dump を
# やり直さず学習だけを再試行する（trees.txt があれば dump も省略）。
#
# このスクリプトが触るファイル:
#   - 削除: data/selfplay*.tmp のみ
#   - 移動: data/selfplay*.json → data/distill_gen<i>/
#   - 上書き: model/best_single.onnx（毎イテレーション）
#   - 追記/更新: model/distill/ 配下（モデル・状態・ログ）、logs/ 配下
#
# バイナリ名は既定で ./cpp_animal_shogi（サフィックスなし）。
# 別の名前でビルドした場合は BIN 環境変数で指定する。

VERSION="2026-11-23 v2（バイナリ名の既定をサフィックスなしに変更、起動前チェック追加）"
echo "run_distill.sh $VERSION"

GAMES=${1:-250}
TOTAL=${2:-20}
ITER_POS=${3:-5000}
STUDENT_PO=${4:-10}
BIN=${BIN:-./cpp_animal_shogi}
MODEL_DIR=model/distill
PY=../.venv/bin/python

if [ ! -x "$BIN" ]; then
  echo "** バイナリ $BIN がありません。ビルド先に合わせて BIN を指定してください **"
  echo "   例: BIN=./cpp_animal_shogi_distill sh run_distill.sh 250 20 5000 10"
  exit 1
fi

if [ ! -f "$MODEL_DIR/best_single.onnx" ]; then
  echo "** $MODEL_DIR/best_single.onnx がありません。先に初期化してください **"
  echo "   $PY train_distill.py --model-dir $MODEL_DIR --init-only"
  exit 1
fi

START=$($PY -c "import json;print(json.load(open('$MODEL_DIR/arm_state.json'))['iterations'])" 2>/dev/null || echo 0)
echo "開始イテレーション: $START / 全 $TOTAL 回（生成 $GAMES 局/回、学習 $ITER_POS 局面/回、学生 po=$STUDENT_PO）"

i=$START
while [ "$i" -lt "$TOTAL" ]; do
  GEN_DIR=data/distill_gen$i
  echo "=== iteration $i ==="

  if ls $GEN_DIR/*.json >/dev/null 2>&1; then
    echo "生成済みのため生成を省略: $GEN_DIR"
  else
    cp $MODEL_DIR/best_single.onnx model/best_single.onnx
    rm -f data/selfplay*.tmp
    ls data/selfplay*.json 2>/dev/null | sort > /tmp/distill_before.txt || true
    "$BIN" "$GAMES" > logs/distill_gen$i.log 2>&1
    PARAMS=$(grep "^PARAMS" logs/distill_gen$i.log | head -1)
    echo "$PARAMS"
    case "$PARAMS" in
      *"use_descent=0"*"cn_search=0"*) : ;;
      *) echo "** PARAMS が想定と違います（use_descent=0 / cn_search=0 のはず）: $PARAMS **"
         echo "** 生成ログの先頭: **"
         head -5 logs/distill_gen$i.log
         exit 1 ;;
    esac
    mkdir -p $GEN_DIR
    ls data/selfplay*.json 2>/dev/null | sort > /tmp/distill_after.txt || true
    comm -13 /tmp/distill_before.txt /tmp/distill_after.txt | while read f; do
      mv "$f" $GEN_DIR/
    done
  fi

  if [ ! -f "$GEN_DIR/trees.txt" ]; then
    $PY train_distill.py --model-dir $MODEL_DIR --data-dir $GEN_DIR \
        --make-keys --keys-out $GEN_DIR/keys.txt || exit 1
    "$BIN" dump $GEN_DIR/keys.txt $STUDENT_PO $GEN_DIR/trees.txt \
        > logs/distill_dump$i.log 2>&1
    DPARAMS=$(grep "^DUMP_PARAMS" logs/distill_dump$i.log | head -1)
    echo "$DPARAMS"
    case "$DPARAMS" in
      *"po_num=$STUDENT_PO"*) : ;;
      *) echo "** DUMP_PARAMS の po_num が $STUDENT_PO ではありません: $DPARAMS **"; exit 1 ;;
    esac
  else
    echo "trees.txt があるため dump を省略: $GEN_DIR/trees.txt"
  fi

  $PY train_distill.py --model-dir $MODEL_DIR --data-dir $GEN_DIR \
      --trees $GEN_DIR/trees.txt --init $MODEL_DIR/best_single.h5 \
      --iter-positions $ITER_POS --student-po $STUDENT_PO || exit 1
  i=$((i + 1))
done
echo "完了: $TOTAL イテレーション"
