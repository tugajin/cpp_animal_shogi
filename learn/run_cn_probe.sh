#!/bin/sh
# 弱いモデルでの CN 対 UBFM 探索比較（CN ライダー）: CN 側の probe 実行
# run_cn_probe.sh 2026-11-25 v1
#
# 問い: CN（共謀数）による葉の選び方は、一番強いモデルでは生成を
# 改善した（教師の正解一致 +2.7pt）。弱いモデルの「探索の出力」を
# 測るとどうか。UBFM 側は予算曲線のときに測った probe 出力があるので、
# ここでは CN 側だけを同じ条件で測り、probe_score.py で対決させる。
#
# 前提:
# - ai/ubfm.hpp の g_use_cn_search = true でビルドしたバイナリが
#   learn/cpp_animal_shogi にあること。最初の probe 出力の
#   PROBE_PARAMS で cn_search=1 を検証し、違えば止める
# - 局面リスト probe_keys_arm0.txt（予算曲線と同じ、正解を引ける
#   2,254 局面）を使う
# - 対象モデル: model/init0（未学習）/ model/arm3（ans 0.749 で最弱）
#
# 出力: probe_cn_init0_po100.txt など4ファイル。
# 終わったら ubfm.hpp のフラグを false に戻して再ビルドすること
# （通常の生成・probe は CN なしが前提）。
#
# 使い方（learn/ ディレクトリで）:
#   sh run_cn_probe.sh

set -u
VERSION="2026-11-25 v1"
BIN=${BIN:-./cpp_animal_shogi}
KEYS=${KEYS:-probe_keys_arm0.txt}

echo "run_cn_probe.sh $VERSION"
if [ ! -x "$BIN" ]; then
  echo "中止: バイナリ $BIN がありません"
  exit 1
fi
if [ ! -f "$KEYS" ]; then
  echo "中止: 局面リスト $KEYS がありません"
  exit 1
fi

for MODEL in init0 arm3; do
  ONNX="model/$MODEL/best_single.onnx"
  if [ ! -f "$ONNX" ]; then
    echo "中止: $ONNX がありません"
    exit 1
  fi
  cp "$ONNX" model/best_single.onnx
  for PO in 100 10; do
    OUT="probe_cn_${MODEL}_po${PO}.txt"
    echo "probe: model=$MODEL po=$PO -> $OUT"
    "$BIN" probe "$KEYS" "$PO" > "$OUT"
    PARAMS=$(grep -m1 PROBE_PARAMS "$OUT" || true)
    echo "  $PARAMS"
    case "$PARAMS" in
      *"cn_search=1"*) ;;
      *) echo "中止: cn_search=1 ではありません。CN ビルドのバイナリか確認してください"
         exit 1 ;;
    esac
    case "$PARAMS" in
      *"po_num=$PO"*) ;;
      *) echo "中止: po_num=$PO ではありません"
         exit 1 ;;
    esac
    SKIP=$(grep -c PROBE_SKIP "$OUT" || true)
    echo "  PROBE_SKIP: $SKIP 件"
  done
done

echo ""
echo "完了。UBFM 側（予算曲線のときの probe 出力）と対決させる例:"
echo "  ../.venv/bin/python probe_score.py --keys $KEYS --probe ubfm=probe_init0_po100.txt --probe cn=probe_cn_init0_po100.txt --out probe_cn_init0_po100.md"
echo "  ../.venv/bin/python probe_score.py --keys $KEYS --probe ubfm=probe_arm3_po100.txt --probe cn=probe_cn_arm3_po100.txt --out probe_cn_arm3_po100.md"
echo "（ubfm 側のファイル名は予算曲線のときに使ったものに合わせてください。"
echo "  無ければ CN を切ったバイナリで同じ probe を取り直してください）"
echo "注意: model/best_single.onnx は最後のモデル（arm3）のままです。"
echo "ubfm.hpp の g_use_cn_search を false に戻して再ビルドしてください。"
