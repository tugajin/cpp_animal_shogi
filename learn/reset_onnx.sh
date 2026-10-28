#!/bin/sh
# ゼロから学習し直すためのリセット (旧 reset.sh のONNX対応版)
# 実行場所: learn/ ディレクトリで実行すること
# 注意: 現在の学習済みモデルは model_backup_<日時>/ に退避してから消す
set -e

# 1. 現在のモデルと履歴を退避
# BACKUP="model_backup_$(date +%Y%m%d_%H%M%S)"
# mkdir -p "$BACKUP"
# cp -a model/. "$BACKUP"/
# if [ -f history.csv ]; then cp history.csv "$BACKUP"/; fi
# if [ -f selfplay_result.csv ]; then cp selfplay_result.csv "$BACKUP"/; fi
# echo "backup: $BACKUP"

# 2. モデルと学習データを初期化 (ONNXも消す: 残すと旧モデルで自己対戦してしまう)
rm -f model/*.h5 model/*.save model/*.pt model/*.onnx
rm -f data/selfplay*.json data/resolved*.json data/const.json
rm -f count*.json
rm -f history.csv selfplay_result.csv log.log

# 3. ランダム初期化した新規モデルを作成 (.venv の Python 3.12 で学習を回す構成)
../.venv/bin/python single_network.py

# 4. 新規モデルのONNXを書き出す (C++側が読むのはこちら)
../.venv/bin/python export_onnx.py

# 5. 最新のC++バイナリを配置 (旧 reset.sh の ../ai/build/ は build_onnx/ に変わった)
cp ../ai/build/cpp_animal_shogi ./

echo "reset done: model/ を確認してください"
ls -l model/
