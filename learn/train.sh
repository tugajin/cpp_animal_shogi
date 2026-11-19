#!/bin/sh

export LD_LIBRARY_PATH="${HOME}/onnxruntime-1.20.0/lib:${LD_LIBRARY_PATH}"

KEEP_GENERATIONS=5

for i in `seq 0 30`
do
    ./cpp_animal_shogi 10000
    python3 train.py $i
    python3 export_onnx.py

    # 直近N世代分のデータを保持（古いものから削除）
    rm -rf data/*.tmp
    gen_dirs=$(ls -dt data/gen_* 2>/dev/null)
    mkdir -p "data/gen_${i}"
    mv data/selfplay*.json "data/gen_${i}/" 2>/dev/null
    gen_count=$(ls -d data/gen_* 2>/dev/null | wc -l)
    if [ "$gen_count" -gt "$KEEP_GENERATIONS" ]; then
        ls -dt data/gen_* | tail -n +$((KEEP_GENERATIONS + 1)) | xargs rm -rf
    fi
done
