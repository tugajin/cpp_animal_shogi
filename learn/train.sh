#!/bin/sh

export LD_LIBRARY_PATH="${HOME}/onnxruntime-1.20.0/lib:${LD_LIBRARY_PATH}"

for i in `seq 0 30`
do
    #./cpp_animal_shogi 20
    ./cpp_animal_shogi 2000
    #python3 line.py
    python3 train.py $i
    python3 export_onnx.py
    rm -rf data/selfplay*.json
    rm -rf data/*.tmp
done
