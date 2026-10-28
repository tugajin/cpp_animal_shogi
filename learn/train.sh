#!/bin/sh

for i in `seq 0 30`
do
    ./cpp_animal_shogi 20
    #./cpp_animal_shogi 200000
    #python3 line.py
    python3 train.py $i
    python3 export_onnx.py
    rm -rf data/selfplay*.json
done
