#!/bin/sh

for i in `seq 0 30`
do
    #./cpp_animal_shogi 2000
    ./cpp_animal_shogi 200000
    #python3 line.py
    python3 train.py $i
    rm -rf data/selfplay*.json
done
