#!/bin/sh

for i in `seq 0 60`
do
    ./cpp_animal_shogi 5000
    python3 line.py
    python3 train.py $i
    rm -rf data/selfplay*.json
done
