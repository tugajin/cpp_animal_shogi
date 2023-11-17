#!/bin/sh

for i in `seq 0 31`
do
    ./cpp_animal_shogi 10
    python3 train_trans.py $i
    rm -rf data/selfplay*.json
done
