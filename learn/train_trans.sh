#!/bin/sh
while true
do
	./cpp_animal_shogi 4500 || exit 1
	python3 train_worker_trans.py || exit 1
done
