#!/bin/sh
while true
do
	./cpp_animal_shogi 18000 || exit 1
	python3 train_worker.py || exit 1
done
