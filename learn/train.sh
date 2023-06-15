#!/bin/sh
while true
do
	./cpp_animal_shogi 4000
	python3 train_worker.py
done
