#!/bin/sh
while true
do
	./cpp_animal_shogi 4500
	python3 train_worker.py
done
