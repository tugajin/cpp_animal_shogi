# ====================
# 学習サイクルの実行
# ====================

# パッケージのインポート
from pathlib import Path
from single_network import single_network
from train_network import train_network
from evaluate_network import *
#from evaluate_best_player import *
import multiprocessing as mp
import sys
import torch
import random
import time

def load_selfplay_data():
    path_list = list(Path('./data').glob('*.json'))
    return path_list

def clean_selfplay_data():
    path_list = list(Path('./data').glob('*.json'))
    for p in path_list:
        p.unlink(True)

if __name__ == '__main__':

    mp.set_start_method('spawn')

    args = sys.argv
    self_play_num = 30
    epoch_num = 10 
    batch_size = 64 
    if len(args) >= 4:
        self_play_num = int(args[1])
        epoch_num = int(args[2])
        batch_size = int(args[3])

    print("GPU") if torch.cuda.is_available() else print("CPU")

    # デュアルネットワークの作成
    print(f"selfplay:{self_play_num}")
    print(f"epoch:{epoch_num}")
    print(f"batch:{batch_size}")
    i = 0
    while True:
        print('Train',i,'====================')
        # 自己対局
        
        load_data_list = load_selfplay_data(self_play_num)
        # パラメータ更新部
        train_network(epoch_num, batch_size, load_data_list)
        # 新パラメータ評価部
        update_best_player()
        conv_jit()
        clean_selfplay_data(self_play_num)
        i += 1
