# ====================
# パラメータ更新部
# ====================

# パッケージのインポート
from single_network import *
from pathlib import Path
from oracle_dataset import *
from torch.utils.data import DataLoader
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

# パラメータの準備
RN_EPOCHS = 32 # 学習回数
RN_BATCH_SIZE = 512 # バッチサイズ
LAMBDA = 0.7

# ネットワークの学習
def train_network(epoch_num=RN_EPOCHS, batch_size=RN_BATCH_SIZE):

    # ベストプレイヤーのモデルの読み込み
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SingleNet()
    model.load_state_dict(torch.load('./model/best_single.h5',device))
    model = model.to(device)
    
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.00001)
    dataset = OracleDataset()
    dataset_len = len(dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True) 
    start = time.time()
    for i in range(epoch_num):
        print(f"epoch:{i}")
        sum_loss = 0
        sum_num = 0 
        for x, y in dataloader:
            x = x.float().to(device)
            y = y.float().to(device)
                
            optimizer.zero_grad()
            outputs = model(x)
            outputs = torch.squeeze(outputs)
            loss =  torch.sum((outputs - y) ** 2)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            sum_num += 1
            if sum_num % 1000 == 0:
                n = batch_size * sum_num
                now = time.time()
                print(f"{n}/{dataset_len} ({100 * (n/dataset_len):.3f}%) loss:{sum_loss/sum_num} sec:{int(now-start)}")

        print(f"avg loss:{sum_loss / sum_num}")
        # 最新プレイヤーのモデルの保存
        # updateが遅いので吐くようにする
        torch.save(model.state_dict(), './model/latest_single.h5')

def check_train_data():
    # 学習データの読み込み
    history = load_all_data()
    print(type(history))
    for hist in history:
        x, y_deep = zip(hist)
        state = State(x[0][0],x[0][1])
        print(state)
        print(f"deep:{y_deep}")
        print("------------------------")

# 動作確認
if __name__ == '__main__':
    train_network()
    #check_train_data()
