from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import json
from single_network import *
import numpy as np
import time
from game import *
class OracleDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.win_data = np.fromfile("../data/win.dat", dtype='uint64')
        self.lose_data = np.fromfile("../data/lose.dat", dtype='uint64')
        self.draw_data = np.fromfile("../data/draw.dat", dtype='uint64')
        
        self.win_data = self.win_data.astype(np.int64)
        self.lose_data = self.lose_data.astype(np.int64)
        self.draw_data = self.draw_data.astype(np.int64)
        
        self.win_size = len(self.win_data)
        self.lose_size = len(self.lose_data)
        self.draw_size = len(self.draw_data)
    # ここで取り出すデータを指定している
    def __getitem__(self, index) :
        if index < self.win_size:
            data = self.win_data[index]
            label = 1
        elif index >= self.win_size and index < (self.win_size + self.lose_size):
            data = self.lose_data[index - self.win_size]
            label = -1
        else:
            data = self.draw_data[index - self.win_size - self.lose_size]
            label = 0
        state = from_hash(data)
        file, rank, channel = DN_INPUT_SHAPE
        feature = np.array(state.feature())
        feature = feature.reshape(channel, file, rank)
        return feature, label
    # この method がないと DataLoader を呼び出す際にエラーを吐かれる
    def __len__(self):
        return self.win_size + self.lose_size + self.draw_size

if __name__ == '__main__':
    dataset = OracleDataset()
 
