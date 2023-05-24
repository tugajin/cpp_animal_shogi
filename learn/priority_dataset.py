from pathlib import Path
import json
from single_network import *
import numpy as np
import time
from game import *
from numpy.random import choice
from history_dataset import HistoryDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class PriorityDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        data = []
        all_weight = 0
        for path in root:
            with path.open(mode='r') as f:
                try:
                    d = json.loads(f.read())
                    data.extend(d)
                except json.decoder.JSONDecodeError:
                    print(f"decode error:{path}")
                    pass
        sum_delta = 0
        for d in data:
            sum_delta += d["d"]
        weight = [ d["d"]/sum_delta for d in data ]
        data = choice(data, 1000000, p=weight)
        data2 = []
        for d in data:
            data2.append([d["p"], d["s"]])
        self.data = data2
        #print(f"len:{len(data)}")
     # ここで取り出すデータを指定している
    def __getitem__(self, index) :
        data = self.data[index][0]
        state = from_hash(data)
        file, rank, channel = DN_INPUT_SHAPE
        feature = np.array(state.feature())
        feature = feature.reshape(channel, file, rank)
        y_deep = self.data[index][1]
        return feature, y_deep

    # この method がないと DataLoader を呼び出す際にエラーを吐かれる
    def __len__(self):
        return len(self.data)
if __name__ == '__main__':
    history_path = sorted(Path('./data').glob('*.json'))
    dataset = PriorityDataset(history_path) 
 
