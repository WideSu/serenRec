import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class SessionDataset(Dataset):    
    def __init__(self, seq_list=None, next_list=None, sample_cnt=0, item_id_map=None):
        self.sample_cnt = sample_cnt
        self.df = pd.DataFrame()
        ks = [int(k) for k in item_id_map.keys()]
        sampler = np.random.randint(low=min(ks), high=max(ks) + 1, size=(sample_cnt, len(self.next_list)))
        if sample_cnt == 0:
            self.df['sequence'] = seq_list
            self.df['next'] = next_list
        else:
            for i in range(sample_cnt):
                tmp_df = pd.DataFrame()
                tmp_df['sequence'] = seq_list
                tmp_df['next'] = next_list
                tmp_df['sampler'] = sampler[i]
                self.df = pd.concat([self.df, tmp_df])
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.sample_cnt == 0:
            return torch.tensor(eval(self.df['sequence'].values[idx])), torch.tensor(self.df['next'].values[idx]), torch.tensor(0)
        else:
            return torch.tensor(eval(self.df['sequence'].values[idx])), torch.tensor(self.df['next'].values[idx]), torch.tensor(self.df['sampler'].values[idx])
    
    def get_loader(self):
        for idx in range(len(self.df)):
            if self.sample_cnt == 0:
                yield eval(self.df['sequence'].values[idx]), self.df['next'].values[idx], 0
            else:
                yield eval(self.df['sequence'].values[idx]), self.df['next'].values[idx], self.df['sampler'].values[idx]
