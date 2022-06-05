from turtle import forward
import torch
import torch.nn as nn
from tqdm import tqdm

class SessionPop(nn.Module):
    def __init__(self, config):
        super(SessionPop, self).__init__()
        self.item_cnt_ref = torch.tensor(1 + config['item_num'])

    def forward(self, item_seq):
        idx, cnt = torch.unique(item_seq, return_counts=True)
        return idx, cnt
        

    def fit(self, train_loader):
        
        pbar = tqdm(train_loader)
        for item_seq, _ in pbar:
            idx, cnt = self.forward(item_seq)
            self.item_cnt_ref[idx] += cnt

    def predict(self, input_ids, next_item):
        pass


    def rank(self, test_loader,topk=50):
        pass
