'''
@article{ludewig2018evaluation,
  title={Evaluation of session-based recommendation algorithms},
  author={Ludewig, Malte and Jannach, Dietmar},
  journal={User Modeling and User-Adapted Interaction},
  volume={28},
  number={4},
  pages={331--390},
  year={2018},
  publisher={Springer}
}
'''
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

class SessionPop(nn.Module):
    def __init__(self, config):
        super(SessionPop, self).__init__()
        self.item_cnt_ref = torch.zeros(1 + config['item_num']) # the index starts from 1
        self.top_n = config['top_n']
        self.max_len = config['max_len']

    def forward(self, item_seq):
        #predict as in the classic popularity model:
        idx, cnt = torch.unique(item_seq, return_counts=True)
        return idx, cnt

    def fit(self, train_loader):
        pbar = tqdm(train_loader)
        for item_seq,_ in pbar:
            idx, cnt = self.forward(item_seq)
            self.item_cnt_ref[idx] += cnt
        self.item_score =  self.item_cnt_ref/(1+self.item_cnt_ref)

    def predict(self, input_ids, next_item):
        item_cnt_ref = self.item_cnt_ref.clone()
        next_item = torch.tensor(next_item)
        seq_max_len = torch.zeros(self.max_len)
        seq_input = torch.tensor(input_ids)
        seq_input = pad_sequence([seq_input, seq_max_len], batch_first=True)[0]
        idx, cnt = torch.unique(seq_input, return_counts=True) # 
        item_cnt_ref[idx] += (cnt + self.item_score[idx])
        return item_cnt_ref[next_item]


    def rank(self, test_loader, topk=50):

        pass