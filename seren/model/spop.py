import numpy as np
import torch

class SessionPop(object):
    def __init__(self, conf, params, logger):
        '''
        Session popularity predictor that gives higher scores to items with higher number of occurrences in the session. 
        Ties are broken up by adding the popularity score of the item.

        Parameters
        ----------
        pop_n : int
            Only give back non-zero scores to the top N ranking items. 
            Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
        '''        
        self.top_n = params['pop_n']
        self.item_key = conf['item_key']
        self.session_key = conf['session_key']
        self.logger = logger

    def fit(self, train, valid_loader=None):
        grp = train.groupby(self.item_key)
        self.pop_list = grp.size()
        self.pop_list = self.pop_list / (self.pop_list + 1)
        self.pop_list.sort_values(ascending=False, inplace=True)
        self.pop_list = self.pop_list.head(self.top_n)

    def predict(self, test, k=15):
        preds, last_item = torch.tensor([]), torch.tensor([])
        for seq, target in test:
            cands_idx = ~np.in1d(self.pop_list.index, seq)
            pred = torch.tensor([self.pop_list.index[cands_idx][:k].tolist()])
            preds = torch.cat((preds, pred), 0)
            last_item = torch.cat((last_item, torch.tensor(target)), 0)

        return preds, last_item

            