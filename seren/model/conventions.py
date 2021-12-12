import pandas as pd
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

class ItemKNN(object):
    def __init__(self, conf, params, logger):
        '''        
        Item-to-item predictor that computes the the similarity to all items to the given item.
        
        Similarity of two items is given by:
        
        .. math::
            s_{i,j}=\sum_{s}I\{(s,i)\in D & (s,j)\in D\} / (supp_i+\\lambda)^{\\alpha}(supp_j+\\lambda)^{1-\\alpha}
            
        Parameters
        --------
        n_sims : int
            Only give back non-zero scores to the N most similar items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
        lambda : float
            Regularization. Discounts the similarity of rare items (incidental co-occurrences). (Default value: 20)
        alpha : float
            Balance between normalizing with the supports of the two items. 0.5 gives cosine similarity, 1.0 gives confidence (as in association rules).
        '''    
        self.n_sims = params['n_sims']
        self.lmbd = params['lambda']
        self.alpha = params['alpha']
        self.item_key = conf['item_key']
        self.session_key = conf['session_key']
        self.time_key = conf['time_key']
        self.logger = logger

    def fit(self, data):
        data.set_index(np.arange(len(data)), inplace=True)
        itemids = data[self.item_key].unique()
        n_items = len(itemids) 
        data = pd.merge(
            data, 
            pd.DataFrame({self.item_key: itemids, 'ItemIdx': np.arange(len(itemids))}), 
            on=self.item_key, how='inner')
        sessionids = data[self.session_key].unique()
        data = pd.merge(
            data, 
            pd.DataFrame({self.session_key: sessionids, 'SessionIdx': np.arange(len(sessionids))}), 
            on=self.session_key, how='inner')
        supp = data.groupby('SessionIdx').size()
        session_offsets = np.zeros(len(supp) + 1, dtype=np.int32)
        session_offsets[1:] = supp.cumsum()
        index_by_sessions = data.sort_values(['SessionIdx', self.time_key]).index.values
        supp = data.groupby('ItemIdx').size()
        item_offsets = np.zeros(n_items + 1, dtype=np.int32)
        item_offsets[1:] = supp.cumsum()
        index_by_items = data.sort_values(['ItemIdx', self.time_key]).index.values
        self.sims = dict()
        for i in range(n_items):
            iarray = np.zeros(n_items)
            start = item_offsets[i]
            end = item_offsets[i+1]
            for e in index_by_items[start:end]:
                uidx = data.SessionIdx.values[e]
                ustart = session_offsets[uidx]
                uend = session_offsets[uidx+1]
                user_events = index_by_sessions[ustart:uend]
                iarray[data.ItemIdx.values[user_events]] += 1
            iarray[i] = 0
            norm = np.power((supp[i] + self.lmbd), self.alpha) * np.power((supp.values + self.lmbd), (1.0 - self.alpha))
            norm[norm == 0] = 1
            iarray = iarray / norm
            indices = np.argsort(iarray)[-1:-1-self.n_sims:-1]
            self.sims[itemids[i]] = pd.Series(data=iarray[indices], index=itemids[indices])

    def predict(self, test, k=15):
        preds, last_item = torch.tensor([]), torch.tensor([])

        for seq, target in test:
            cands_idx = ~np.in1d(self.sims[seq[-1]].index, seq)
            pred = torch.tensor([self.sims[seq[-1]].index[cands_idx][:k].tolist()])
            preds = torch.cat((preds, pred), 0)
            last_item = torch.cat((last_item, torch.tensor(target)), 0)

        return preds, last_item