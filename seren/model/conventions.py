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
        for seq, target, _ in test:
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

        for seq, target, _ in test:
            cands_idx = ~np.in1d(self.sims[seq[-1]].index, seq)
            pred = torch.tensor([self.sims[seq[-1]].index[cands_idx][:k].tolist()])
            preds = torch.cat((preds, pred), 0)
            last_item = torch.cat((last_item, torch.tensor(target)), 0)

        return preds, last_item

class BPRMF(object):
    def __init__(self, conf, params, logger, init_normal=True):
        '''
        BPR(n_factors = 100, n_iterations = 10, learning_rate = 0.01, lambda_session = 0.0, lambda_item = 0.0, sigma = 0.05, init_normal = False, session_key = 'SessionId', item_key = 'ItemId')
        
        Bayesian Personalized Ranking Matrix Factorization (BPR-MF). During prediction time, the current state of the session is modelled as the average of the feature vectors of the items that have occurred in it so far.
            
        Parameters
        --------
        n_factor : int
            The number of features in a feature vector. (Default value: 100)
        n_iterations : int
            The number of epoch for training. (Default value: 10)
        learning_rate : float
            Learning rate. (Default value: 0.01)
        lambda_session : float
            Regularization for session features. (Default value: 0.0)
        lambda_item : float
            Regularization for item features. (Default value: 0.0)
        sigma : float
            The width of the initialization. (Default value: 0.05)
        init_normal : boolean
            Whether to use uniform or normal distribution based initialization.
        session_key : string
            header of the session ID column in the input file (default: 'SessionId')
        item_key : string
            header of the item ID column in the input file (default: 'ItemId')
        '''
        self.n_factors = params['item_embedding_dim']
        self.n_iterations = params['epochs']
        self.learning_rate = params['learning_rate']
        self.lambda_session = params['lambda_session']
        self.lambda_item = params['lambda_item']
        self.sigma = params['sigma'] if params['sigma'] is not None else 0.05
        self.session_key = conf['session_key']
        self.item_key = conf['item_key']
        self.logger = logger

        self.init_normal = init_normal

    def init(self):
        self.U = np.random.rand(self.n_sessions, self.n_factors) * 2 * self.sigma - self.sigma if not self.init_normal else np.random.randn(self.n_sessions, self.n_factors) * self.sigma
        self.I = np.random.rand(self.n_items, self.n_factors) * 2 * self.sigma - self.sigma if not self.init_normal else np.random.randn(self.n_items, self.n_factors) * self.sigma
        self.bU = np.zeros(self.n_sessions)
        self.bI = np.zeros(self.n_items)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def update(self, uidx, p, n):
        uF = np.copy(self.U[uidx,:])
        iF1 = np.copy(self.I[p,:])
        iF2 = np.copy(self.I[n,:])
        sigm = self.sigmoid(iF1.T.dot(uF) - iF2.T.dot(uF) + self.bI[p] - self.bI[n])
        c = 1.0 - sigm
        self.U[uidx,:] += self.learning_rate * (c * (iF1 - iF2) - self.lambda_session * uF)
        self.I[p,:] += self.learning_rate * (c * uF - self.lambda_item * iF1)
        self.I[n,:] += self.learning_rate * (-c * uF - self.lambda_item * iF2)
        return np.log(sigm)

    def fit(self, data):
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)

        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)

        data = pd.merge(data, pd.DataFrame({self.item_key:itemids, 'ItemIdx':np.arange(self.n_items)}), on=self.item_key, how='inner')
        data = pd.merge(data, pd.DataFrame({self.session_key:sessionids, 'SessionIdx':np.arange(self.n_sessions)}), on=self.session_key, how='inner')     
        self.init()

        for it in range(self.n_iterations):
            c = []
            for e in np.random.permutation(len(data)):
                uidx = data.SessionIdx.values[e]
                iidx = data.ItemIdx.values[e]
                iidx2 = data.ItemIdx.values[np.random.randint(self.n_items)]
                err = self.update(uidx, iidx, iidx2)
                c.append(err)
            self.logger.info(f'training epoch: {it + 1}\tTrain Loss: {np.mean(c):.3f}')
            
    
    def predict(self, test, k=15):
        preds, last_item = torch.tensor([]), torch.tensor([])
        self.iditemmap = pd.Series(data=self.itemidmap.index, index=self.itemidmap.values)

        for seq, target, _ in test:
            iidx = self.itemidmap[seq].values
            uF = self.I[iidx].mean(axis=0)
            pred_iidx = np.argsort(self.I.dot(uF))[::-1]
            cands_idx = ~np.in1d(pred_iidx, iidx)
            pred_iidx = pred_iidx[cands_idx][:k]
            pred = torch.tensor([self.iditemmap[pred_iidx].values.tolist()])
            preds = torch.cat((preds, pred), 0)
            last_item = torch.cat((last_item, torch.tensor(target)), 0)

        return preds, last_item
