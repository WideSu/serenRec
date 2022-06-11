import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy

class SKNN(object):
    def __init__(self,config):
        '''
        SessionKNN

        Parameters
        -----------
        k : int
            Number of neighboring session to calculate the item scores from, default is 100.
        asymmetric_alpha : float
            When using asymmetric cosine similarity, it permits ad-hoc optimizations of the similarity function for the domain of interest, default is 0.5.
        similarity : String
            The function to use for calculating the similarity, default is 'jaccard'. 
        shrink : float
            The value added to the similarity denominator which ensure that it != 0, default is 0.
        normalize : bool
            Whether to divide the dot product by the product of the norms, default is True.
        item_num : int
            The number of unique items in training set. 
        session_num : int
            The number of sessions in the training set.
        

        '''
        self.asymmetric_alpha = config['asymmetric_alpha']
        self.similarity = config['similarity']
        self.k = config['k']
        self.shrink = config['shrink'] 
        self.normalize = config['normalize'] 

        self.item_num = config['item_num']
        self.session_num = config['session_num']

        self.asymmetric_cosine = False

    def fit(self, train_loader):
        '''
        Training interface for KNN

        Parameters
        ----------
        train_loader : generator
            A generator to yied session list and corresponding next item one-by-one, such as [1,2,3,0,0], [4]
        '''
        current_session_idx = 0
        self.train_matrix = sp.lil_matrix((self.session_num, self.item_num + 1))
        
        for item_seq, next_item in train_loader:
            if current_session_idx >= self.session_num:
                print(f'More sessions than expected, current maximum number of recording session: {self.session_num}')
                break
            item_seq = item_seq + next_item
            for c in item_seq:
                if c != 0:
                    self.train_matrix[current_session_idx, c] += 1       
            current_session_idx += 1
        self.train_matrix = self.train_matrix.tocsr()
        self.binary_train_matrix = self.train_matrix.copy()
        self.binary_train_matrix.data = np.ones_like(self.binary_train_matrix.data)

    def rank(self, test_loader, topk=50):


        res_scs, res_ids = [], []
        for item_seq in test_loader:
            new_session = sp.lil_matrix(1, self.item_num + 1)
            for c in item_seq:
                if c != 0:
                    new_session[0, c] += 1
            new_session = new_session.tocsr()
            if self.similarity == 'jaccard':
                binary_new_session = new_session.copy()
                binary_new_session.data = np.ones_like(binary_new_session.data)
                sim_vec = self._compute_jaccard(binary_new_session, self.binary_train_matrix)
            else:
                sim_vec = self._compute_cosine(new_session, self.train_matrix)
            # K-NN for (session_num) array
            top_k_idx = (-sim_vec).argpartition(self.k-1)[0:self.k]
            mask = np.zeros_like(sim_vec)
            mask[top_k_idx] = 1
            sim_vec = sim_vec * mask
            sim_vec = sp.csr_matrix(sim_vec)  # 1 * session_num
            score = sim_vec.dot(self.binary_train_matrix).A.squeeze() # (1, session_num) (session_num, item_num) -> (1, item_num)
            ids = np.argsort(score[1:])[::-1]
            ids += 1
            scs = score[ids]
            if topk is not None and topk <= self.item_num:
                ids, scs = ids[:topk], scs[:topk]

            res_ids.append(ids)
            res_scs.append(scs)

    
    def _compute_cosine(self, session1, sessions):
        numerator = session1.dot(sessions.T).A.squeeze() # (1, item_num) (item_num, session_num) -> (1, session_num) -> (session_num,)
        s1_norm = np.sqrt(session1.power(2).sum()) # single value
        ss_norm = np.sqrt(sessions.power(2).sum(axis=1).A).reshape(-1)  # (session_num, 1) array -> (session_num,) array
        if self.asymmetric_cosine:
            s1_norm = np.power(s1_norm + 1e-6, 2 * self.asymmetric_alpha)
            ss_norm = np.power(ss_norm + 1e-6, 2 * (1 - self.asymmetric_alpha))
        denominator = s1_norm * ss_norm
        sim = numerator / denominator

        return sim

    def _compute_jaccard(self, session1, sessions):
        self.binary_train_matrix
        pass
  
            
            