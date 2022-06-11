import numpy as np
import pandas as pd
import scipy.sparse as sp

class SKNN(object):
    def __init__(self,config):
        '''
        SessionKNN
        Parameters
        -----------
        k : int
            Number of neighboring session to calculate the item scores from. (Default value: 100)
        asymmetric_alpha:
        similarity:

        '''
        self.asymmetric_alpha = config['asymmetric_alpha'] # 0.5
        self.similarity = config['similarity'] # 'cosine'
        self.k = config['k'] # 50
        self.shrink = config['shrink'] # 0
        self.normalize = config['normalize'] # True
        self.asymmetric_cosine = config['asymmetric_cosine']
        self.session_key = config['session_key']
        self.item_key = config['item_key']
        self.time_key = config['time_key']

        self.item_num = config['item_num']
        self.session_num = config['session_num']

        self.jaccard_correlation = False
        self.pearson_correlation = False

    def fit(self, train_loader):
        '''
        Training interface for KNN

        Parameters
        ----------
        train_loader : _generator_
            agenerator to yied session sequence (list) one-by-one
        '''
        current_session_idx = 0
        self.binary_train_matrix = sp.lil_matrix((self.session_num, self.item_num + 1), dtype=np.int8)
        self.train_matrix = sp.lil_matrix((self.session_num, self.item_num + 1))
        
        for item_seq in train_loader:
            if current_session_idx >= self.session_num:
                print(f'More sessions than expected, current maximum number of recording session: {self.session_num}')
                break
            ids = np.zeros((len(item_seq), 2), dtype=np.int8)
            ids[:, 1] = np.array(item_seq)
            ids[:, 0] = np.ones_like(item_seq) * current_session_idx
            for r, c in ids:
                self.binary_train_matrix[r, c] = 1
                self.train_matrix[r, c] += 1
            current_session_idx += 1
        # finish filling the binary interaction matrix
        self.binary_train_matrix = self.binary_train_matrix.tocsr()
        self.train_matrix = self.train_matrix.tocsr()

    def rank(self, test_loader):
        if self.similarity == 'jaccard':
            self.normalize = False
            self.jaccard_correlation=True
        elif self.similarity == 'pearson':
            self.pearson_correlation = True

        if self.pearson_correlation:
            self.apply_pearson_correlation()

        if self.jaccard_correlation:
            self.use_boolean_interactions()


        current_session_idx = self.train_matrix.shape[0]
        for item_seq in test_loader:
            idx, cnts = np.unique(item_seq, return_counts=True)
            new_session = np.array(self.item_num + 1)
            new_session[idx] = cnts
            
            