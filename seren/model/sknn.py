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
@inproceedings{sarwar2001item,
  title={Item-based collaborative filtering recommendation algorithms},
  author={Sarwar, Badrul and Karypis, George and Konstan, Joseph and Riedl, John},
  booktitle={Proceedings of the 10th international conference on World Wide Web},
  pages={285--295},
  year={2001}
}
@inproceedings{aiolli2013efficient,
  title={Efficient top-n recommendation for very large scale binary rated datasets},
  author={Aiolli, Fabio},
  booktitle={Proceedings of the 7th ACM conference on Recommender systems},
  pages={273--280},
  year={2013}
}
'''

import numpy as np
import scipy.sparse as sp

class SKNN(object):
    def __init__(self,config):
        '''
        SessionKNN Recommender, the similarity for session-based KNN can only be Jaccard,
        Cosine similarity could be calculated if there are some duplicated items in one
        session. Otherwise, Cosine is kind of similar to Jaccard

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
        self.k = config['k']
        self.shrink = config['shrink'] 
        self.normalize = config['normalize'] 
        self.asymmetric_alpha = config['asymmetric_alpha']
        self.similarity = config['similarity']

        self.item_num = config['item_num']
        self.session_num = config['session_num']

        self.asymmetric_cosine = True if self.similarity == 'asymmetric' else False

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
            if current_session_idx > self.session_num:
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

    def predict(self, input_ids, next_item):
        # TODO
        res_ids,res_scs = [], []
        sim_vec = self._compute_similarity(input_ids)
        score = sim_vec.dot(self.binary_train_matrix).A.squeeze() # (1, session_num) (session_num, item_num) -> (1, item_num)
        ids = np.argsort(score[1:])[::-1]
        ids += 1
        scs = score[ids]
        if self.k is not None and self.k <= self.item_num:
            ids, scs = ids[:self.k], scs[:self.k]

        res_ids.append(ids)
        res_scs.append(scs)
        pass

    def _compute_similarity(self, input_ids):
        # TODO 先把rank里昨晚说过的部分提取到这, 然后再写predict
        new_session = sp.lil_matrix(1, self.item_num + 1)
        for c in input_ids:
            if c != 0:
                new_session[0, c] += 1
        new_session = new_session.tocsr()
        if self.similarity == 'jaccard':
            binary_new_session = new_session.copy()
            binary_new_session.data = np.ones_like(binary_new_session.data)
            sim_vec = self._compute_jaccard(binary_new_session, self.binary_train_matrix)
        else:
            sim_vec = self._compute_cosine(new_session, self.train_matrix)

        if self.normalize:
            sim_vec = sim_vec / np.sum(sim_vec)

        # K-NN for (session_num) array
        top_k_idx = (-sim_vec).argpartition(self.k-1)[0:self.k]
        mask = np.zeros_like(sim_vec)
        mask[top_k_idx] = 1
        sim_vec = sim_vec * mask
        sim_vec = sp.csr_matrix(sim_vec)  # 1 * session_num

        return sim_vec


    def rank(self, test_loader, topk=50):
        res_scs, res_ids = [], []
        for item_seq,_ in test_loader:
            sim_vec = self._compute_similarity(item_seq)
            score = sim_vec.dot(self.binary_train_matrix).A.squeeze() # (1, session_num) (session_num, item_num) -> (1, item_num)
            ids = np.argsort(score[1:])[::-1]
            ids += 1
            scs = score[ids]
            if topk is not None and topk <= self.item_num:
                ids, scs = ids[:topk], scs[:topk]

            res_ids.append(ids)
            res_scs.append(scs)
        return res_ids, res_scs

    def _compute_cosine(self, session1, sessions):
        '''
        cosine similarity = \frac{\sum \limits_{i \in I(s1, s2)}count_{s_1,i} * count_{s_2,i}}{\sqrt{\sum \limits_{i \in I(s1, s2)}count_{s_1, i}^2}  \sqrt{\sum \limits_{i \in I(s1, s2)}count_{s_2, i}^2} } 
        '''
        numerator = session1.dot(sessions.T).A.squeeze() # (1, item_num) (item_num, session_num) -> (1, session_num) -> (session_num,)
        s1_norm = np.sqrt(session1.power(2).sum()) # single value
        ss_norm = np.sqrt(sessions.power(2).sum(axis=1).A).reshape(-1)  # (session_num, 1) array -> (session_num,) array
        if self.asymmetric_cosine:
            s1_norm = np.power(s1_norm + self.shrink + 1e-6, 2 * self.asymmetric_alpha)
            ss_norm = np.power(ss_norm + self.shrink + 1e-6, 2 * (1 - self.asymmetric_alpha))
        denominator = s1_norm * ss_norm
        sim = numerator / denominator

        return sim

    def _compute_jaccard(self, session1, sessions):
        '''
        jaccard similarity = \frac{I_{s_1} \cap I_{s_2}}{I_{s_1} \cup I_{s_2}}
        '''
        nominator = session1.dot(self.binary_train_matrix.T).A.squeeze()
        batch_sum = (sp.vstack([session1 for _ in range(self.session_num)]) + sessions)
        batch_sum.data = np.ones_like(batch_sum.data)
        denominator = batch_sum.sum(axis = 1).A.reshape(-1)
        denominator = denominator + self.shrink + 1e-6
        sim = nominator / denominator

        return sim  

# TODO sknn里的所有TODO都弄完后, 整理一下那个itemknn, 保证保留下的cosine, pearson, jaccard similarity代码逻辑和那个recsys2019的一致, 等我周日晚上弄完实习了再和你核对下
            