import numpy as np
import scipy.sparse as sp
import pandas as pd

class SKNN(object):
    def _init_(self,config):
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

        self.jaccard_correlation = False
        self.pearson_correlation = False


    def fit(self, train):
        '''
        train: DataFrame(SessionID, ItemID,TimeStamp)
            the ItemID starts from 1, but sessionID can start from 1
        '''
        train.sort_values([self.session_key], inplace=True)
        tmp = train.groupby([self.session_key, self.item_key])[self.time_key].count().reset_index()
        session_num = train[self.session_key].nunique()
        item_num = train[self.item_key].nunique()
        
        self.train = train
        self.binary_train_matrix = sp.csr_matrix(
            (np.ones_like(tmp[self.time_key].values), (tmp[self.session_key], tmp[self.item_key])), 
            shape=(session_num, item_num + 1))
        self.train_ids = list(train[self.session_key].unique())

    def predict(self, input_ids, next_item):
        pass

   
    def apply_pearson_correlation(self):
        """
        Remove from every data point the average for the corresponding column
        """
        self.data_matrix = self.data_matrix.tocsc()
        interactions_per_col = np.diff(self.data_matrix.indptr)
        nonzero_cols = interactions_per_col > 0
        sum_per_col = np.asarray(self.data_matrix.sum(axis=0)).ravel()
        col_average = np.zeros_like(sum_per_col)
        col_average[nonzero_cols] = sum_per_col[nonzero_cols] / interactions_per_col[nonzero_cols]
        # Split in blocks to avoid duplicating the whole data structure
        start_col = 0
        end_col= 0
        block_size = 800
        while end_col < self.n_cols:
            end_col = min(self.n_cols, end_col + block_size)
            self.data_matrix.data[self.data_matrix.indptr[start_col]:self.data_matrix.indptr[end_col]] -= \
                np.repeat(col_average[start_col:end_col], interactions_per_col[start_col:end_col])
            start_col += block_size

    def use_boolean_interactions(self):
        # Split in blocks to avoid duplicating the whole data structure
        start_pos = 0
        end_pos= 0
        block_size = 800
        while end_pos < len(self.data_matrix.data):
            end_pos = min(len(self.data_matrix.data), end_pos + block_size)
            self.data_matrix.data[start_pos:end_pos] = np.ones(end_pos - start_pos)
            start_pos += block_size

    def rank(self, test, topk=None, candidates=None):
        '''
        test: DataFrame(SessionID, ItemID,TimeStamp)
            the SessionID starts from number_train_sessions + 1
        '''
        values = []
        rows = []
        cols = []
        processed_items = 0
        # start_col=None
        # end_col=None
        block_size = 100

        data = pd.concat([self.train, test], ignore_index=True) # the min session_idx of the test must be one larger than the max of session_idx of train data and auto increasing.
        session_num = data[self.session_key].nunique()
        item_num = data[self.item_key].nunique()
        self.data_matrix = sp.csr_matrix((data[self.time_key], (data[self.session_key], data[self.item_key])), shape=(session_num, item_num + 1)).T

        self.n_rows, self.n_cols = self.data_matrix.shape
        if self.similarity == 'jaccard':
            self.normalize = False
            self.jaccard_correlation=True
        elif self.similarity == 'pearson':
            self.pearson_correlation = True

        if self.pearson_correlation:
            self.apply_pearson_correlation()

        if self.jaccard_correlation:
            self.use_boolean_interactions()

        # Compute sum of squared values to be used in normalization, denomitor
        sum_of_squared = np.array(self.data_matrix.power(2).sum(axis=0)).ravel()
        if not self.jaccard_correlation:
            sum_of_squared = np.sqrt(sum_of_squared)

        if self.asymmetric_cosine:
             # this is for asymmetric cosine proposed by recsys 2013 todo, default is False
            sum_of_squared_to_alpha = np.power(sum_of_squared + 1e-6, 2 * self.asymmetric_alpha)
            sum_of_squared_to_1_minus_alpha = np.power(sum_of_squared + 1e-6, 2 * (1 - self.asymmetric_alpha))

        self.data_matrix = self.data_matrix.tocsc()
        start_col_local = 0
        end_col_local = self.n_cols
        # if start_col is not None and start_col>0 and start_col<n_cols:
        #     start_col_local = start_col

        # if end_col is not None and end_col>start_col_local and end_col<n_cols:
        #     end_col_local = end_col

        start_col_block = start_col_local
        this_block_size = 0

        # Compute all similarities for each item using vectorization
        #range = end_col_local - start_col_block
        while start_col_block < end_col_local:
            # Compute block first and last column
            end_col_block = min(start_col_block + block_size, end_col_local)
            this_block_size = end_col_block - start_col_block

            # All data points for a given item
            item_data = self.data_matrix[:, start_col_block:end_col_block]
            item_data = item_data.toarray()
            this_block_weights = self.data_matrix.T.dot(item_data)
            for col_index_in_block in range(this_block_size):
                if this_block_size == 1:
                    this_column_weights = this_block_weights.ravel()
                else:
                    this_column_weights = this_block_weights[:,col_index_in_block]
                column_index = col_index_in_block + start_col_block
                this_column_weights[column_index] = 0.0

                # Apply normalization and shrinkage, ensure denominator != 0
                if self.normalize: # Pearson and cosine
                    if self.asymmetric_cosine:
                        denominator = sum_of_squared_to_alpha[column_index] * sum_of_squared_to_1_minus_alpha + self.shrink + 1e-6
                    else:
                        denominator = sum_of_squared[column_index] * sum_of_squared + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)
                elif self.jaccard_correlation:
                    denominator = sum_of_squared[column_index] + sum_of_squared - this_column_weights + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)
                # If no normalization or tanimoto is selected, apply only shrink
                elif self.shrink != 0:
                    this_column_weights = this_column_weights / self.shrink

                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                relevant_items_partition = (-this_column_weights).argpartition(self.k-1)[0:self.k]
                # - Sort only the relevant items
                relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                # - Get the original item index
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix, do not add zeros
                not_zeros_mask = this_column_weights[top_k_idx] != 0.0
                num_not_zeros = np.sum(not_zeros_mask)

                values.extend(this_column_weights[top_k_idx][not_zeros_mask])
                rows.extend(top_k_idx[not_zeros_mask])
                cols.extend(np.ones(num_not_zeros) * column_index)

            # Add previous block size
            start_col_block += this_block_size
            processed_items += this_block_size
        # End while on columns 
        sim_matrix = sp.csr_matrix((values, (rows, cols)),
                                        shape=(self.n_cols, self.n_cols),
                                        dtype=np.float32)  # total_session_num * total_session_num
        res_ids, res_scs = np.array([]), np.array([])
        for test_idx in test[self.session_key].unique():
            sim_vec = sim_matrix[test_idx, self.train_ids]
            scores = sim_vec.dot(self.binary_train_matrix).toarray().squeeze()
            ids = np.argsort(scores[1:])[::-1][:topk] + 1
            scs = scores[ids]
            res_ids = np.vstack((res_ids, ids))
            res_scs = np.vstack((res_scs, scs))

        return res_ids, res_scs