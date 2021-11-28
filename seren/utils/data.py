import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

class Interactions(object): 
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.user_key = config['user_key']
        self.item_key = config['item_key'] 
        self.session_key = config['session_key']
        self.time_key = config['time_key']

        self._process_flow()

    def _process_flow(self):
        self._load_data()
        self._make_sessions()
        self._core_filter()
        self._reindex()

        self.user_num = self.df[self.user_key].nunique()
        self.item_num = self.df[self.item_key].nunique()

        self.logger.info(f'Finish loading {self.dataset_name} data, current length is: {len(self.df)}, user number: {self.user_num}, item number: {self.item_num}')

    def _load_data(self):
        '''
        load raw data to dataframe and rename columns as required

        Parameters
        ----------
        user_key : str, optional
            column to present users, by default 'user_id'
        item_key : str, optional
            column to present items, by default 'item_id'
        time_key : str, optional
            column to present timestamp, by default 'ts'
        '''        
        dataset_name = self.config['dataset']
        self.dataset_name = dataset_name
        
        if not os.path.exists(f'./dataset/{dataset_name}/'):
            self.logger.error('unexisted dataset...')
        if dataset_name == 'ml-100k':
            df = pd.read_csv(
                './dataset/ml-100k/u.data', 
                delimiter='\t', 
                names=[self.user_key, self.item_key, 'rating', self.time_key]
            )
        else:
            self.logger.error(f'cannot load data: {dataset_name}')
            raise ValueError(f'cannot load data: {dataset_name}')

        self.df = df


    def _set_map(self, df, key):
        codes = pd.Categorical(df[key]).codes + 1
        res = dict(zip(df[key], codes))
        return res, codes

    def _make_sessions(self, is_ordered=True):
        if is_ordered:
            self.df.sort_values(
                by=[self.user_key, self.time_key], 
                ascending=True, 
                inplace=True
            )

        self.df['date'] = pd.to_datetime(self.df[self.time_key], unit='s').dt.date

        # check whether the day changes
        split_session = self.df['date'].values[1:] != self.df['date'].values[:-1]
        split_session = np.r_[True, split_session]
        # check whether the user changes
        new_user = self.df[self.user_key].values[1:] != self.df[self.user_key].values[:-1]
        new_user = np.r_[True, new_user]
        # a new sessions stars when at least one of the two conditions is verified
        new_session = np.logical_or(new_user, split_session)
        # compute the session ids
        session_ids = np.cumsum(new_session)
        self.df[self.session_key] = session_ids

        self.logger.info(f'Finish making {self.session_key} for data')

    def _core_filter(self, pop_num=5, bad_sess_len=1, user_sess_num=5, user_num_good_sess=200):
        # drop duplicate interactions within the same session
        self.df.drop_duplicates(
            subset=[self.user_key, self.item_key, self.time_key], 
            keep='first', 
            inplace=True
        )
        # TODO this is totally different from daisy filter-core
        # keep items with >=pop_num interactions 
        item_pop = self.df[self.item_key].value_counts()
        good_items = item_pop[item_pop >= pop_num].index
        self.df = self.df[self.df[self.item_key].isin(good_items)].reset_index(drop=True)

        # remove sessions with length < bad_sess_len
        session_length = self.df[self.session_key].value_counts()
        good_sessions = session_length[session_length > bad_sess_len].index
        self.df = self.df[self.df.session_id.isin(good_sessions)]

        # let's keep only returning users (with >= 5 sessions) and remove overly active ones (>=200 sessions)
        sess_per_user = self.df.groupby(self.user_key)[self.session_key].nunique()
        good_users = sess_per_user[(sess_per_user >= user_sess_num) & (sess_per_user < user_num_good_sess)].index
        self.df = self.df[self.df[self.user_key].isin(good_users)]

        self.user_num = self.df[self.user_key].nunique()
        self.item_num = self.df[self.item_key].nunique()

        self.logger.info(f'Finish filtering data, current length is: {len(self.df)}, user number: {self.user_num}, item number: {self.item_num}')

    def _reindex(self):
        self.used_items = self.df[self.item_key].unique()
        self.user_map, self.df[self.user_key] = self._set_map(self.df, self.user_key)
        self.item_map, self.df[self.item_key] = self._set_map(self.df, self.item_key)
  
    
    def get_seq_from_df(self):
        dic = self.df[[self.user_key, self.session_key]].drop_duplicates()
        seq = []
        for u, s in dic.values:
            items = self.df.query(f'{self.user_key} == {u} and {self.session_key} == {s}')[self.item_key].tolist()
            seq.append([u, s, items])

        return seq


class Categories(object):
    # TODO read category info, used for IDLS
    def __init__(self, item_map, item_set, config, logger, category_key='category_vec'):
        self.config = config
        self.logger = logger

        self.item_set = item_set
        self.item_num = len(item_set) + 1
        self.item_key = config['item_key'] 
        self.item_map = item_map
        self.category_key = category_key

        self._process_flow()

    def _process_flow(self):
        self._load_data()
        self._reindex()
        self._one_hot()
        self._generate_cat_mat()

    def _load_data(self):
        dataset_name = self.config['dataset']
        self.dataset_name = dataset_name
        
        if not os.path.exists(f'./dataset/{dataset_name}/'):
            self.logger.error('unexisted dataset...')
        if dataset_name == 'ml-100k':
            df = pd.read_csv(
                './dataset/ml-100k/u.item', 
                delimiter='|', 
                header=None,
                encoding="ISO-8859-1"
            )

            genres = df.iloc[:,6:].values  # TODO not consider 'unknown'
            df[self.category_key] = pd.Series(genres.tolist())
            df.rename(columns={0: self.item_key}, inplace=True)
            df = df[[self.item_key, self.category_key]].copy()
            df = df[df[self.item_key].isin(self.item_set)].reset_index()
            
            self.n_cates = len(df[self.category_key][0]) + 1

        else:
            self.logger.error(f'cannot load item information data: {dataset_name}')
            raise ValueError(f'cannot load item information data: {dataset_name}')

        self.df = df

    def _reindex(self):
        self.df[self.item_key] = self.df[self.item_key].map(self.item_map)

    def _generate_cat_mat(self):
        item_cate_matrix = torch.zeros(self.item_num, self.n_cates)
        self.df.sort_values(by=self.item_key, inplace=True)
        item_cate = torch.tensor(self.df[self.category_key])
        item_cate_matrix[1:, 1:] = item_cate

        self.item_cate_matrix = item_cate_matrix

    def _one_hot(self):
        '''
        TODO
        some dataset categories are not like ml-100k, 
        so we need to process these data to one-hot vector
        '''        
        # item_cate_matrix[1:, :] = F.one_hot(item_cate, num_classes=self.n_cates)  
        pass

