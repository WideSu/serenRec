import os
import numpy as np
import pandas as pd
from logging import getLogger

class Interactions(object): 
    def __init__(self, config):
        self.config = config
        self.logger = getLogger()

        self.user_key = config['user_key']
        self.item_key = config['item_key'] 
        self.session_key = config['session_key']
        self.time_key = config['time_key']

        self._process_flow()

    def _process_flow(self):
        self._load_data()
        self._make_sessions()
        self._core_filter()

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

        self.user_map, df[self.user_key] = self._set_map(df, self.user_key)
        self.item_map, df[self.item_key] = self._set_map(df, self.item_key)

        self.df = df

    def _set_map(self, df, key):
        codes = pd.Categorical(df[key]).codes
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

    def get_seq_from_df(self, max_len=10):
        dic = self.df[[self.user_key, self.session_key]].drop_duplicates()
        seq = []
        for u, s in dic.values:
            items = self.df.query(f'{self.user_key} == {u} and {self.session_key} == {s}')[self.item_key].tolist()
            seq.append([u, s, items])

        return seq


class Categories(object):
    # TODO read category info, used for IDLS
    def __init__(self, config):
        self.config = config
        self.logger = getLogger()

        self.user_key = config['user_key']
        self.item_key = config['item_key'] 
        self.session_key = config['session_key']
        self.time_key = config['time_key']

        self._process_flow()
