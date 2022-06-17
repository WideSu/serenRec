import math
import json
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

class Preprocess():
    @staticmethod
    def clean_review_data(df=None):
        """
            This static method is used to clean steam review data
        :param df: The dataframe of steam review data. (pandas.DataFrame, default value: None)
        :return: The pandas.DataFrame class data, only with three columns ('s_id', 'product_id', 'time_id')
        """
        # Remove all the unnecessary columns
        df.drop(['found_funny', 'compensation', 
                 'text', 'page', 'page_order', 
                 'hours', 'early_access', 'products', 
                 'user_id'], axis=1, inplace=True)
        # Generate user_id to replace username
        tmp_df = pd.DataFrame(df['username'].value_counts())
        max_length = len(tmp_df['username'])
        tmp_df['user_id'] = [i for i in range(1, max_length + 1)]
        tmp_df = pd.DataFrame({
            "username": list(tmp_df.index),
            "user_id": list(tmp_df['user_id'])
        })
        data = pd.merge(left=df,
                        right=tmp_df,
                        how='left',
                        on='username')
        data.drop('username', axis=1, inplace=True)
        # Generate session ID
        data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d")
        data['product_id'] = data['product_id'].astype(int)
        data.sort_values(by=['date', 'user_id'], inplace=True)
        tmp_df = data.groupby(['date', 'user_id']).size().reset_index(name='Freq')
        tmp_df['s_id'] = [i for i in range(0, len(tmp_df['user_id']))]
        # Generate time_id
        tmp_list = []
        for num in list(tmp_df['Freq']):
            order_list = list(range(1, num + 1))
            tmp_list += order_list
        tmp_df.drop('Freq', axis=1, inplace=True)
        data = pd.merge(left=data,
                        right=tmp_df,
                        how='left',
                        on=['date', 'user_id'])
        data['time_id'] = tmp_list
        data.drop(['user_id', 'date'], axis=1, inplace=True)
        data = data.reindex(columns=['s_id', 'product_id', 'time_id'])
        return data
    
    @staticmethod
    def filter_review_data(df=None, popularity_num=1, min_session_len=1):
        """
            This static method is used to filter steam review data, only keep popular product_id and session with multiple games.
        :param df: The dataframe of steam review data. (pandas.DataFrame, default value: None)
        :param popularity_num: The minimum number of times id appears in different sessions. (int, default value: 1)
        :param min_session_len: The minimum length of a session. (int, default value: 1)
        :return: The pandas.DataFrame class data, only with three columns ('s_id', 'product_id', 'time_id')
        """
        tmp_df = df.groupby(['product_id'])['s_id'].size().reset_index(name='counts')
        tmp_df = tmp_df[tmp_df['counts'] >= popularity_num]
        df = df[df['product_id'].isin(list(tmp_df['product_id']))]
        tmp_df = df.groupby(['s_id'])['product_id'].size().reset_index(name='counts')
        tmp_df = tmp_df[tmp_df['counts'] > min_session_len]
        df = df[df['s_id'].isin(list(tmp_df['s_id']))]
        return df
    
    @staticmethod
    def item_map(df=None):
        """
            This static method is used to generate mapping relationship between product_id and item_id
        :param df: The dataframe of steam review data. (pandas.DataFrame, default value: None)
        :return: The pandas.DataFrame class data, only with three columns ('s_id', 'item_id', 'time_id') and 
                 dictionary of mapping relationship between product_id and item_id
        """
        tmp_list = df['product_id'].unique()
        id_list = list(range(1, len(tmp_list) + 1))
        tmp_df = pd.DataFrame()
        tmp_df['product_id'] = tmp_list
        tmp_df['item_id'] = id_list
        t_df = tmp_df.set_index('item_id')
        tmp_dict = t_df.to_dict()
        mapping_dict = tmp_dict['product_id']
        raw_df = pd.merge(left=df,
                          right=tmp_df,
                          how='left',
                          on=['product_id'])
        raw_df.drop(['product_id'], axis=1, inplace=True)
        raw_df = raw_df.reindex(columns=['s_id', 'item_id', 'time_id'])
        return raw_df, mapping_dict
    
    @staticmethod
    def to_sequence(df=None, max_session_len=5):
        """
            This static method is used to generate sequence, do dropout and do augmentation
        :param df: The dataframe of steam review data. (pandas.DataFrame, default value: None)
        :param max_session_len: The length of max session. (int, default value: 5)
        :return: The pandas.DataFrame class data.
        """
        tmp_df = df.groupby(['s_id'])['item_id'].size().reset_index(name='counts')
        tmp_df = tmp_df[tmp_df['counts'] > max_session_len]
        merge_data = pd.merge(left=df,
                              right=tmp_df,
                              how='left',
                              on=['s_id'])
        raw_data = merge_data.dropna()
        raw_data.drop(['counts'], axis=1, inplace=True)
        s_id_list = raw_data['s_id'].unique()
        seq_list = []
        next_list = []
        for s_id in s_id_list:
            tmp_df = raw_data[raw_data['s_id'] == s_id]
            s_id_len = len(tmp_df)
            for index in range(s_id_len - max_session_len):
                tmp_list = list(tmp_df['item_id'])
                seq_list.append(json.dumps(tmp_list[index:index + max_session_len]))
                next_list.append(tmp_list[index + max_session_len])
        seq_df = pd.DataFrame()
        seq_df['sequence'] = seq_list
        seq_df['next'] = next_list
        return seq_df
    
    @staticmethod
    def train_test_split(seq_df=None, split_rate=0.8):
        """
            This static method is used to split training data and test data
        :param df: The dataframe of steam review data need to be split. (pandas.DataFrame, default value: None)
        :param split_rate: The split ratio, (float, default value: 0.8)
        :return: Two pandas.DataFrame class data. [training DataFrame, test DataFrame]
        """
        training_data, test_data = train_test_split(seq_df, train_size=split_rate, shuffle=True)
        return training_data, rest_data

    @staticmethod
    def drop_and_aug(df=None, drop_flag=False, drop_ratio=0.05, aug_flag=True, max_session_len=5):
        """
            This static method is used to generate sequence, do dropout and do augmentation
        :param df: The dataframe of steam review data. (pandas.DataFrame, default value: None)
        :param drop_flag: The dropout flag. (boolean, default value: False)
        :param drop_ratio: The ratio of dropping out operation. (float, default value: 0.05)
        :param aug_flag: The augmentation flag. (boolean, default value: True)
        :param max_session_len: The length of max session. (int, default value: 5)
        :return: The pandas.DataFrame class data.
        """
        if drop_flag:
            index_list = list(np.arange(0, len(df)))
            random.shuffle(index_list)
            drop_index = index_list[:math.ceil(drop_ratio * len(index_list))]
            for index in drop_index:
                tmp_seq = df[df.index == index]['sequence'].item()
                tmp_seq = json.loads(tmp_seq)
                for i in range(len(tmp_seq)):
                    if random.random() > 0.5:
                        tmp_seq[i] = 0
                if sum(tmp_seq) == 0:
                    df.drop(df[df.index == index].index, inplace=True)
                else:
                    tmp_seq = [i for i in tmp_seq if i != 0]
                    tmp_seq += [0] * (max_session_len - len(tmp_seq))
                    tmp_seq = json.dumps(tmp_seq)
                    df['sequence'][index] = tmp_seq
        if aug_flag:
            drop_df = df.copy()
            seq_list = []
            next_list = []
            loop_list = list(drop_df.index)
            for _id in loop_list:
                tmp_df = drop_df[drop_df.index == _id]
                tmp_list = json.loads(tmp_df['sequence'].item())
                tmp_list = [0] * (max_session_len - 1) + tmp_list
                s_id_len = len(tmp_list)
                for index in range(s_id_len - max_session_len):
                    seq = tmp_list[index:index + max_session_len]
                    seq = [i for i in seq if i != 0]
                    seq += [0] * (max_session_len - len(seq))
                    seq_list.append(json.dumps(seq))
                    next_list.append(tmp_list[index + max_session_len])
            aug_df = pd.DataFrame()
            aug_df['sequence'] = seq_list
            aug_df['next'] = next_list
            df = aug_df
        return df
