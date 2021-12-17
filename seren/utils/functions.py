import os
import torch
import numpy as np
import networkx as nx
import pandas as pd
from logging import getLogger

logger = getLogger()

def to_csv(data, name, params):
    op = params['output_path']
    if not os.path.exists(op):
        os.makedirs(op)
    if isinstance(data, pd.DataFrame):
        data.to_csv(f'{op}{name}.csv', index=False)
    else:
        logger.error(f'Invalid variable to write into {op}')

def build_seqs(seqs, max_len):
    user_seqs, targets, sess = [], [], []
    for seq in seqs:
        s_id = seq[0]
        items = seq[1]
        tmp_len = len(items) if len(items) <= max_len else max_len
        for j in range(tmp_len - 1):
            targets.append([items[j + 1]])
            user_seq = items[0:j + 1]
            user_seqs.append(user_seq)
            sess.append(s_id)

        if len(items) > max_len:
            for j in range(max_len, len(items)):
                targets.append([items[j]])
                user_seq = items[j - max_len + 1:j]
                user_seqs.append(user_seq)
                sess.append(s_id)
    
    return user_seqs, targets, sess

def get_seq_from_df(df, args):
    dic = df[args['session_key']].drop_duplicates() #[args['user_key'], ]
    seq = []
    # u,
    for s in dic.values:
        #items = df.query(f'{args["user_key"]} == {u} and {args["session_key"]} == {s}')[args["item_key"]].tolist()
        items = df.query(f'{args["session_key"]} == {s}')[args["item_key"]].tolist()
        seq.append([s, items])

    return seq


def pad_zero_for_seq(data):
    '''
    pad sessions to max length

    Parameters
    ----------
    data : List
        results generated by get_seq_from_df()

    Returns
    -------
        padded vectors, labels and lengths of each session before padding
    '''    
    data.sort(key=lambda x: len(x[0]), reverse=True)  # x[0][0]
    labels = []
    lens = [len(sess) for sess, _ in data]  # sess[0]
    padded_sesss_item = torch.zeros(len(data), max(lens)).long()  # zero-pad
    for i, (sess, label) in enumerate(data):
        padded_sesss_item[i, :lens[i]] = torch.LongTensor(sess)
        labels.append(label)

    padded_sesss_item = padded_sesss_item.transpose(0, 1)
    return padded_sesss_item, torch.tensor(labels).long(), lens


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph
