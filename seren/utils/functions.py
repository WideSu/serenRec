import os
import numpy as np
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
    user_seqs, targets = [], []
    for seq in seqs:
        items = seq[2]
        tmp_len = len(items) if len(items) <= max_len else max_len
        for j in range(tmp_len - 1):
            targets.append([items[j + 1]])
            user_seq = items[0:j + 1]
            user_seqs.append(user_seq)

        if len(items) > max_len:
            for j in range(max_len, len(items)):
                targets.append([items[j]])
                user_seq = items[j - max_len + 1:j]
                user_seqs.append(user_seq)
    
    return user_seqs, targets

def get_seq_from_df(df, args):
    dic = df[[args['user_key'], args['session_key']]].drop_duplicates()
    seq = []
    for u, s in dic.values:
        items = df.query(f'{args["user_key"]} == {u} and {args["session_key"]} == {s}')[args["item_key"]].tolist()
        seq.append([u, s, items])

    return seq

