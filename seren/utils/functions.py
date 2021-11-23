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
    for seq in range(seqs):
        items = seq[2]
        tmp_len = len(items) if len(items) <= max_len else max_len
        for j in range(tmp_len - 1):
            targets.append([items[j + 1]])
            user_seq = items[0:j + 1]
            user_seqs.append(user_seq)

        if len(items) > max_len:
            for j in range(max_len, len(items)):
                targets.append([items[j + 1]])
                user_seq = items[j - max_len:j + 1]
                user_seqs.append(user_seq)
    
    return user_seqs, targets

