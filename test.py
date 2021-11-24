import argparse


from seren.utils.dataset import Interactions
from seren.utils.config import get_parameters, get_logger
from seren.utils.functions import build_seqs, get_seq_from_df
from seren.utils.model_selection import fold_out

logger = get_logger(__file__.split('.')[0])

parser = argparse.ArgumentParser()
parser.add_argument("--user_key", default="user_id", type=str)
parser.add_argument("--item_key", default="item_id", type=str)
parser.add_argument("--session_key", default="session_id", type=str)
parser.add_argument("--time_key", default="timestamp", type=str)
parser.add_argument("--dataset", default="ml-100k", type=str)
parser.add_argument("--desc", default="nothing", type=str)

args = parser.parse_args()
conf = get_parameters(args)

ds = Interactions(conf, logger)
train, test = fold_out(ds.df, conf)
train, valid = fold_out(train, conf)

train_sequences = build_seqs(get_seq_from_df(train, conf), conf['session_len'])
valid_sequences = build_seqs(get_seq_from_df(valid, conf), conf['session_len'])
test_sequences = build_seqs(get_seq_from_df(test, conf), conf['session_len'])


