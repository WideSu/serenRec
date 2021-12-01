import argparse
import logging


from seren.utils.data import Interactions, Categories
from seren.config import get_parameters, get_logger, ACC_KPI
from seren.utils.functions import build_seqs, get_seq_from_df
from seren.utils.model_selection import fold_out
from seren.utils.dataset import NARMDataset, SRGNNDataset
from seren.utils.metrics import accuracy_calculator, diversity_calculator
from seren.model.narm import NARM
from seren.model.srgnn import SessionGraph

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="srgnn", type=str)
parser.add_argument("--user_key", default="user_id", type=str)
parser.add_argument("--item_key", default="item_id", type=str)
parser.add_argument("--session_key", default="session_id", type=str)
parser.add_argument("--time_key", default="timestamp", type=str)
parser.add_argument("--dataset", default="ml-100k", type=str)
parser.add_argument("--desc", default="nothing", type=str)
parser.add_argument("--topk", default=15, type=int)

parser.add_argument('--batch_size', type=int, default=128, help='batch size for loader')
parser.add_argument('--item_embedding_dim', type=int, default=100, help='dimension of item embedding')
parser.add_argument('--hidden_size', type=int, default=100, help='dimension of linear layer')
parser.add_argument('--epochs', type=int, default=20, help='training epochs number')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--n_layers', type=int, default=1, help='the number of gru layers')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')

args = parser.parse_args()

conf, model_conf = get_parameters(args)
logger = get_logger(__file__.split('.')[0] + f'_{conf["description"]}')

ds = Interactions(conf, logger)
train, test = fold_out(ds.df, conf)
train, valid = fold_out(train, conf)

train_sequences = build_seqs(get_seq_from_df(train, conf), conf['session_len'])
valid_sequences = build_seqs(get_seq_from_df(valid, conf), conf['session_len'])
test_sequences = build_seqs(get_seq_from_df(test, conf), conf['session_len'])

if conf['model'] == 'narm':
    train_dataset = NARMDataset(train_sequences, logger)
    valid_dataset = NARMDataset(valid_sequences, logger)
    test_dataset = NARMDataset(test_sequences, logger)
    logger.info(f'Length of Train: {len(train_dataset)}, Validation: {len(valid_dataset)}, Test: {len(test_dataset)}')
    # logger.debug(ds.item_num)
    train_loader = train_dataset.get_loader(model_conf, shuffle=True)
    valid_loader = valid_dataset.get_loader(model_conf, shuffle=False)
    test_loader = test_dataset.get_loader(model_conf, shuffle=False)
    model = NARM(ds.item_num, model_conf, logger)
    model.fit(train_loader, valid_loader)
    preds, truth = model.predict(test_loader, conf['topk'])
elif conf['model'] == 'srgnn':
    train_dataset = SRGNNDataset(train_sequences, shuffle=True)
    valid_dataset = SRGNNDataset(valid_sequences, shuffle=False)
    test_dataset = SRGNNDataset(test_sequences, shuffle=False)
    model = SessionGraph(ds.item_num, model_conf, logger)
    model.fit(train_dataset, valid_dataset)
    preds, truth = model.predict(test_dataset, conf['topk'])
else:
    logger.error('Invalid model name')
    raise ValueError('Invalid model name')


logger.info(f"Finish predicting, start calculating {conf['model']}'s KPI...")
metrics = accuracy_calculator(preds, truth, ACC_KPI)
foo = [f'{ACC_KPI[i].upper()}: {metrics[i]:5f}' for i in range(len(ACC_KPI))]
logger.info(f'{" ".join(foo)}')

cats = Categories(ds.item_map, ds.used_items, conf, logger)
diveristy = diversity_calculator(preds, cats.item_cate_matrix)
logger.info(f'Diversity: {diveristy:4f}')
