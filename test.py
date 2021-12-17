import torch
import argparse

from seren.utils.data import Interactions, Categories
from seren.config import get_parameters, get_logger, ACC_KPI
from seren.utils.model_selection import fold_out, train_test_split
from seren.utils.dataset import NARMDataset, SRGNNDataset, GRU4RECDataset, ConventionDataset
from seren.utils.metrics import accuracy_calculator, diversity_calculator
from seren.model.narm import NARM
from seren.model.srgnn import SessionGraph
from seren.model.gru4rec import GRU4REC
from seren.model.conventions import Pop, SessionPop, ItemKNN, BPRMF

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="bprmf", type=str)
parser.add_argument("--user_key", default="user_id", type=str)
parser.add_argument("--item_key", default="item_id", type=str)
parser.add_argument("--session_key", default="session_id", type=str)
parser.add_argument("--time_key", default="timestamp", type=str)
parser.add_argument("--dataset", default="ml-100k", type=str)
parser.add_argument("--desc", default="nothing", type=str)
parser.add_argument("--topk", default=15, type=int)
parser.add_argument("-seed", type=int, default=22, help="Seed for random initialization") #Random seed setting

parser.add_argument('--batch_size', type=int, default=128, help='batch size for loader')
parser.add_argument('--item_embedding_dim', type=int, default=100, help='dimension of item embedding')
parser.add_argument('--hidden_size', type=int, default=100, help='dimension of linear layer')
parser.add_argument('--epochs', type=int, default=20, help='training epochs number')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=1e-5, help='l2/BPR penalty')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--n_layers', type=int, default=1, help='the number of gru layers')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument("-sigma", type=float, default=None, help="init weight -1: range [-sigma, sigma], -2: range [0, sigma]") # weight initialization [-sigma sigma] in literature
parser.add_argument('--dropout_input', default=0, type=float) #0.5 for TOP and 0.3 for BPR
parser.add_argument('--dropout_hidden', default=0, type=float) #0.5 for TOP and 0.3 for BPR
parser.add_argument('--optimizer', default='Adagrad', type=str)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--momentum', default=0, type=float)
parser.add_argument('--eps', default=1e-6, type=float) #not used
parser.add_argument('--final_act', default='tanh', type=str)
parser.add_argument('--loss_type', default='BPR-max', type=str) #type of loss function TOP1 / BPR for GRU4REC, TOP1-max / BPR-max for GRU4REC+
parser.add_argument('--pop_n', type=int, default=100, help='top popular N items')
parser.add_argument('--n_sims', type=int, default=100, help='non-zero scores to the N most similar items given back')
parser.add_argument('--lmbd', type=float, default=20, help='Regularization. Discounts the similarity of rare items')
parser.add_argument('--alpha', type=float, default=0.5, help='Balance between normalizing with the supports of the two items')
parser.add_argument('--lambda_session', type=float, default=0, help='session embedding penalty')
parser.add_argument('--lambda_item', type=float, default=0, help='item embedding penalty')

args = parser.parse_args()

if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

conf, model_conf = get_parameters(args)
logger = get_logger(__file__.split('.')[0] + f'_{conf["description"]}')
# logger = get_logger(f'_{conf["description"]}')

ds = Interactions(conf, logger)

if conf['model'] == 'narm':
    train, test = fold_out(ds.df, conf)
    train, valid = fold_out(train, conf)

    train_dataset = NARMDataset(train, conf)
    valid_dataset = NARMDataset(valid, conf)
    test_dataset = NARMDataset(test, conf)
    # logger.debug(ds.item_num)
    train_loader = train_dataset.get_loader(model_conf, shuffle=True)
    valid_loader = valid_dataset.get_loader(model_conf, shuffle=False)
    test_loader = test_dataset.get_loader(model_conf, shuffle=False)
    model = NARM(ds.item_num, model_conf, logger)
    model.fit(train_loader, valid_loader)
    preds, truth = model.predict(test_loader, conf['topk'])
elif conf['model'] == 'srgnn':
    train, test = fold_out(ds.df, conf)
    train, valid = fold_out(train, conf)

    train_dataset = SRGNNDataset(train, conf, shuffle=True)
    valid_dataset = SRGNNDataset(valid, conf, shuffle=False)
    test_dataset = SRGNNDataset(test, conf, shuffle=False)
    model = SessionGraph(ds.item_num, model_conf, logger)
    model.fit(train_dataset, valid_dataset)
    preds, truth = model.predict(test_dataset, conf['topk'])
elif conf['model'] == 'gru4rec':
    train, test = fold_out(ds.df, conf)
    train, valid = fold_out(train, conf)

    suitable_batch = min(
        model_conf['batch_size'],
        train[conf['session_key']].nunique(), 
        test[conf['session_key']].nunique(), 
        valid[conf['session_key']].nunique()
    )
    if suitable_batch < model_conf['batch_size']:
        logger.warning(
            f'Currrent batch size {model_conf["batch_size"]} is not suitable, the maximum tolerance for batch size is {suitable_batch}')
        model_conf['batch_size'] = suitable_batch

    train_loader = GRU4RECDataset(train, conf, model_conf['batch_size'])
    valid_loader = GRU4RECDataset(valid, conf, model_conf['batch_size'])
    test_loader = GRU4RECDataset(test, conf, model_conf['batch_size'])
    model = GRU4REC(ds.item_num, model_conf, logger)
    model.fit(train_loader, valid_loader)
    preds, truth = model.predict(test_loader, conf['topk'])
elif conf['model'] == 'pop':
    train, test = train_test_split(ds.df, conf, logger) #fold_out(ds.df, conf)
    test_dataset = ConventionDataset(test, conf)
    model = Pop(conf, model_conf, logger)
    model.fit(train)
    preds, truth = model.predict(test_dataset)
elif conf['model'] == 'spop':
    train, test = train_test_split(ds.df, conf, logger) #fold_out(ds.df, conf)
    test_dataset = ConventionDataset(test, conf)
    model = SessionPop(conf, model_conf, logger)
    model.fit(train)
    preds, truth = model.predict(test_dataset)
elif conf['model'] == 'itemknn':
    train, test = train_test_split(ds.df, conf, logger)
    test_dataset = ConventionDataset(test, conf)
    model = ItemKNN(conf, model_conf, logger)
    model.fit(train)
    preds, truth = model.predict(test_dataset)
elif conf['model'] == 'bprmf':
    train, test = train_test_split(ds.df, conf, logger)
    test_dataset = ConventionDataset(test, conf)
    model = BPRMF(conf, model_conf, logger)
    model.fit(train)
    preds, truth = model.predict(test_dataset)
else:
    logger.error('Invalid model name')
    raise ValueError('Invalid model name')

logger.info(f"Finish predicting, start calculating {conf['model']}'s KPI...")
metrics = accuracy_calculator(preds, truth, ACC_KPI)
foo = [f'{ACC_KPI[i].upper()}: {metrics[i]:5f}' for i in range(len(ACC_KPI))]
logger.info(f'{" ".join(foo)}')

#cats = Categories(ds.item_map, ds.used_items, conf, logger)
#diveristy = diversity_calculator(preds, cats.item_cate_matrix)
#logger.info(f'Diversity: {diveristy:4f}')
