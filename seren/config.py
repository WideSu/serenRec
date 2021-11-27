import logging

ACC_KPI = ['ndcg', 'mrr', 'hr']

CAT_COLS = {
    'ml-100k': []
}

MAX_LEN = 10

def get_logger(file_name):
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)

    # set two handlers
    log_file = file_name + '.log'

    fileHandler = logging.FileHandler(log_file, mode = 'w')
    fileHandler.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)

    # set formatter
    formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    # add
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)

    # logger.info("test")

    return logger

def get_parameters(args):
    params = {
        'user_key': args.user_key,
        'item_key': args.item_key,
        'session_key': args.session_key,
        'time_key': args.time_key,
        # 'split_ratio': 0.8,
        'output_path': f'./temp/{args.dataset}/{args.desc}/',
        'description': args.desc,
        'dataset': args.dataset,
        'session_len': MAX_LEN,
        'topk': args.topk,
    }

    model_params = {
        'batch_size': args.batch_size,
        'item_embedding_dim': args.item_embedding_dim,
        'hidden_size': args.hidden_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'l2': args.l2,
        'lr_dc_step': args.lr_dc_step,
        'lr_dc': args.lr_dc,
        'n_layers': args.n_layers,

    }

    return params, model_params
