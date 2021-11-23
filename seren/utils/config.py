CAT_COLS = {
    'ml-100k': []
}


def get_parameters(args):
    params = {
        'user_key': args.user_id,
        'item_key': args.item_id,
        'session_key': args.session_id,
        'time_key': args.timestamp,
        'split_ratio': 0.8,
        'output_path': f'./temp/{args.dataset}/{args.desc}/',
        'description': args.desc,
        'dataset': args.dataset,
        'session_len': 10,
    }

    model_params = {
        
    }

    return params
