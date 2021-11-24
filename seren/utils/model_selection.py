def fold_out(data, args, split_ratio=0.8, clean_test=True, min_session_length=3, time_aware=True):
    '''
    user-level fold-out split

    Parameters
    ----------
    data : pd.DataFrame
        dataframe waiting for split
    args : dict
        parameters dictionary
    split_ratio : float
        ratio for train set
    clean_test : bool, optional
        whether to remove items not occur in train and bad sessions after split, by default True
    min_session_length : int, optional
        determin length of bad sessions, by default 3
    time_aware : bool, optional
        whether sort by time, by default True

    Returns
    -------
    tuple of pd.DataFrame
        train and test dataframe
    '''    
    if time_aware:
        data = data.sort_values(by=[args['user_key'], args['time_key']])
    else:
        data = data.sort_values(by=[args['user_key']])
    user_sessions = data.groupby(args['user_key'])[args['session_key']]

    train_session_ids = set()
    for _, session_ids in user_sessions:
        split_point = int(split_ratio * len(session_ids))
        u_sess = set(session_ids[:split_point])
        train_session_ids = train_session_ids | u_sess
    train = data[data.session_id.isin(train_session_ids)].copy()
    test = data[~data.session_id.isin(train_session_ids)].copy()

    if clean_test:
        #  remove items in test not occur in train and remove sessions in test shorter than min_session_length
        train_items = train[args['item_key']].unique()
        slen = test[args['session_key']].value_counts()
        good_sessions = slen[slen >= min_session_length].index
        test = test.query(f"{args['session_key']} in @good_sessions and {args['item_key']} in @train_items").copy()

    return train, test

