import numpy as np

def fold_out(data, args, split_ratio=0.8, clean_test=True, min_session_length=3, time_aware=True, train_items=None):
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
        # remove items in test not occur in train and remove sessions in test shorter than min_session_length
        train_items1 = train[args['item_key']].unique() if train_items is None else train_items
        slen = test[args['session_key']].value_counts()
        good_sessions = slen[slen >= min_session_length].index
        test = test[test[args['session_key']].isin(good_sessions) & test[args['item_key']].isin(train_items1)].copy()

    return train, test

def train_test_split(data, args, logger, clean_test=True, min_session_length=2, n_days=1):
    Time = args['time_key']
    SessionId = args['session_key']
    ItemId = args['item_key']
    tmax = data[Time].max()
    session_max_times = data.groupby(SessionId)[Time].max()
    session_train = session_max_times[session_max_times < tmax - (86400*n_days)].index
    session_test = session_max_times[session_max_times >= tmax - (86400*n_days)].index
    train = data[np.in1d(data[SessionId], session_train)]
    test = data[np.in1d(data[SessionId], session_test)]
    if clean_test:
        test = test[np.in1d(test[ItemId], train[ItemId])]
        tslength = test.groupby(SessionId).size()
        test = test[np.in1d(test[SessionId], tslength[tslength>=min_session_length].index)]
    logger.info('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train[SessionId].nunique(), train[ItemId].nunique()))
    #train.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_train_full.txt', sep='\t', index=False)
    logger.info('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test[SessionId].nunique(), test[ItemId].nunique()))
    #test.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_test.txt', sep='\t', index=False)
    
    return train, test
