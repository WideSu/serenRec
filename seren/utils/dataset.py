import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .functions import pad_zero_for_seq, build_seqs, get_seq_from_df


class NARMDataset(Dataset):
    def __init__(self, data, conf):
        '''
        Session sequences dataset class

        Parameters
        ----------
        data : pd.DataFrame
            dataframe by Data.py
        logger : logging.logger
            Logger used for recording process
        '''     
        # self.data is list of [[seqs],[targets]]   
        self.data = build_seqs(get_seq_from_df(data, conf), conf['session_len'])
        
    def __getitem__(self, index):
        session_items = self.data[0][index]
        target_item = self.data[1][index]
        return session_items, target_item

    def __len__(self):
        return len(self.data[1])


    def get_loader(self, args, shuffle=True):
        loader = DataLoader(
            self, 
            batch_size=args['batch_size'], 
            shuffle=shuffle, 
            collate_fn=pad_zero_for_seq
        )

        return loader


class SPOPDataset(object):
    def __init__(self, data, conf):
        self.data = build_seqs(get_seq_from_df(data, conf), conf['session_len'])

    def __iter__(self):
        seqs = self.data[0]
        tar = self.data[1]
        for i in range(len(tar)):
            yield seqs[i], tar[i]

class SRGNNDataset(object):
    def __init__(self, data, conf, shuffle=False, graph=None):
        data = build_seqs(get_seq_from_df(data, conf), conf['session_len'])
        inputs = data[0]
        inputs, mask, len_max = self.data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def data_masks(self, all_usr_pois, item_tail):
        us_lens = [len(upois) for upois in all_usr_pois]
        len_max = max(us_lens)
        us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
        us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
        return us_pois, us_msks, len_max

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items, mask, targets

class GRU4RECDataset(object):
    def __init__(self, data, conf, batch_size, time_sort=False):
        data[conf['item_key']] -= 1 # go back to codes from 0
        self.df = data
        self.batch_size = batch_size
        self.session_key = conf['session_key']
        self.item_key = conf['item_key']
        self.time_key = conf['time_key']
        self.time_sort = time_sort

        self.df.sort_values([self.session_key, self.time_key], inplace=True)
        self.click_offsets = self._get_click_offset()
        self.session_idx_arr = self._order_session_idx()


    def _get_click_offset(self):
        self.batch_lim = self.df[self.session_key].nunique()
        offsets = np.zeros(self.batch_lim + 1, dtype=np.int32)
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum()
        return offsets

    def _order_session_idx(self):
        if self.time_sort:
            sessions_start_time = self.df.groupby(self.session_key)[self.time_key].min().values
            session_idx_arr = np.argsort(sessions_start_time)
        else:
            session_idx_arr = np.arange(self.df[self.session_key].nunique())
        return session_idx_arr

    def __iter__(self):
        '''
        Returns the iterator for producing session-parallel training mini-batches.

        Yields
        -------
        input : torch.FloatTensor
            Item indices that will be encoded as one-hot vectors later. size (B,)
        target : torch.teFloatTensornsor
            a Variable that stores the target item indices, size (B,)
        masks : Numpy.array
            indicating the positions of the sessions to be terminated
        '''    
        click_offsets = self.click_offsets
        session_idx_arr = self.session_idx_arr
        iters = np.arange(self.batch_size)
        maxiter = iters.max()

        start = click_offsets[session_idx_arr[iters]]
        end = click_offsets[session_idx_arr[iters] + 1]
        mask = []  # indicator for the sessions to be terminated
        finished = False

        while not finished:
            minlen = (end - start).min()
            # Item indices(for embedding) for clicks where the first sessions start
            idx_target = self.df[self.item_key].values[start]

            for i in range(minlen - 1):
                # Build inputs & targets
                idx_input = idx_target
                idx_target = self.df[self.item_key].values[start + i + 1]
                input = torch.LongTensor(idx_input)
                target = torch.LongTensor(idx_target)
                yield input, target, mask

            # click indices where a particular session meets second-to-last element
            start = start + (minlen - 1)
            # see if how many sessions should terminate
            mask = np.arange(len(iters))[(end - start) <= 1]
            for idx in mask:
                maxiter += 1
                if maxiter >= len(click_offsets) - 1:
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = maxiter
                start[idx] = click_offsets[session_idx_arr[maxiter]]
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]

