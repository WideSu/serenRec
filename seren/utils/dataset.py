import numpy as np
from torch.utils.data import Dataset, DataLoader
from .functions import pad_zero_for_seq


class NARMDataset(Dataset):
    def __init__(self, data, logger):
        '''
        Session sequences dataset class

        Parameters
        ----------
        data : list
            [[seqs],[targets]]
        logger : logging.logger
            Logger used for recording process
        '''        
        self.data = data
        self.logger = logger
        logger.debug('Number of sessions: {}'.format(len(data[0])))
        
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

class SRGNNDataset(object):
    def __init__(self, data, shuffle=False, graph=None):
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


