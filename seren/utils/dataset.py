import numpy as np
from torch.utils.data import Dataset, DataLoader
from .functions import pad_zero_for_seq


class SeqDataset(Dataset):
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

class SGNNDataset(object):
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


