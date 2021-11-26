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


def get_loader(dataset, args, shuffle=True):
    loader = DataLoader(
        dataset, 
        batch_size=args['batch_size'], 
        shuffle=shuffle, 
        collate_fn=pad_zero_for_seq
    )

    return loader
