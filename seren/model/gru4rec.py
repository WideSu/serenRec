'''
@inproceedings{tan2016improved,
  title={Improved recurrent neural networks for session-based recommendations},
  author={Tan, Yong Kiam and Xu, Xinxing and Liu, Yong},
  booktitle={Proceedings of the 1st workshop on deep learning for recommender systems},
  pages={17--22},
  year={2016}
}
@article{hidasi2015session,
  title={Session-based recommendations with recurrent neural networks},
  author={Hidasi, Bal{\'a}zs and Karatzoglou, Alexandros and Baltrunas, Linas and Tikk, Domonkos},
  journal={arXiv preprint arXiv:1511.06939},
  year={2015}
}
'''
from turtle import forward
import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

class GRU4REC(nn.Module):
    def __init__(self, config):
        '''
        GRU4REC Recommender

        Parameters
        ----------
        dropout_rate : float
            the rate to keep element be zero in tensor
        embedding_dim : int
            dimension of item embedding
        num_layers : int
            number of layers for GRU
        hidden_dim : int 
            dimension of the hidden layer in GRU
        learning_rate : float
            learning rate
        weight_decay : float
            weight decaying rate for learning rate
        n_epoch : int
            epochs to train the model
        early_stop : bool
            use early stop mechanism or not, default is True
        learner : String
            the name of optimizaer to train the model
        device : String
            the mode of training, default is `cpu`
        max_len : int
            the maximum length for a session
        item_num : int
            the number of unique items in training set
        loss_type : String
            loss function name, default is `BPR`
        '''        
        super(GRU4REC, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.num_layers = config['num_layers']
        self.hidden_dim = config['hidden_dim']
        self.dropout_rate = config['dropout_rate']
        self.lr = config['learning_rate']
        self.wd = config['weight_decay'] 
        self.n_epoch = config['n_epoch'] 
        self.early_step = config['early_step']
        self.learner = config['learner']
        self.device = config['device']
        self.max_len = config['max_len']
        self.item_num = config['item_num']
        self.loss_type = config['loss_type']

        self.item_embedding = nn.Embedding(self.item_num + 1, self.embedding_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(self.dropout_rate)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        # use a dense layer to map the output of GRU back to item embedding dimension
        self.dense = nn.Linear(self.hidden_dim, self.embedding_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Embedding):
            nn.init.xavier_normal_(m.weight)
            with torch.no_grad():
                m.weight[0] = torch.zeros(self.embedding_dim)
        elif isinstance(m, nn.GRU):
            nn.init.xavier_normal_(m.weight_hh_l0)
            nn.init.xavier_normal_(m.weight_ih_l0)
        
    def forward(self, item_seq):
        item_seq_len = torch.count_nonzero(item_seq, dim=1)
        last_index = item_seq_len - 1

        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)

        output = gru_output.gather(
            dim=1, index=last_index.view(-1, 1, 1).expand(-1, -1, gru_output.shape[-1])).squeeze(1)

        return output

    def fit(self, train_loader):
        self.to(self.device)
        optimizer = self._select_optimizer(learning_rate=self.lr, weight_decay=self.wd)
        criterion = nn.CrossEntropyLoss() # TODO add BPR, since current we only have positive labels

        last_loss = 0.
        for epoch in range(1, self.n_epoch + 1):
            self.train()

            current_loss, sample_cnt = 0., 0
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')
            for item_seq, next_item, _ in pbar:
                self.zero_grad()
                if self.loss_type in ['CE']:
                    output = self.forward(item_seq)
                    logits = torch.matmul(output, self.item_embedding.weight.transpose(0, 1))
                    loss = criterion(logits, next_item)
                elif self.loss_type in ['BPR', 'TOP1']:
                    pass

                if torch.isnan(loss):
                    raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the recommender')

                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
                current_loss += loss.item()
                sample_cnt += 1
            
            current_loss /= sample_cnt

            self.eval()
            delta_loss = float(current_loss - last_loss)
            if (abs(delta_loss) < 1e-5) and self.early_stop:
                print('Satisfy early stop mechanism')
                break
            else:
                last_loss = current_loss

    def predict(self, input_ids, next_item):
        pass

    def rank(self, test_loader, topk=50):
        pass
