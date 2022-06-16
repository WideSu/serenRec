'''
@article{ludewig2018evaluation,
  title={Evaluation of session-based recommendation algorithms},
  author={Ludewig, Malte and Jannach, Dietmar},
  journal={User Modeling and User-Adapted Interaction},
  volume={28},
  number={4},
  pages={331--390},
  year={2018},
  publisher={Springer}
}
@article{rendle2012bpr,
  title={BPR: Bayesian personalized ranking from implicit feedback},
  author={Rendle, Steffen and Freudenthaler, Christoph and Gantner, Zeno and Schmidt-Thieme, Lars},
  journal={arXiv preprint arXiv:1205.2618},
  year={2012}
}
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class SBPR(nn.Module):
    def __init__(self, config):
        '''
        Session-BPR Recommender

        Parameters
        ----------
        item_num : int
            the number of unique items in training set
        embedding_dim : int
            embedding dimension for items, default is 100
        lambda_item : float
            l2-regularization term for item embedding, default is 0.001
        learning_rate : float
            learning tate, default is 0.01
        weight_decay : float
            weight decaying rate for learning rate, default is 1.0
        n_epoch : int
            epochs for training, default is 20
        early_stop : bool
            activate early stop mechanism or not, default is True
        max_len : int 
            maximum length for one session
        device : String
            running type for code, default is 'cpu'
        learner : String
            name of optimizer used for training, default is 'sgd'
        '''     
        super(SBPR, self).__init__() 
        self.item_num = config['item_num']
        self.embedding_dim = config['embedding_dim']
        self.lambda_item = config['lambda_item']
        self.lr = config['learning_rate']
        self.wd = config['weight_decay']
        self.n_epoch = config['n_epoch']
        self.early_stop = config['early_stop']
        self.max_len = config['max_len']
        self.device = config['device']
        self.learner = config['learner']

        self.item_embed = nn.Embedding(self.item_num+1, self.embedding_dim, padding_idx=0)
        
        self.apply(self._init_weight)
    
    def _init_weight(self, m):        
        if type(m) == nn.Embedding:
            nn.init.normal_(m.weight.data, 0, 0.05)
            with torch.no_grad():
                m.weight[0] = torch.zeros(self.embedding_dim)

    def _select_optimizer(self, **kwargs):
        params = kwargs.pop('params', self.parameters())
        learner = kwargs.pop('learner', self.learner)
        learning_rate = kwargs.pop('learning_rate', self.lr)
        weight_decay = kwargs.pop('weight_decay', self.wd)

        if learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            self.logger.warning('Invalid optimizer name, set default SGD optimizer instead')
            optimizer = optim.SGD(params, lr=learning_rate)
        return optimizer

    def forward(self, item_seq, next_item):
        next_item_embed = self.item_embed(next_item)
        session_items_embed =  self.item_embed(item_seq)
        # torch.count_nonzero(item_seq, dim = 1) # 1-D tensor for each batch
        session_embed = torch.div(
            torch.sum(session_items_embed,dim=1), 
            torch.count_nonzero(item_seq, dim=1).reshape(-1,1)) # shape: batch * max_len
        next_item_score = (session_embed * next_item_embed).sum(dim=-1)
        
        return next_item_score
    
    def fit(self, train_loader):
        self.to(self.device)
        optimizer = self._select_optimizer(learning_rate=self.lr, weight_decay=self.wd)
        
        last_loss = 0.
        for epoch in range(1, self.n_epoch + 1):
            self.train()
            current_loss, sample_cnt = 0, 0
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')
            for item_seq, pos_next_item, neg_next_item in pbar:

                self.zero_grad()
                r_si = self.forward(item_seq, pos_next_item)
                r_sj = self.forward(item_seq, neg_next_item)
                loss = -(r_si - r_sj).sigmoid().log().mean() + self.lambda_item * self.item_embed.weight.norm()
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
        if len(input_ids) > self.max_len or len(input_ids) == 0:
            raise ValueError('Invalid sequence length to predict, current supported maximum length is {self.max_len}...')

        self.eval()
        item_seq = torch.tensor(input_ids).to(self.device)
        item_seq = F.pad(item_seq, (0,self.max_len - len(input_ids))).unsqueeze(0)

        next_item = torch.tensor(next_item).to(self.device)
        score = self.forward(item_seq, next_item)
        return score.detach().cpu().item()

    def rank(self, test_loader, topk=50):
        self.eval()
        res_ids,res_scs = torch.tensor([]).to(self.device),torch.tensor([]).to(self.device)
        pbar = tqdm(test_loader)
        with torch.no_grad():
            for btch in pbar:
                item_seq = btch[0]
                item_seq = item_seq.to(self.device)
                session_items_embed =  self.item_embed(item_seq)
                session_embed = torch.div(
                    torch.sum(session_items_embed,dim=1), # batch_num * embed_dim
                    torch.count_nonzero(item_seq, dim=1).reshape(-1,1)) # shape: batch * embed_dim
                all_item_embs = self.item_embed.weight
                scores = torch.matmul(session_embed, all_item_embs.transpose(0, 1))
                scs, ids = torch.sort(scores[:, 1:], descending=True)
                ids += 1

                if topk is not None and topk <= self.item_num:
                    ids, scs = ids[:, :topk], scs[:, :topk]

                res_ids = torch.cat((res_ids, ids), 0)
                res_scs = torch.cat((res_scs, scs), 0)

        return res_ids.detach().cpu(), res_scs.detach().cpu()
