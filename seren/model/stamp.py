'''
@inproceedings{liu2018stamp,
  title={STAMP: short-term attention/memory priority model for session-based recommendation},
  author={Liu, Qiao and Zeng, Yifu and Mokhosi, Refuoe and Zhang, Haibin},
  booktitle={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1831--1839},
  year={2018}
}
'''
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

class STAMP(nn.Module):
    def __init__(self, config):
        '''
        STAMP Recommender

        Parameters
        ----------
        embedding_dim : int
            embedding dimension for items, default is 100
        mlp_a_dim : int
            dimension for the hidden layer of MLP A, default is the same as `embedding_dim`
        mlp_b_dim : int
            dimension for the hidden layer of MLP B, default is the same as `embedding_dim`
        learning_rate : float
            learning rate, default is 0.005
        weight_decay : float
            weight decaying rate for learning rate, default is 1.0
        n_epoch : int
            epochs for training, default is 30
        early_stop : bool
            activate early stop mechanism or not, default is True
        learner : String
            name of optimizaer used for training, default is 'sgd'
        device : String
            running type for code, default is 'cpu'
        max_len : int
            maximum length for one session
        use_attention : bool
            use STAMP or STMP, default is True (for STAMP)
        item_num : int
            the number of unique items in training set
        '''              
        super(STAMP, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.lr = config['learning_rate']
        self.wd = config['weight_decay']  
        self.n_epoch = config['n_epoch'] 
        self.early_step = config['early_step']
        self.learner = config['learner']
        self.device = config['device']
        self.max_len = config['max_len']
        self.item_num = config['item_num']

        self.mlp_a_dim = self.embedding_dim if config['mlp_a_dim'] is None else config['mlp_a_dim']
        self.mlp_b_dim = self.embedding_dim if config['mlp_b_dim'] is None else config['mlp_b_dim']

        self.item_embedding = nn.Embedding(self.item_num + 1, self.embedding_dim, padding_idx=0)
        self.mlp_a = nn.Linear(self.embedding_dim, self.mlp_a_dim)
        self.mlp_b = nn.Linear(self.embedding_dim, self.mlp_b_dim)

        # attention related
        self.W_0 = nn.Linear(self.embedding_dim, 1, bias=False)
        self.W_1 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_2 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_3 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.b_a = nn.Parameter(torch.zeros(self.embedding_dim), requires_grad=True)

        # activate function
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.apply(self._init_weight())

        self.use_attention = config['use_attention'] # True for STAMP, o.w. STMP

    def _init_weight(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight.data, 0, 0.05)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif type(m) == nn.Embedding:
            nn.init.normal_(m.weight.data, 0, 0.002)
            with torch.no_grad():
                m.weight[0] = torch.zeros(self.embedding_dim)

    def forward(self, item_seq):
        item_seq_len = torch.count_nonzero(item_seq, dim=1)
        last_index = item_seq_len - 1
        # batch_size * seq_len * embedding_dim
        x_i = self.item_embedding(item_seq) 
        # batch_size * 1 * embedding_dim, x_t = m_t
        x_t = x_i.gather(
            dim=1, index=last_index.view(-1, 1, 1).expand(-1, -1 ,item_seq.shape[-1])).squeeze(1)
        m_s = torch.div(torch.sum(x_i, dim=1), item_seq_len.unsqueeze(1).float())
        
        if self.use_attention:
            # attention score
            alpha = self._calc_att_score(x_i, x_t, m_s)
            m_a = torch.matmul(alpha.unsqueeze(1), x_i).squeeze(1)
            m_s = m_a + m_s

        h_s = self.tanh(self.mlp_a(m_s))
        h_t = self.tanh(self.mlp_b(x_t))
        output = h_s * h_t

        return output

    def _calc_att_score(self, x_i, x_t, m_s):
        timesteps = x_i.size(1)
        x_t_reshape = x_t.repeat(1, timesteps).view(-1, timesteps, self.embedding_size)
        m_s_reshape = m_s.repeat(1, timesteps).view(-1, timesteps, self.embedding_size)

        alpha = self.W_0(self.sigmoid(
            self.W_1(x_i) + self.W_2(x_t_reshape) + self.W_3(m_s_reshape) + self.b_a))
        alpha = alpha.squeeze(2)
        return alpha

    def _select_optimizer(self, **kwargs):
        params = kwargs.pop('params', self.parameters())
        learner = kwargs.pop('learner', self.learner)
        learning_rate = kwargs.pop('learning_rate', self.lr)
        weight_decay = kwargs.pop('weight_decay', self.wd)

        if self.config['reg_weight'] and weight_decay and weight_decay * self.config['reg_weight'] > 0:
            self.logger.warning(
                'The parameters [weight_decay] and [reg_weight] are specified simultaneously, '
                'which may lead to double regularization.'
            )

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

    def fit(self, train_loader):
        self.to(self.device)
        # calculate loss
        optimizer = self._select_optimizer(learning_rate=self.lr, weight_decay=self.wd)
        criterion = nn.CrossEntropyLoss()

        last_loss = 0.
        for epoch in range(1, self.n_epoch + 1):
            self.train()

            current_loss, sample_cnt = 0., 0
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')
            for item_seq, next_item in pbar:
                self.zero_grad()
                output = self.forward(item_seq)
                logits = self.sigmoid(torch.matmul(output, self.item_embedding.weight.transpose(0, 1)))
                pred_y = self.softmax(logits)
                loss = criterion(pred_y, next_item)

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
        '''
        method to predict the score of a target item given certain session items basket

        Parameters
        ----------
        input_ids : List
            a list of items in certain session
        next_item : int
            the index of the target next item

        Returns
        -------
        scores : float
            predicted scores of corresponding target items
        '''      
        if len(input_ids) > self.max_len or len(input_ids) == 0:
            raise ValueError(f'Invalid sequence length to predict, current supported maximum length is {self.max_len}...')

        self.eval()
        item_seq = torch.tensor(input_ids).to(self.device)
        item_seq = F.pad(item_seq, (0, self.max_len - len(input_ids))).unsqueeze(0)
        next_item = torch.tensor(next_item).to(self.device)

        seq_output = self.forward(item_seq)
        next_item_emb = self.item_embedding(next_item)
        score = torch.mul(seq_output, next_item_emb).sum(dim=1) 

        return score.detach().cpu().item()

    def rank(self, test_loader,topk=50):
        """_summary_

        Args:
            test_loader (_type_): _description_
            topk (int, optional): _description_. Defaults to 50.

        Returns:
            _type_: _description_
        """        
        self.eval()

        res_ids, res_scs = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        pbar = tqdm(test_loader)
        with torch.no_grad():
            for item_seq, _ in pbar:
                item_seq = item_seq.to(self.device)
                output = self.forward(item_seq)
                logits = self.sigmoid(torch.matmul(output, self.item_embedding.weight.transpose(0, 1)))
                scores = self.softmax(logits)
                scs, ids = torch.sort(scores[:, 1:], descending=True)
                ids += 1

                if topk is not None and topk <= self.item_num:
                    ids, scs = ids[:, :topk], scs[:, :topk]

                res_ids = torch.cat((res_ids, ids), 0)
                res_scs = torch.cat((res_scs, scs), 0)

        return res_ids.detach().cpu(), res_scs.detach().cpu()

