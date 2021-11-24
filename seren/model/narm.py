import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

epsilon = 1e-4

class NARM(nn.Module):
    def __init__(self, n_items, embedding_item_dim, hidden_size, 
                 lr, l2, lr_dc_step, lr_dc=0.1, n_layers=1):
        '''
        NARM model class: https://dl.acm.org/doi/pdf/10.1145/3132847.3132926

        Parameters
        ----------
        n_items : int
            the number of items
        embedding_item_dim : int
            the dimension of item embedding
        hidden_size : int
            the hidden size of gru
        lr : float
            learning rate
        l2 : float
            L2-regularization term
        lr_dc_step : int
            Period of learning rate decay
        lr_dc : float
            Multiplicative factor of learning rate decay, by default 1 0.1
        n_layers : int, optional
            the number of gru layers, by default 1
        '''        
        super(NARM, self).__init__()
        # parameters
        self.n_items = n_items
        self.embedding_item_dim = embedding_item_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        # Embedding layer
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_item_dim, padding_idx = 0)
        # Dropout layer
        self.emb_dropout = nn.Dropout(0.25)
        self.ct_dropout = nn.Dropout(0.5)
        # GRU layer
        self.gru = nn.GRU(self.embedding_item_dim, self.hidden_size, self.n_layers)
        # Linear layer
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.b = nn.Linear(self.embedding_item_dim, 2 * self.hidden_size, bias=False)
        
        self.loss_function = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_dc_step, gamma=lr_dc)
        self.sf = nn.Softmax(dim=1) #nn.LogSoftmax(dim=1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)

    def forward(self, seq, lengths):
        batch_size = seq.size(1)
        hidden = self.init_hidden(batch_size)
        embs = self.emb_dropout(self.item_embedding(seq))
        embs = pack_padded_sequence(embs, lengths)
        gru_out, hidden = self.gru(embs, hidden)
        gru_out, lengths = pad_packed_sequence(gru_out)

        # fetch the last hidden state of last timestamp
        ht = hidden[-1]
        gru_out = gru_out.permute(1, 0, 2)

        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())
        q2 = self.a_2(ht)

        mask = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device = self.device), torch.tensor([0.], device = self.device))
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand

        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_size)).view(mask.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)
        
        item_embs = self.item_embedding(torch.arange(self.n_items).to(self.device))
        scores = torch.matmul(c_t, self.b(item_embs).permute(1, 0)) # batch_size * item_size
        item_scores = self.sf(scores)
        
        return item_scores

    

