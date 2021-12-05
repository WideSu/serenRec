import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

class GRU4REC(nn.Module):
    def __init__(self, input_size, conf, logger):
        '''
        GRU4REC model class: https://arxiv.org/pdf/1511.06939.pdf
        GRU4REC+ model class: https://openreview.net/pdf?id=ryCM8zWRb

        Parameters
        ----------
        input_size : int
            the number of items
        item_embedding_dim : int
            the dimension of item embedding, by defualt -1 with not use
        hidden_size : int
            the hidden size of gru
        learning_rate : float
            learning rate
        n_layers : int
            hidden layer, by default 3
        dropout_hidden : float
            dropout rate of hidden layer
        dropout_input : float
            dropout rate of input layer
        batch_size : int
            batch size
        final_act : str
            Final Activation Function, by default 'tanh'
        optimizer : str
            optimizer type, by default 'Adagrad'
        momentum : float
            momentum value
        wd : float
            weight decay
        loss_type : str
            loss function type
        '''    
        super(GRU4REC, self).__init__()
        self.use_cuda = True if torch.cuda.is_available() else False
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.logger = logger
        self.input_size = input_size  # item number
        self.hidden_size = conf['hidden_size']
        self.output_size = input_size # output_size is the same as input_size
        self.num_layers = conf['n_layers']
        self.dropout_hidden = conf['dropout_hidden']
        self.dropout_input = conf['dropout_input']
        self.embedding_dim = conf['item_embedding_dim'] # -1 t0 not use embedding, otherwise use item embedding
        self.batch_size = conf['batch_size']
        self.onehot_buffer = self.init_emb()
        self.h2o = nn.Linear(self.hidden_size, self.output_size)

        if self.embedding_dim != -1:
            self.look_up = nn.Embedding(input_size, self.embedding_dim)
            self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        else:
            self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        
        self.create_final_activation(conf['final_act'])
        
        self.epochs = conf['epochs']
        self.sigma = conf['sigma']
        
        self.loss_func = LossFunction(conf['loss_type'], self.use_cuda)

        self.reset_parameters()
        self.set_optimizer(
            conf['optimizer'], conf['learning_rate'], conf['momentum'], conf['wd'], conf['eps'])

        self = self.to(self.device)

    def _reset_hidden(self, hidden, mask):
        '''
        Helper function that resets hidden state when some sessions terminate
        '''
        if len(mask) != 0:
            hidden[:, mask, :] = 0
        return hidden


    def fit(self, train_loader, valid_loader=None):
        self.logger.info('Start training...')
        for epoch in range(1, self.epochs + 1):
            self.train()
            total_loss = []
            hidden = self.init_hidden()

            for _, (input, target, mask) in enumerate(train_loader):
                input = input.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                hidden = self._reset_hidden(hidden, mask).detach()
                logit, hidden = self.forward(input, hidden)
                logit_sampled = logit[:, target.view(-1)]
                loss = self.loss_func(logit_sampled)
                total_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            s = ''
            if valid_loader:
                self.eval()
                val_loss = []
                with torch.no_grad():
                    hidden = self.init_hidden()
                    for _, (input, target, mask) in enumerate(valid_loader):
                        input = input.to(self.device)
                        target = target.to(self.device)
                        logit, hidden = self.forward(input, hidden)
                        logit_sampled = logit[:, target.view(-1)]
                        loss = self.loss_func(logit_sampled)
                        val_loss.append(loss.item())
                s = f'\tValidation Loss: {np.mean(val_loss)}'

            self.logger.info(f'training epoch: {epoch}\tTrain Loss: {np.mean(total_loss):.3f}' + s)
    
    
    def set_optimizer(self, optimizer_type='Adagrad', lr=.05, momentum=0, weight_decay=0, eps=1e-6):
        '''
        An optimizer function for handling various kinds of optimizers.
        You can specify the optimizer type and related parameters as you want.
        Usage is exactly the same as an instance of torch.optim

        Parameters
        ----------
        optimizer_type : str, optional
            type of the optimizer to use, by default 'Adagrad'
        lr : float, optional
            learning rate, by default .05
        momentum : int, optional
            momentum, if needed, by default 0
        weight_decay : int, optional
            weight decay, if needed. Equivalent to L2 regulariztion., by default 0
        eps : [type], optional
            eps parameter, if needed., by default 1e-6

        Raises
        ------
        NotImplementedError
            Invalid optimizer name
        '''        
        if optimizer_type == 'RMSProp':
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr, eps=eps, weight_decay=weight_decay, momentum=momentum)
        elif optimizer_type == 'Adagrad':
            self.optimizer = optim.Adagrad(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'Adadelta':
            self.optimizer = optim.Adadelta(self.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
        elif optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
        elif optimizer_type == 'SparseAdam':
            self.optimizer = optim.SparseAdam(self.parameters(), lr=lr, eps=eps)
        elif optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise NotImplementedError('Invalid optimizer name...')

    def reset_parameters(self):
        '''
        weight initialization if sigma was defined
        '''
        if self.sigma is not None:
            for p in self.parameters():
                if self.sigma != -1 and self.sigma != -2:
                    sigma = self.sigma
                    p.data.uniform_(-sigma, sigma)
                elif len(list(p.size())) > 1:
                    sigma = np.sqrt(6.0 / (p.size(0) + p.size(1)))
                    if self.sigma == -1:
                        p.data.uniform_(-sigma, sigma)
                    else:
                        p.data.uniform_(0, sigma)

    def create_final_activation(self, final_act):
        if final_act == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_act == 'relu':
            self.final_activation = nn.ReLU()
        elif final_act == 'softmax':
            self.final_activation = nn.Softmax()
        elif final_act == 'softmax_logit':
            self.final_activation = nn.LogSoftmax()
        elif final_act.startswith('elu-'):
            self.final_activation = nn.ELU(alpha=float(final_act.split('-')[1]))
        elif final_act.startswith('leaky-'):
            self.final_activation = nn.LeakyReLU(negative_slope=float(final_act.split('-')[1]))

    def forward(self, input, hidden):
        '''
        Args:
            input (B,): a batch of item indices from a session-parallel mini-batch.
            target (B,): torch.LongTensor of next item indices from a session-parallel mini-batch.
        Returns:
            logit (B,C): Variable that stores the logits for the next items in the session-parallel mini-batch
            hidden: GRU hidden state
        '''

        if self.embedding_dim == -1:
            embedded = self.onehot_encode(input)
            if self.training and self.dropout_input > 0: embedded = self.embedding_dropout(embedded)
            embedded = embedded.unsqueeze(0)
        else:
            embedded = input.unsqueeze(0)
            embedded = self.look_up(embedded)

        output, hidden = self.gru(embedded, hidden) #(num_layer, B, H)
        output = output.view(-1, output.size(-1))  #(B,H)
        logit = self.final_activation(self.h2o(output))

        return logit, hidden

    def init_emb(self):
        '''
        Initialize the one_hot embedding buffer, which will be used for producing 
        the one-hot embeddings efficiently
        '''        

        onehot_buffer = torch.FloatTensor(self.batch_size, self.output_size)
        onehot_buffer = onehot_buffer.to(self.device)
        return onehot_buffer

    def onehot_encode(self, input):
        '''
        [summary]

        Parameters
        ----------
        input : torch.LongTensor
            (B,): torch.LongTensor of item indices

        Returns
        -------
        torch.FloatTensor
            (B,C): torch.FloatTensor of one-hot vectors
        '''        
        self.onehot_buffer.zero_()
        index = input.view(-1, 1)
        one_hot = self.onehot_buffer.scatter_(1, index, 1)
        return one_hot

    def embedding_dropout(self, input):
        p_drop = torch.Tensor(input.size(0), 1).fill_(1 - self.dropout_input)
        mask = torch.bernoulli(p_drop).expand_as(input) / (1 - self.dropout_input)
        mask = mask.to(self.device)
        input = input * mask
        return input

    def init_hidden(self):
        '''
        Initialize the hidden state of the GRU
        '''
        try:
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        except:
            self.device = 'cpu'
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        return h0

class TOP1Loss(nn.Module):
    def __init__(self):
        super(TOP1Loss, self).__init__()
    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        """
        diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
        loss = torch.sigmoid(diff).mean() + torch.sigmoid(logit ** 2).mean()
        return loss

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        """
        # differences between the item scores
        diff = logit.diag().view(-1, 1).expand_as(logit) - logit
        # final loss
        loss = -torch.mean(F.logsigmoid(diff))
        return loss

class LossFunction(nn.Module):
    def __init__(self, loss_type='TOP1', use_cuda=False):
        """ An abstract loss function that can supports custom loss functions compatible with PyTorch."""
        super(LossFunction, self).__init__()
        self.loss_type = loss_type
        self.use_cuda = use_cuda
        if loss_type == 'TOP1':
            self._loss_fn = TOP1Loss()
        elif loss_type == 'BPR':
            self._loss_fn = BPRLoss()
        else:
            raise NotImplementedError

    def forward(self, logit):
        return self._loss_fn(logit)