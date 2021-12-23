import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import torch.optim as optim

def hard_sigmoid(x):
    point_two = 0.2
    point_five =0.5
    x = torch.multiply(x, point_two)
    x = torch.add(x, point_five)
    x = torch.clamp(x, 0., 1.)
    return x

def new_softmax(x):
    x0 = torch.sum(x, dim=-1)
    x1 = torch.reshape(x0, [x.size()[0], 1])
    x2 = torch.tile(x1, [1, x.size()[1]])
    x3 = x / x2
    return x3

def tensor_ILD(topk):
    item_genre = pd.read_csv('./properties.csv')
    genre_data = item_genre.T.copy()
    genre_data = genre_data.tail(19).T
    genre_data = np.array(genre_data, dtype=np.float32)
    genre_data = torch.tensor(genre_data)
    id = copy.copy(topk)
    K = topk.get_shape().as_list()[-1]
    genre_vector = torch.nn.Embedding(genre_data, id)
    ILD = 0.0
    for i in range(K):
        genre_vector_0 = torch.reshape(genre_vector[:, i, :],
                                    [genre_vector[:, i, :].size()[0], 1, genre_vector[:, i, :].size()[1]])
        ele_1 = torch.tile(genre_vector_0, [1, genre_vector.size()[1], 1]).float()
        ele_1_prod = torch.sqrt(torch.sum(torch.square(ele_1 - genre_vector), dim=2))
        ILD = ILD + ele_1_prod
    ILD = torch.sum(ILD, dim=-1) / K
    ILD = ILD / (K - 1)
    return ILD

class DSR_RNN(nn.Module):
    def __init__(self, item_num, args):
        self.item_num = item_num
        self.arg = args
        self.item_emb_table = self.get_item_embeddings(self.item_num + 1, self.arg['hidden_units'])
        self.enc_gru = nn.GRU(self.arg['hidden_units'], self.args['enc_units'])
        self.dec_gru = nn.GRU(self.arg['hidden_units'], self.args['enc_units']) # TODO output


    def get_item_embeddings(self, vocab_size, num_units, zero_pad=True):
        if zero_pad:
            embeddings = nn.Embedding(vocab_size, num_units, padding_idx=0)
        else:
            embeddings = nn.Embedding(vocab_size, num_units)
        nn.init.xavier_uniform_(embeddings.weight)
        return embeddings

    def encoder(self, xs):
        seq = self.item_emb_table(xs)
        seq_len = torch.not_equal(xs, 0).int().sum(dim=-1).int() # all seq len should be processed to same size

        enc_output, state = self.enc_gru(seq)

        xsss = torch.not_equal(xs, 0).float()
        xsss = torch.reshape(xsss, [xsss.size()[0], xsss.size()[1], 1])
        enc_output = enc_output * xsss  

        ln_state = self.ln(state)

        role_pre01, A1, V1,Q1 = self.self_attention(xsss, ln_state, enc_output, enc_output, self.args['enc_units'], self.args['self_attention_mode'])
        role_pre1 = torch.concat((role_pre01, ln_state), -1)

        role_pre02, A2, V2,Q2 = self.self_attention(xsss, ln_state, enc_output, enc_output, self.args['enc_units'], self.args['self_attention_mode'])
        role_pre2 = torch.concat((role_pre02, ln_state), -1)

        role_pre03, A3, V3,Q3 = self.self_attention(xsss, ln_state, enc_output, enc_output, self.args['enc_units'], self.args['self_attention_mode'])  
        role_pre3 = torch.concat((role_pre03, ln_state), -1)

        W_w = torch.nn.Parameter(torch.zeros((self.args['enc_units'], self.args['enc_units'])))
        torch.nn.init.xavier_normal_(W_w)
        
        role_pref0 = torch.concat((role_pre01, role_pre02, role_pre03), 1)  
        role_pref01 = torch.reshape(role_pref0, [-1, role_pref0.size()[2]])  
        role_pref02 = torch.matmul(role_pref01, W_w)
        role_pref03 = torch.reshape(role_pref02, [role_pref0.size()[0], role_pref0.size()[1], -1])
        role_weight0 = torch.nn.softmax(torch.matmul(ln_state, torch.transpose(role_pref03, [0, 2, 1])), -1) 

        W_a = torch.nn.Parameter(torch.zeros((self.args['hidden_units'], self.args['enc_units'])))
        torch.nn.init.xavier_normal_(W_a)
        
        emb2 = torch.matmul(self.item_emb_table, W_a)
        role_pref = torch.concat((role_pre1, role_pre2, role_pre3), 1)  
        role_pref1 = torch.reshape(role_pref, [-1, role_pref.size()[2]])  
        role_rele = torch.matmul(role_pref1, torch.transpose(emb2)) 

        role_rele1 = torch.nn.softmax(torch.reshape(role_rele, [role_pref.size()[0], role_pref.size()[1], -1]),-1)

        user_rele2 = torch.matmul(role_weight0, role_rele1)  
        user_rele3 = torch.reshape(user_rele2, [user_rele2.size()[0], -1])  
        s_rel = new_softmax(user_rele3)  

        sub_pre = torch.concat((V1, V2, V3), 1)
        pos_pre = torch.concat((A1, A2, A3), 1)

        return role_pref0, s_rel, sub_pre, pos_pre, role_weight0

    def decoder(self, role_pre, s_rel, y_out, y_out_greedy, role_weight, training=True):
        emb1 = self.item_emb_table
        emb1_stop = emb1
        role_pre_stop = role_pre
        role_weight_stop0 = role_weight


        # dec gru
        dec_seq = self.item_emb_table(y_out)
        seq_len = torch.not_equal(y_out, 0).int().sum(dim=-1).int()

        dec_output, state = self.enc_gru(dec_seq)
        dec_pre = self.ln(state)

        # diversity score
        role_pref01 = torch.reshape(role_pre_stop, [-1, role_pre.size()[2]]) 

        dW_A = torch.nn.Parameter(torch.zeros((self.args['enc_units'], self.args['enc_units'])))
        dW_B = torch.nn.Parameter(torch.zeros((self.args['enc_units'], self.args['enc_units'])))
        dW_V = torch.nn.Parameter(torch.zeros((1, self.args['enc_units'])))
        torch.nn.init.xavier_normal_(dW_A)
        torch.nn.init.xavier_normal_(dW_B)
        torch.nn.init.xavier_normal_(dW_V)

        l11 = torch.reshape(
            torch.matmul(dW_A, torch.transpose(role_pref01)), [-1, role_pre.size()[0], role_pre.size()[1]])  
        l22 = torch.reshape(torch.matmul(dW_B, torch.transpose(dec_pre)), [-1, role_pre.size()[0], 1]) 
        l22 = torch.tile(l22, [1, 1, role_pre.size()[1]])  
        l33 = torch.reshape(hard_sigmoid(l11 + l22), [l22.size()[0], -1])  
        q_jh = torch.reshape(torch.matmul(dW_V, l33), [role_pre.size()[0], 1, role_pre.size()[1]])  

        e_q_jh = torch.reshape(torch.exp(q_jh) * role_weight_stop0, [q_jh.size()[0], -1])
        role_weight0 = 1 - new_softmax(e_q_jh)
        role_weight_att = torch.reshape(role_weight0, [q_jh.size()[0], 1, -1])

        role_pre_stop1 = torch.reshape(role_pre_stop, [-1, role_pre.size()[2]])  
        role_pre_rele = torch.matmul(role_pre_stop1, torch.transpose(emb1_stop))  
        role_pre_rele1 = torch.nn.softmax(
                torch.reshape(role_pre_rele, [role_pre.size()[0], role_pre.size()[1], -1]), -1) 

        s_div = torch.matmul(role_weight_att, role_pre_rele1)  
        s_div = torch.reshape(s_div, [s_div.size()[0], -1])  
        s_div = new_softmax(s_div)  

        # total score
        s_rel_1 = s_rel
        s_div_1 = s_div

        s_rel_stop = s_rel
        score_0 = self.args['lamb1'] * s_rel_stop + (1 - self.args['lamb1']) * s_div  
        score_0 = new_softmax(score_0)
        score1 = score_0 - (torch.sum(one_hot(y_out, score_0.size()[-1]), dim=1) * score_0).float()
        
        y_hat0 = torch.argmax(score1, axis=-1).int()

        y_hat1 = torch.reshape(one_hot(y_hat0, self.arg.item_num + 1).float(), [score_0.size()[0], score_0.size()[1]]) 
        y_hat_score0 = torch.sum(score_0 * y_hat1, dim=-1)  
        y_hat_score = torch.reshape(y_hat_score0, [y_hat0.size()[0], 1]) 
        y_hat = torch.reshape(y_hat0, [-1, 1])

        # greedy selection
        if training:
            ss_rel = s_rel - (torch.sum(one_hot(y_out_greedy, s_rel.size()[-1]), 1).float() * s_rel)
            y_hat0_greedy = torch.argmax(ss_rel, dim=-1).int()  
            y_hat01_greedy = one_hot(y_hat0_greedy, self.item_num + 1).float()
            y_hat1_greedy = torch.reshape(y_hat01_greedy, [score_0.size()[0], score_0.size()[1]])
            y_hat_socre0_greedy = torch.sum(score_0 * y_hat1_greedy, dim=-1)  
            y_hat_score_greedy = torch.reshape(y_hat_socre0_greedy, [y_hat0_greedy.size()[0], 1])  
            y_hat_greedy = torch.reshape(y_hat0_greedy, [-1, 1])

        if training:
            return y_hat, y_hat_greedy, state, s_rel_1, s_div_1, y_hat_score, y_hat_score_greedy
        else:
            return y_hat, state, s_rel_1, s_div_1

    def forward(self, xs):
        role_pre, s_rel, sub_pre, pos_pre, role_weight0 = self.encoder(xs)
        y_hat = torch.reshape(torch.argmax(s_rel, dim=-1).int(), [-1, 1]) 

        y_out = y_hat
        y_out_greedy = y_hat

        y_hat1 = torch.reshape(
            one_hot(y_hat, self.arg.item_num + 1).float()), [s_rel.size()[0], s_rel.size()[1]])
        y_hat_score0 = torch.sum(s_rel * y_hat1, dim=-1)
        y_hat_score = torch.reshape(y_hat_score0, [tf.shape(y_hat)[0], 1])
        div_scores = y_hat_score
        div_scores_greedy = y_hat_score

        for _ in range(self.args['k'] - 1):
            y_hat, y_hat_greedy, dec_hidden, s_rel_1, s_div_1, y_hat_score, y_hat_score_greedy = self.decoder(role_pre,
                                                                                                              s_rel,
                                                                                                              y_out,
                                                                                                              y_out_greedy,
                                                                                                              role_weight0)  
            y_out = torch.concat((y_out, y_hat), 1)
            y_out_greedy = torch.concat((y_out_greedy, y_hat_greedy), 1)
            div_scores = torch.concat((div_scores, y_hat_score), 1)
            div_scores_greedy = torch.concat((div_scores_greedy, y_hat_score_greedy), 1)

        ys = y_out  
        ys_greedy = y_out_greedy
        div_score = div_scores  
        div_score_greedy = div_scores_greedy  

        return s_rel, ys, ys_greedy, div_score, div_score_greedy, pos_pre, role_weight0 , s_rel_1, s_div_1

    def fit(self, train_loader):
        self.train()

        optimizer = optim.Adam(lr=self.args['learning_rate'])

        for epochs in range(1, self.args['epochs'] + 1):
            for xs, y_ground in train_loader:
                xs = torch.tensor(xs).to(self.device)
                y_ground = torch.tensor(y_ground).to(self.device)
                optimizer.zero_grad()

                s_rel, ys, ys_greedy, div_score, div_score_greedy, pos_pre, role_weight0 , s_rel_1, s_div_1 = self.forward(xs)

                # ########################------------------loss relevance-0---------------------######################

                y_ground1 = torch.reshape(one_hot(y_ground, self.item_num + 1), [y_ground.size()[0], -1])  
                CE_loss_pos = -torch.sum((torch.log(s_rel + 1e-24) * y_ground1), -1)  
                loss_rel = CE_loss_pos

                # ########################------------------loss diversity 1---------------------######################
                greedy_ILD = tensor_ILD(ys_greedy)
                ys_ILD = tensor_ILD(ys)
                prob_ys = torch.sum(torch.log(div_score + 1e-24), -1)
                prob_ys_greedy = torch.sum(torch.log(div_score_greedy + 1e-24), dim=-1)
                prob = 1.0 / (1.0 + torch.exp(prob_ys_greedy - prob_ys))

                pos = torch.ones((ys_ILD.size())).float()
                neg = torch.zeros((ys_ILD.size())).float()

                ispositive = torch.where(torch.greater((ys_ILD - greedy_ILD), 0), pos, neg)
                loss_div = -(ys_ILD - greedy_ILD) * (
                        ispositive * torch.log(prob + 1e-24) + (1 - ispositive) * torch.log(prob + 1e-24))

                ########################------------------disaggrement loss positon---------------------######################
                pos_pre_0 = torch.reshape(pos_pre[:, 0, :], [pos_pre[:, 0, :].size()[0], 1, pos_pre[:, 0, :].size()[1]])
                pos_pre_1 = torch.reshape(pos_pre[:, 1, :], [pos_pre[:, 1, :].size()[0], 1, pos_pre[:, 1, :].size()[1]])
                pos_pre_2 = torch.reshape(pos_pre[:, 2, :], [pos_pre[:, 2, :].size()[0], 1, pos_pre[:, 2, :].size()[1]])
                ele_1 = torch.tile(pos_pre_0, [1, pos_pre.size()[1], 1]).float()
                ele_2 = torch.tile(pos_pre_1, [1, pos_pre.size()[1], 1]).float()
                ele_3 = torch.tile(pos_pre_2, [1, pos_pre.size()[1], 1]).float()

                pos_pre = pos_pre.float()

                ele_1_prod_norm = torch.sqrt(torch.sum(torch.square(ele_1), dim=2))
                ele_2_prod_norm = torch.sqrt(torch.sum(torch.square(ele_2), dim=2))
                ele_3_prod_norm = torch.sqrt(torch.sum(torch.square(ele_3), dim=2))
                pos_pre_norm = torch.sqrt(torch.sum(torch.square(pos_pre), dim=2))

                ele_1_x = torch.sum(torch.multiply(ele_1, pos_pre), dim=2)
                ele_2_x = torch.sum(torch.multiply(ele_2, pos_pre), dim=2)
                ele_3_x = torch.sum(torch.multiply(ele_3, pos_pre), dim=2)

                ele_1_prod = ele_1_x / (ele_1_prod_norm * pos_pre_norm)
                ele_2_prod = ele_2_x / (ele_2_prod_norm * pos_pre_norm)
                ele_3_prod = ele_3_x / (ele_3_prod_norm * pos_pre_norm)

                loss_dist_pos = torch.sum((ele_1_prod + ele_2_prod + ele_3_prod), -1) / 9.0  

                ########################------------------ME loss---------------------######################
                role_weig = torch.reshape(role_weight0, [role_weight0.size()[0], -1])  
                loss_ME = torch.sum((torch.log(role_weig + 1e-24) * role_weig), -1)/2

                ########################------------------total loss---------------------######################
                loss = self.args['lamb2'] * loss_rel + loss_div + loss_ME + self.args['pos'] * loss_dist_pos
                loss = loss.mean()

                loss.backward()
                optimizer.step()

        return role_weig, pos_pre, s_rel_1, s_div_1, loss, loss_rel, loss_div, loss_dist_pos, loss_ME


    def self_attention(self, xsss, Q, K, V, size_head, mode):
        padding_num = -2 ** 32 + 1
        WQ = nn.Linear(Q.size()[-1], size_head, bias=False)
        WK = nn.Linear(K.size()[-1], size_head, bias=False)
        WV = nn.Linear(V.size()[-1], size_head, bias=False)
        if mode == 0:
            Q = WQ(Q)
            Q = torch.reshape(Q, (-1, Q.size()[1], size_head))

            K = WK(K)
            K = torch.reshape(K, (-1, K.size()[1], size_head))

            V = WV(V)
            V = torch.reshape(V, (-1, V.size()[1], size_head))

            A0 = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(float(size_head))
            A1 = torch.reshape(A0, [A0.size()[0], -1, 1])
            paddings = torch.ones_like(A1) * padding_num
            outputs = torch.where(torch.eq(xsss, 0), paddings, A1)
            A2 = torch.nn.softmax(outputs, 1)

            O = torch.sum(A2 * V, dim=1)
            O = torch.reshape(O, [Q.size[0], 1, size_head])
            A = torch.reshape(A2, [Q.size[0], 1, K.size()[1]])

            return O, A, V, Q
        elif mode == 1:
            Q = WQ(Q)
            Q = torch.reshape(Q, (-1, Q.size()[1], size_head))

            K = WK(K)
            K = torch.reshape(K, (-1, K.size()[1], size_head))

            V = WV(V)
            V = torch.reshape(V, (-1, V.size()[1], size_head))

            
            W = torch.nn.Parameter(torch.zeros((Q.size()[-1], K.size()[-1])))
            torch.nn.init.xavier_normal_(W)

            QW = torch.matmul(torch.reshape(Q, (-1, Q.size()[-1])), W)
            QW = torch.reshape(QW, (-1, Q.size()[1], Q.size()[-1]))
            QWK = torch.matmul(QW, torch.transpose(K, [0, 2, 1])) / math.sqrt(float(size_head))
            A = torch.reshape(QWK, [K.size()[0], K.size()[1], 1])
            paddings = torch.ones_like(A) * padding_num
            outputs = torch.where(torch.eq(xsss, 0), paddings, A)
            A2 = torch.nn.softmax(outputs, 1)

            O = torch.sum(A2 * V, dim=1)
            O = torch.reshape(O, [Q.size()[0], 1, size_head])
            A = torch.reshape(A2, [K.size()[0], 1, K.size()[1]])
            return O, A, V, Q

        elif mode == 2:
            Q0 = torch.reshape(Q, [Q.size()[0], size_head])

            W_A = torch.nn.Parameter(torch.zeros((size_head, size_head)))
            torch.nn.init.xavier_normal_(W_A)

            W_B = torch.nn.Parameter(torch.zeros((size_head, size_head)))
            torch.nn.init.xavier_normal_(W_B)

            W_V = torch.nn.Parameter(torch.zeros((1, size_head)))
            torch.nn.init.xavier_normal_(W_V)

            K1 = torch.reshape(K, [-1, size_head]) 
            l11 = torch.reshape(torch.matmul(W_A, torch.transpose(K1)),[-1, K.size()[0], K.size()[1]])  
            l22 = torch.reshape(torch.matmul(W_B, torch.transpose(Q0)), [-1, K.size()[0], 1])  

            l22 = torch.tile(l22, [1, 1, K.size()[1]]) 
            # tf.keras.backend.hard_sigmoid(l11 + l22)
            l33 = torch.reshape(hard_sigmoid(l11 + l22), [l22.size()[0], -1])  

            q_jh = torch.reshape(torch.matmul(W_V, l33), [K.size()[0], K.size()[1], 1])  
            paddings = torch.ones_like(q_jh) * padding_num
            outputs = torch.where(torch.eq(xsss, 0), paddings, q_jh)
            att = torch.nn.softmax(outputs, 1)  
            A2 = att  

            O = torch.sum(A2 * V, dim=1)  
            O = torch.reshape(O, [Q.size()[0], 1, size_head])
            A = torch.reshape(A2, [K.size()[0], 1, K.size()[1]])
            return O, A, V, Q


    def ln(self, inputs, epsilon=1e-8, scope="ln"):
        '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
        inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
        epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        scope: Optional scope for `variable_scope`.

        Returns:
        A tensor with the same shape and data dtype as `inputs`.
        '''
        params_shape = inputs.size()[-1:]
        
        mean, std = torch.std_mean(inputs, dim=-1, keepdim=True)
        variance = std ** 2
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))

        beta = torch.zeros(params_shape)
        gamma = torch.ones(params_shape)
        outputs = gamma * normalized + beta

        return outputs