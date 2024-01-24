import numpy as np
import torch
from torch.nn import functional as F
from torch.autograd import Function
from typing import Any, Optional, Tuple
from torch import nn


class MF(nn.Module):
    def __init__(self, num_users, num_items, MF_latent_dim):
        super(MF, self).__init__()

        self.MF_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=MF_latent_dim)
        self.MF_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=MF_latent_dim)
        self.sigmoid = nn.Sigmoid()

    def get_embedding(self, user_input):
        return self.MF_Embedding_User(user_input)

    def get_rating(self, user_embedding, item_input):
        item_embedding = self.MF_Embedding_Item(item_input)
        predict_vector = torch.mul(user_embedding, item_embedding)
        prediction = torch.sum(predict_vector, dim=1)
        prediction = self.sigmoid(prediction)
        return prediction

    def forward(self, user_input, item_input):
        MF_User_Vector = self.MF_Embedding_User(user_input)
        MF_Item_Vector = self.MF_Embedding_Item(item_input)
        predict_vector = torch.mul(MF_User_Vector, MF_Item_Vector)
        prediction = torch.sum(predict_vector, dim=1)
        prediction = self.sigmoid(prediction)

        return prediction

class GMF(nn.Module):
    def __init__(self, num_users, num_items, MF_latent_dim):
        super(GMF, self).__init__()

        self.MF_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=MF_latent_dim)
        self.MF_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=MF_latent_dim)
        self.fc = nn.Linear(in_features=MF_latent_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def get_embedding(self, user_input):
        return self.MF_Embedding_User(user_input)

    def get_rating(self, user_embedding, item_input):
        item_embedding = self.MF_Embedding_Item(item_input)
        predict_vector = torch.mul(user_embedding, item_embedding)
        prediction = self.fc(predict_vector)
        prediction = self.sigmoid(prediction)
        return prediction

    def forward(self, user_input, item_input):
        MF_User_Vector = self.MF_Embedding_User(user_input)
        MF_Item_Vector = self.MF_Embedding_Item(item_input)

        predict_vector = torch.mul(MF_User_Vector, MF_Item_Vector)
        prediction = self.fc(predict_vector)
        prediction = self.sigmoid(prediction)

        return prediction

class MLP_rec(nn.Module):
    def __init__(self, num_users, num_items, MF_latent_dim):
        super(MLP_rec, self).__init__()

        self.MLP_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=MF_latent_dim)
        self.MLP_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=MF_latent_dim)
        self.fc1 = nn.Linear(int(MF_latent_dim*2), MF_latent_dim)
        self.fc2 = nn.Linear(MF_latent_dim, int(MF_latent_dim/2))
        self.fc3 = nn.Linear(int(MF_latent_dim/2), int(MF_latent_dim/4))

        self.output = nn.Linear(int(MF_latent_dim/4), 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self, user_input, item_input):


        MLP_User_Vector = self.MLP_Embedding_User(user_input)
        MLP_Item_Vector = self.MLP_Embedding_Item(item_input)

        mlp_vector = torch.cat((MLP_User_Vector, MLP_Item_Vector), 1)
        mlp_vector = self.fc1(mlp_vector)
        mlp_vector = self.relu(mlp_vector)
        mlp_vector = self.fc2(mlp_vector)
        mlp_vector = self.relu(mlp_vector)
        mlp_vector = self.fc3(mlp_vector)
        mlp_vector = self.relu(mlp_vector)
        prediction = self.output(mlp_vector)
        prediction = self.sigmoid(prediction)
        prediction = torch.flatten(prediction)
        return prediction


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py



class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class GRL_Layer(nn.Module):
    def __init__(self):
        super(GRL_Layer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class Extractor(nn.Module):
    def __init__(self, args):
        super(Extractor, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(args.hidden_units, args.hidden_units),
            nn.ReLU(),
            nn.Linear(args.hidden_units, args.hidden_units)
        )
        self.GRL = GRL_Layer()

    def forward(self, x):
        return self.MLP(x)

    def grl_forward(self, x):
        x = self.MLP(x)
        x = self.GRL(x)
        return x


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(args.hidden_units, int(args.hidden_units/2)),
            nn.ReLU(),
            nn.Linear(int(args.hidden_units/2), int(args.hidden_units/4)),
            nn.ReLU(),
            nn.Linear(int(args.hidden_units/4), int(args.hidden_units/8)),
            nn.ReLU(),
            nn.Linear(int(args.hidden_units/8), 1),
        )
    def forward(self, x):
        return self.MLP(x)


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        self.sigmoid = torch.nn.Sigmoid()


    def log2feats(self, log_seqs):

        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training

        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        pos_pred = self.sigmoid(pos_logits)
        neg_pred = self.sigmoid(neg_logits)

        return pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)

    def get_user_feature(self, log_seqs):
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet
        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste
        return final_feat


class TimeAwareMultiHeadAttention(torch.nn.Module):
    # required homebrewed mha layer for Ti/SASRec experiments
    def __init__(self, hidden_size, head_num, dropout_rate, dev):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    def forward(self, queries, keys, time_mask, attn_mask, time_matrix_K, time_matrix_V):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        # head dim * batch dim for parallelization (h*N, T, C/h)
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)
        # abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        # abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        # batched channel wise matmul to gen attention weights
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        # attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)

        # seq length adaptive scaling
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        # key masking, -2^32 lead to leaking, inf lead to nan
        # 0 * inf = nan, then reduce_sum([nan,...]) = nan

        # fixed a bug pointed out in https://github.com/pmixer/TiSASRec.pytorch/issues/2
        # time_mask = time_mask.unsqueeze(-1).expand(attn_weights.shape[0], -1, attn_weights.shape[-1])
        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) *  (-2**32+1) # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(time_mask, paddings, attn_weights) # True:pick padding
        attn_weights = torch.where(attn_mask, paddings, attn_weights) # enforcing causality

        attn_weights = self.softmax(attn_weights) # code as below invalids pytorch backward rules
        # attn_weights = torch.where(time_mask, paddings, attn_weights) # weird query mask in tf impl
        # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4
        # attn_weights[attn_weights != attn_weights] = 0 # rm nan for -inf into softmax case
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        # outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2) # div batch_size

        return outputs





class TPUF(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(TPUF, self).__init__()
        self.args = args
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.GRL = GRL_Layer()


        # self.combine_weight = nn.Parameter(torch.normal(mean=0, std=0.01, size=(2, user_num+1)))
        self.combine_weight = args.combine_weight
        self.softmax = nn.Softmax(dim=0)


        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList() 


        for _ in range(args.num_blocks+1):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


        self.extractor = Extractor(args)
        self.fc = nn.Linear(args.hidden_units, args.hidden_units)
        self.classifier = Classifier(args)

        self.sigmoid = torch.nn.Sigmoid()

        # Ti-SASRec
        # self.abs_pos_K_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        # self.abs_pos_V_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.time_matrix_K_emb = torch.nn.Embedding(args.time_span+1, args.hidden_units)
        self.time_matrix_V_emb = torch.nn.Embedding(args.time_span+1, args.hidden_units)

        # self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        # self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.ti_attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.ti_attention_layers = torch.nn.ModuleList()
        self.ti_forward_layernorms = torch.nn.ModuleList()
        self.ti_forward_layers = torch.nn.ModuleList()

        self.ti_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)  

        for _ in range(args.num_blocks+1):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.ti_attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = TimeAwareMultiHeadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate,
                                                            args.device)
            self.ti_attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.ti_forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.ti_forward_layers.append(new_fwd_layer)

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)  


    # def log2feats(self, log_seqs):
    #     seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
    #     seqs *= self.item_emb.embedding_dim ** 0.5
    #     positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
    #     seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
    #     t_embs = seqs
    #     seqs = self.emb_dropout(seqs)

    #     timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
    #     seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

    #     tl = seqs.shape[1] # time dim len for enforce causality
    #     attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

    #     for i in range(self.args.num_blocks):
    #         seqs = torch.transpose(seqs, 0, 1)
    #         Q = self.attention_layernorms[i](seqs)
    #         mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
    #                                         attn_mask=attention_mask)
    #                                         # key_padding_mask=timeline_mask
    #                                         # need_weights=False) this arg do not work?
    #         seqs = Q + mha_outputs
    #         seqs = torch.transpose(seqs, 0, 1)

    #         seqs = self.forward_layernorms[i](seqs)
    #         seqs = self.forward_layers[i](seqs)
    #         seqs *=  ~timeline_mask.unsqueeze(-1)

    #     # log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
    #     log_feats = seqs

    #     return log_feats, t_embs

    def s_embs2feats(self, s_embs):
        seqs = self.emb_dropout(s_embs)

        # timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        # seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(self.args.num_blocks, self.args.num_blocks+1):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            # seqs *= ~timeline_mask.unsqueeze(-1)

        feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)
        return feats
    
    def log2feats(self, log_seqs, time_matrices):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        positions = torch.LongTensor(positions).to(self.dev)
        seqs += self.pos_emb(positions)
        t_embs = seqs
        seqs = self.emb_dropout(seqs)

        time_matrices = torch.LongTensor(time_matrices).to(self.dev)
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        # mask 0th items(placeholder for dry-run) in log_seqs
        # would be easier if 0th item could be an exception for training
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.ti_attention_layers)):
            # Self-attention, Q=layernorm(seqs), K=V=seqs
            # seqs = torch.transpose(seqs, 0, 1) # (N, T, C) -> (T, N, C)
            Q = self.ti_attention_layernorms[i](seqs) # PyTorch mha requires time first fmt
            mha_outputs = self.ti_attention_layers[i](Q, seqs,
                                            timeline_mask, attention_mask,
                                            time_matrix_K, time_matrix_V)
            seqs = Q + mha_outputs
            # seqs = torch.transpose(seqs, 0, 1) # (T, N, C) -> (N, T, C)

            # Point-wise Feed-forward, actually 2 Conv1D for channel wise fusion
            seqs = self.ti_forward_layernorms[i](seqs)
            seqs = self.ti_forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats, t_embs    

    def forward(self, user_ids, log_seqs, time_matrices, pos_seqs, neg_seqs, s_embs): # for training
        positions = torch.LongTensor(np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])).to(self.dev)
        time_mat = self.pos_emb(positions)

        s_embs = s_embs.unsqueeze(1).repeat(1, log_seqs.shape[1], 1)
        ds_embs = self.extractor(s_embs)
        ds_embs += time_mat
        ds_embs = self.fc(ds_embs)

        ds_embs = self.s_embs2feats(ds_embs)


        log_feats, t_embs = self.log2feats(log_seqs, time_matrices) # user_ids hasn't been used yet
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))


        # combine_weight = self.softmax(self.combine_weight)
        # MF_User_Vector_Shared = torch.mul(combine_weight[0][user_ids].unsqueeze(1).repeat(1,log_seqs.shape[1]).unsqueeze(2), ds_embs)
        # MF_User_Vector_Specific = torch.mul(combine_weight[1][user_ids].unsqueeze(1).repeat(1,log_seqs.shape[1]).unsqueeze(2), log_feats)
        # MF_User_Vector = MF_User_Vector_Shared + MF_User_Vector_Specific

        MF_User_Vector_Shared = ds_embs * self.combine_weight
        MF_User_Vector_Specific = log_feats * (1 - self.combine_weight)
        MF_User_Vector = MF_User_Vector_Shared + MF_User_Vector_Specific

        # User_Vector = self.tisas2feats(MF_User_Matrix, log_seqs, time_matrices)

        pos_logits = (MF_User_Vector * pos_embs).sum(dim=-1)
        neg_logits = (MF_User_Vector * neg_embs).sum(dim=-1)

        pos_pred = self.sigmoid(pos_logits)
        neg_pred = self.sigmoid(neg_logits)


        pos_embs = self.classifier(self.GRL(self.fc(self.extractor(s_embs) + time_mat)))
        neg_embs = self.classifier(self.GRL(t_embs))


        pos_embs = self.sigmoid(pos_embs).squeeze(-1)
        neg_embs = self.sigmoid(neg_embs).squeeze(-1)

        return pos_pred, neg_pred, pos_embs, neg_embs



    def predict(self, user_ids, log_seqs, time_matrices, item_indices, s_embs): # for inference

        log_feats, t_embs = self.log2feats(log_seqs, time_matrices) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]

        

        positions = torch.LongTensor(np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])).to(self.dev)

        time_mat = self.pos_emb(positions)
        s_embs = torch.FloatTensor(s_embs).to(self.dev).unsqueeze(0)


        s_embs = s_embs.unsqueeze(1).repeat(1, log_seqs.shape[1], 1)
        ds_embs = self.extractor(s_embs)

        ds_embs += time_mat
        ds_embs = self.fc(ds_embs)

        ds_embs = self.s_embs2feats(ds_embs)[:, -1, :]


        MF_User_Vector_Shared = ds_embs * self.combine_weight
        MF_User_Vector_Specific = final_feat * (1 - self.combine_weight)

        MF_User_Vector = MF_User_Vector_Shared + MF_User_Vector_Specific

        # User_Vector = self.tisas2feats(MF_User_Matrix, log_seqs, time_matrices)

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(MF_User_Vector.unsqueeze(-1)).squeeze(-1)

        # preds = self.sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
    


class TPUF_wo_TFM(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(TPUF_wo_TFM, self).__init__()
        self.args = args
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.GRL = GRL_Layer()


        # self.combine_weight = nn.Parameter(torch.normal(mean=0, std=0.01, size=(2, user_num+1)))
        self.combine_weight = args.combine_weight
        self.softmax = nn.Softmax(dim=0)


        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)


        for _ in range(args.num_blocks+1):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


        self.extractor = Extractor(args)
        self.fc = nn.Linear(args.hidden_units, args.hidden_units)
        self.classifier = Classifier(args)
        self.sigmoid = torch.nn.Sigmoid()


    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        t_embs = seqs
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(self.args.num_blocks):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats, t_embs

    def s_embs2feats(self, s_embs):
        seqs = self.emb_dropout(s_embs)

        # timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        # seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(self.args.num_blocks, self.args.num_blocks+1):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            # seqs *= ~timeline_mask.unsqueeze(-1)

        feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)
        return feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, s_embs): # for training
        positions = torch.LongTensor(np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])).to(self.dev)


        s_embs = s_embs.unsqueeze(1).repeat(1, log_seqs.shape[1], 1)
        ds_embs = self.extractor(s_embs)



        log_feats, t_embs = self.log2feats(log_seqs) # user_ids hasn't been used yet
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))


        MF_User_Vector_Shared = ds_embs * self.combine_weight
        MF_User_Vector_Specific = log_feats * (1 - self.combine_weight)
        MF_User_Vector = MF_User_Vector_Shared + MF_User_Vector_Specific

        pos_logits = (MF_User_Vector * pos_embs).sum(dim=-1)
        neg_logits = (MF_User_Vector * neg_embs).sum(dim=-1)

        pos_pred = self.sigmoid(pos_logits)
        neg_pred = self.sigmoid(neg_logits)

        pos_embs = self.classifier(self.GRL(self.extractor(s_embs)))
        neg_embs = self.classifier(self.GRL(t_embs))


        pos_embs = self.sigmoid(pos_embs).squeeze(-1)
        neg_embs = self.sigmoid(neg_embs).squeeze(-1)



        return pos_pred, neg_pred, pos_embs, neg_embs #0是补位用的

    def predict(self, user_ids, log_seqs, item_indices, s_embs): # for inference

        log_feats, t_embs = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        positions = torch.LongTensor(np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])).to(self.dev)
        # positions = torch.LongTensor([final_feat.shape[1]-1]).to(self.dev)
        time_mat = self.pos_emb(positions)
        s_embs = torch.FloatTensor(s_embs).to(self.dev).unsqueeze(0)



        ds_embs = self.extractor(s_embs)


        # combine_weight = self.softmax(self.combine_weight)
        #
        #
        # MF_User_Vector_Shared = torch.mul(combine_weight[0][user_ids].reshape(len(combine_weight[0][user_ids]), 1), ds_embs)
        #
        # MF_User_Vector_Specific = torch.mul(combine_weight[1][user_ids].reshape(len(combine_weight[1][user_ids]), 1), final_feat)

        MF_User_Vector_Shared = ds_embs * self.combine_weight
        MF_User_Vector_Specific = final_feat * (1 - self.combine_weight)

        MF_User_Vector = MF_User_Vector_Shared + MF_User_Vector_Specific



        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(MF_User_Vector.unsqueeze(-1)).squeeze(-1)

        # preds = self.sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)


