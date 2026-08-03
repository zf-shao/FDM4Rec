import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from mamba_ssm import Mamba
import math


class FDM4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(FDM4Rec, self).__init__(config, dataset)

        self.lambda_ = config['lambda_']
        self.n_layers = config["n_layers"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]

        self.max_len = config["MAX_ITEM_LIST_LENGTH"]
        self.freq_dropout_prob = config["freq_dropout_prob"]

        # Hyperparameters for Mamba block
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]
        self.dropout_prob = config["dropout_prob"]

        self.is_ffn = config["is_ffn"]

        # Embedding
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)

        # === Dual-path Mamba  ===
        self.mamba_layers_forward = nn.ModuleList([
            MambaForwardLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.n_layers,
                layer_norm_eps=self.layer_norm_eps,
                is_ffn=self.is_ffn
            ) for _ in range(self.n_layers)
        ])

        self.mamba_layers_backward = nn.ModuleList([
            MambaBackwardLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.n_layers,
                layer_norm_eps=self.layer_norm_eps,
                is_ffn=self.is_ffn                
            ) for _ in range(self.n_layers)
        ])


        self.filter_layer = FilterLayer(max_seq_length=self.max_len, hidden_size=self.hidden_size, dropout_prob=self.freq_dropout_prob)

        # 拼接层：正向 + 反向 -> hidden_size
        self.concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)


        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self, input_ids, item_seq_len, return_all=False):
        input_emb = self.item_embedding(input_ids)
        input_emb = self.dropout(input_emb)
        input_emb = self.LayerNorm(input_emb)

        # === Mamba ===
        # x = input_emb
        x = self.filter_layer(input_emb)

        x_fwd = x
        x_bwd = x

        for i in range(self.n_layers):
            x_fwd = self.mamba_layers_forward[i](x_fwd)
            x_bwd  = self.mamba_layers_backward[i](x_bwd)

        forward_last = self.gather_indexes(x_fwd, item_seq_len - 1)
        backward_last = self.gather_indexes(x_bwd, item_seq_len - 1)

        concat_output = torch.cat((forward_last, backward_last), dim=-1)
        output = self.concat_layer(concat_output) #

        last_hidden_state = self.gather_indexes(x, item_seq_len - 1)
        output = self.LayerNorm(output + last_hidden_state)
        output = self.dropout(output)

        if return_all:
            return output, forward_last, backward_last
        else:
            return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output, forward_out, backward_out = self.forward(item_seq, item_seq_len, return_all=True)
        pos_items = interaction[self.POS_ITEM_ID]
        test_item_emb = self.item_embedding.weight

        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)

        logits_f = torch.matmul(forward_out, test_item_emb.transpose(0, 1))
        loss_f = self.loss_fct(logits_f, pos_items)

        logits_b = torch.matmul(backward_out, test_item_emb.transpose(0, 1))
        loss_b = self.loss_fct(logits_b, pos_items)

        loss = loss + self.lambda_ * (loss_f + loss_b)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores


class MambaForwardLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers, layer_norm_eps, is_ffn):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = FeedForward(d_model, d_model * 4, dropout, layer_norm_eps)
        
    def forward(self, x):
        hidden_states = self.mamba(x)
        if self.num_layers == 1:        # one Mamba layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(hidden_states))        
        else:
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + x)

        return self.ffn(hidden_states)



class MambaBackwardLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers, layer_norm_eps, is_ffn):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = FeedForward(d_model, d_model * 4, dropout, layer_norm_eps)
        
    def forward(self, x):
        # Backward path (reverse sequence before and after)
        reversed_x = torch.flip(x, dims=[1])
        backward_out = self.mamba(reversed_x)
        backward_out = torch.flip(backward_out, dims=[1])

        if self.num_layers == 1:        # one Mamba layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(backward_out))        
        else:
            hidden_states = self.LayerNorm(self.dropout(backward_out) + x)
        
        return self.ffn(hidden_states)


        


class FilterLayer(nn.Module):
    def __init__(self, max_seq_length, hidden_size, dropout_prob):
        super(FilterLayer, self).__init__()
        
        self.complex_weight = nn.Parameter(
            torch.randn(1, max_seq_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02
        )
        self.out_dropout = nn.Dropout(dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states



class FeedForward(nn.Module):
    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.intermediate_act_fn = nn.GELU()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, layer_norm_eps)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)


        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
