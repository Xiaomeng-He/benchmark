import torch
import math
import torch.nn as nn
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, prefix_len):
        """

        Parameters
        ----------
        d_model: int
            Dimensionality of the concatenated tensor
        prefix_len: int
            Length of prefix

        """
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(prefix_len, d_model)
        position = torch.arange(0, prefix_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # ensure pe isn't a learnable parameter during training
        
    def forward(self, x):
        # add the positional embeddings to the token embeddings
        x = x + self.pe[:, :x.size(1)]
        return x

class Transformer(nn.Module):
    def __init__(self, 
                 prefix_len, 
                 num_act, num_time_features, d_embed, 
                 d_model, num_heads, d_ff, dropout,
                 num_layers):
        super().__init__()
        self.d_model = d_model
        self.prefix_len = prefix_len
        self.embedding = nn.Embedding(num_act, d_embed, 0)
        self.time_proj = nn.Linear(num_time_features, d_embed)
        self.input_proj = nn.Linear(d_embed * 2, d_model)
        self.positional_encoding = PositionalEncoding(d_model, prefix_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                        nhead=num_heads,
                                                        dim_feedforward=d_ff,
                                                        dropout=dropout,
                                                        batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_act)

    def forward(self,
                trace_prefix_act,
                trace_prefix_time):
        
        act_embed = self.embedding(trace_prefix_act) * math.sqrt(self.d_model)
        time_proj = self.time_proj(trace_prefix_time)
        x = torch.cat([act_embed, time_proj], dim=-1)
        x = self.input_proj(x)
        assert self.d_model == x.size(-1), "d_model must be equal to the last dimension of x"
        x = self.positional_encoding(x)

        # pad mask
        padding_mask = (trace_prefix_act == 0)
        padding_mask = torch.where(padding_mask, float('-inf'), 0.0)

        # causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(self.prefix_len).to(trace_prefix_act.device)

        # pass x to transformer
        outputs = self.encoder(src=x, 
                                mask=causal_mask,
                               src_key_padding_mask=padding_mask,
                               is_causal=True) # shape: (num_obs, trace_len, d_model)
        
        # make predictions
        predictions = self.classifier(outputs) # shape: (num_obs, trace_len, num_act)

        return predictions

class LSTM(nn.Module):

    def __init__(self, 
                 num_act, num_time_features, d_embed,
                 hidden_size, num_layers, dropout, bidirect):
        
        super().__init__()
        
        self.embedding = nn.Embedding(num_act, d_embed, 0)
        self.time_proj = nn.Linear(num_time_features, d_embed)

        self.lstm = nn.LSTM(input_size=d_embed * 2,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  dropout=dropout,
                                  bidirectional=bidirect)
        
        if bidirect:
            bi = 2
        else:
            bi = 1
        self.classifier = nn.Linear(hidden_size*bi, num_act)

    def forward(self, 
                trace_prefix_act,
                trace_prefix_time):
        
        act_embed = self.embedding(trace_prefix_act)
        time_proj = self.time_proj(trace_prefix_time)
        x = torch.cat([act_embed, time_proj], dim=-1) # shape: (batch_size, prefix_len, d_embed * 2)
        
        # pass x to lstm
        outputs, _ = self.lstm(x) # shape: (batch_size, prefix_len, hidden_size (*2 if bidirectional))

        # make predictions
        predictions = self.classifier(outputs) # shape: (batch_size, prefix_len, num_act)

        return predictions

class xLSTM(nn.Module):

    def __init__(self,
                 prefix_len,
                 num_act, num_time_features, d_embed,
                 mix_mode,
                 num_blocks, slstm_at):
        
        super().__init__()
        
        self.embedding = nn.Embedding(num_act, d_embed, 0)
        self.time_proj = nn.Linear(num_time_features, d_embed)
        
        mlstmconfig = mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, 
                    qkv_proj_blocksize=4, 
                    num_heads=4
                    ))
        
        slstmconfig = sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend= 'vanilla',
                    # "cuda" if torch.cuda.is_available() else 'vanilla'
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                    ),
                    feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"))

        mlstm_block_config = mlstmconfig if mix_mode in ("mlstm", "mix") else None
        slstm_block_config = slstmconfig if mix_mode in ("slstm", "mix") else None

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mlstm_block_config,
            slstm_block=slstm_block_config,
            context_length=prefix_len,
            num_blocks=num_blocks,
            embedding_dim=d_embed*2,
            slstm_at=slstm_at)

        self.xlstm_stack = xLSTMBlockStack(cfg)
        
        self.classifier = nn.Linear(d_embed*2, num_act)

    def forward(self, 
                trace_prefix_act,
                trace_prefix_time):
        
        
        act_embed = self.embedding(trace_prefix_act)
        time_proj = self.time_proj(trace_prefix_time)
        x = torch.cat([act_embed, time_proj], dim=-1) # shape: (batch_size, prefix_len, d_embed * 2)
        
        # pass x to xlstm
        outputs = self.xlstm_stack(x) # shape: (batch_size, prefix_len, d_embed*2)

        # make predictions
        predictions = self.classifier(outputs) # shape: (batch_size, prefix_len, num_act)

        return predictions
