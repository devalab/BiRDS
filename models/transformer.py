import math

import torch
import torch.nn as nn
from constants import DEVICE, AA_ID_DICT


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if len(div_term) % 2:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
        self, feat_vec_len, ntoken=21, nhead=7, nhid=512, nlayers=2, dropout=0.5
    ):
        # ntoken: number of amino acids
        # nhead: the number of heads in the multiheadattention models
        # nhid: the dimension of the feedforward network model in nn.TransformerEncoder
        # nlayers: the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        super(Transformer, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feat_vec_len, dropout)
        encoder_layers = nn.TransformerEncoderLayer(feat_vec_len, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        # Final output which gives probability of the AA being in the binding site or not
        self.decoder = nn.Linear(feat_vec_len, 1)

        self.init_weights()

    # def _generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = (
    #         mask.float()
    #         .masked_fill(mask == 0, float("-inf"))
    #         .masked_fill(mask == 1, float(0.0))
    #     )
    #     return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, X, lengths, **kwargs):
        # if self.src_mask is None or self.src_mask.size(0) != len(X):
        #     mask = self._generate_square_subsequent_mask(len(X)).to(DEVICE)
        #     self.src_mask = mask

        # [Batch, 21, Max_length] -> [Batch, Max_length, 21]
        X = X.transpose(1, 2)
        X = self.pos_encoder(X)
        # Batch size 1 so mask is not included...
        output = self.transformer_encoder(X)
        output = self.decoder(output)
        return output.view(-1, lengths[0])
