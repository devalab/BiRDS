import math

import torch
import torch.nn as nn


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
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, n_heads=4, num_transformer_layers=2):
        super(Transformer, self).__init__()
        self.fc1 = nn.Linear(21, 64)
        self.positional_encoding = PositionalEncoding(64)
        transformer_encoder_layer = nn.TransformerEncoderLayer(64, n_heads)
        self.transformer = nn.TransformerEncoder(
            transformer_encoder_layer, num_transformer_layers
        )
        self.fc2 = nn.Linear(64, 1)

    def forward(self, X, lengths, *args):
        # batch_size = len(lengths)
        # mask = torch.zeros(batch_size, lengths[0], device=DEVICE, dtype=torch.bool)
        # for i in range(len(lengths)):
        #     mask[i, : lengths[i]] = 1

        # [Batch, 21, Max_length] -> [Batch, Max_length, 64]
        X = torch.transpose(X, 1, 2)
        X = self.fc1(X)
        X = self.positional_encoding(X)
        X = self.transformer(X)

        # Run through linear and activation layers
        X = self.fc2(X)
        X = X.view(-1, lengths[0])

        return X
