import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM(torch.nn.Module):
    def __init__(
        self, feat_vec_len, hidden_sizes=[32, 1], num_layers=2, dropout=0.5
    ):
        super(BiLSTM, self).__init__()
        self.feat_vec_len = feat_vec_len
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.depth = len(hidden_sizes)
        self.brnn = torch.nn.ModuleList([])
        for i in range(self.depth):
            self.brnn.append(
                torch.nn.LSTM(
                    input_size=feat_vec_len,
                    hidden_size=hidden_sizes[i],
                    num_layers=num_layers,
                    bias=True,
                    dropout=dropout,
                    bidirectional=True,
                )
            )
            feat_vec_len = hidden_sizes[i]

    def forward(self, X, lengths, **kwargs):
        # [Batch, feat_vec_len, Max_length] -> [Max_length, Batch, feat_vec_len]
        output = X.transpose(1, 2).transpose(0, 1)
        batch_size = len(lengths)
        for i in range(self.depth):
            # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
            output = pack_padded_sequence(output, lengths)
            # If we don't send hidden and cell state to LSTM, they default to zero
            # [Max_length, Batch, hidden_sizes[i-1]] -> [Max_length, Batch, 2 * hidden_sizes[i]]
            # If i is 0 then hidden_sizes[i-1] is self.feat_vec_len
            output, _ = self.brnn[i](output)
            # undo the packing operation
            output, _ = pad_packed_sequence(output)
            # We need to change our last output dimension. We'll use averaging
            # [Max_length, Batch, 2 * hidden_sizes[i]] -> [Max_length, Batch, hidden_sizes[i]]
            output = output.view(lengths[0], batch_size, 2, self.hidden_sizes[i])
            output = output.mean(dim=2)

        # [Max_length, Batch, hidden_sizes[-1]] -> [Batch, Max_length]
        output = output.transpose(0, 1).squeeze(dim=2)

        return output

