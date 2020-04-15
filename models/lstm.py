import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from constants import DEVICE


class LSTM(torch.nn.Module):
    def __init__(
        self,
        feat_vec_len,
        num_lstm_units=256,
        num_units=64,
        num_lstm_layers=2,
        dropout=0.0,
    ):
        super(LSTM, self).__init__()
        self.num_input_dims = feat_vec_len
        self.num_lstm_units = num_lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.num_units = 64
        # self.brnn = nn.LSTM(
        #     input_size=input_dims,
        #     hidden_size=lstm_dims,
        #     num_lstm_layers=2,
        #     bias=True,
        #     batch_first=True,
        #     bidirectional=True,
        # )
        self.lstm = torch.nn.LSTM(
            input_size=self.num_input_dims,
            hidden_size=self.num_lstm_units,
            num_layers=self.num_lstm_layers,
            batch_first=True,
        )
        self.fc1 = torch.nn.Linear(self.num_lstm_units, self.num_units)
        self.act1 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(num_units, 1)

    def init_hidden(self, batch_size):
        # the weights are of the form (num_lstm_layers, batch_size, num_lstm_units)
        hidden_state = torch.randn(self.num_lstm_layers, batch_size, self.num_lstm_units).to(DEVICE)
        cell_state = torch.randn(self.num_lstm_layers, batch_size, self.num_lstm_units).to(DEVICE)
        return (hidden_state, cell_state)

    def forward(self, X, lengths, **kwargs):
        self.hidden = self.init_hidden(len(lengths))
        # [Batch, feat_vec_len, Max_length] -> [Batch, Max_length, feat_vec_len]
        output = X.transpose(1, 2)

        # [Batch, Max_length, feat_vec_len] -> [Batch, Max_length, num_lstm_units]
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        output = pack_padded_sequence(output, lengths, batch_first=True)
        # now run through LSTM
        output, self.hidden = self.lstm(output, self.hidden)

        # undo the packing operation
        output, _ = pad_packed_sequence(output, batch_first=True)

        # Project to target space
        # [Batch, Max_length, num_lstm_units] -> [Batch * Max_length, num_lstm_units]
        output = output.contiguous().view(-1, self.num_lstm_units)

        # Run through linear and activation layers
        output = self.fc1(output)
        output = self.act1(output)
        output = self.dropout(output)
        output = self.fc2(output)

        # [Batch * Max_length, 1] -> [Batch, Max_length]
        output = output.view(-1, lengths[0])

        return output
