import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from constants import DEVICE
from models.resnet_1d import resnet6, resnet18
from models.lstm import LSTM


class ResNetLSTM(torch.nn.Module):
    def __init__(self, feat_vec_len):
        super(ResNetLSTM, self).__init__()
        self.resnet_layer = resnet6(feat_vec_len=feat_vec_len)
        self.lstm = torch.nn.LSTM(
            input_size=512, hidden_size=256, num_layers=2, batch_first=True,
        )
        self.fc1 = torch.nn.Linear(256, 64)
        self.act1 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.0)
        self.fc2 = torch.nn.Linear(64, 1)

    def init_hidden(self, batch_size):
        # the weights are of the form (num_lstm_layers, batch_size, 256)
        hidden_state = torch.randn(2, batch_size, 256).to(DEVICE)
        cell_state = torch.randn(2, batch_size, 256).to(DEVICE)
        return (hidden_state, cell_state)

    def forward(self, X, lengths, **kwargs):
        # [Batch, feat_vec_len, Max_length] -> [Batch, 512, Max_length]
        X = self.resnet_layer(X)
        self.hidden = self.init_hidden(len(lengths))

        # [Batch, 512, Max_length] -> [Batch, Max_length, 512]
        X = X.transpose(1, 2)
        X = pack_padded_sequence(X, lengths, batch_first=True)
        # now run through LSTM
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = pad_packed_sequence(X, batch_first=True)

        # Project to target space
        # [Batch, Max_length, 256] -> [Batch * Max_length, 256]
        X = X.contiguous().view(-1, 256)

        # Run through linear and activation layers
        X = self.fc1(X)
        X = self.act1(X)
        X = self.dropout(X)
        X = self.fc2(X)

        # [Batch * Max_length, 1] -> [Batch, Max_length]
        X = X.view(-1, lengths[0])

        return X
