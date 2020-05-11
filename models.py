from argparse import ArgumentParser

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BasicBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=[3, 3],
        norm_layer=None,
        downsample=None,
    ):
        # Since we need same length output, we can't have
        # downsampling, dilations or strides
        super().__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm1d
        if type(kernel_size) is not list or len(kernel_size) != 2:
            raise ValueError("BasicBlock requires a list of length 2 for kernel_size")
        self.conv1 = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size[0],
            padding=kernel_size[0] // 2,
            bias=False,
        )
        self.bn1 = norm_layer(out_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size[1],
            padding=kernel_size[1] // 2,
            bias=False,
        )
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample

    def forward(self, x, **kwargs):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.relu(out)
        out += identity

        return out


class MakeResNet(torch.nn.Module):
    def __init__(
        self, layers, kernel_size, feat_vec_len, hidden_sizes, norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.start_planes = hidden_sizes[0]
        self.conv1 = torch.nn.Conv1d(
            in_channels=feat_vec_len,
            out_channels=self.start_planes,
            kernel_size=7,
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(self.start_planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.depth = len(hidden_sizes)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.layers = torch.nn.ModuleList([])
        for i in range(self.depth):
            self.layers.append(
                self._make_layer(hidden_sizes[i], layers[i], kernel_size)
            )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_planes, blocks, kernel_size):
        norm_layer = self._norm_layer
        downsample = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=self.start_planes,
                out_channels=out_planes,
                kernel_size=1,
                bias=False,
            ),
            norm_layer(out_planes),
        )
        layers = []
        layers.append(
            BasicBlock(
                self.start_planes, out_planes, kernel_size, norm_layer, downsample
            )
        )
        self.start_planes = out_planes
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(self.start_planes, out_planes, kernel_size, norm_layer)
            )

        return torch.nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(self.depth):
            x = self.layers[i](x)

        return x


def resnet6(**kwargs):
    return MakeResNet([1, 1, 1, 1], [3, 3], **kwargs)


def resnet14(**kwargs):
    return MakeResNet([1, 2, 2, 1], [3, 3], **kwargs)


def resnet18(**kwargs):
    return MakeResNet([2, 2, 2, 2], [5, 5], **kwargs)


def resnet34(**kwargs):
    return MakeResNet([3, 4, 6, 3], [7, 7], **kwargs)


def resnet66(**kwargs):
    return MakeResNet([3, 8, 18, 3], [9, 9], **kwargs)


def resnet98(**kwargs):
    return MakeResNet([3, 12, 30, 3], [11, 11], **kwargs)


class ResNet(torch.nn.Module):
    def __init__(self, feat_vec_len, hparams):
        super().__init__()
        assert len(hparams.layers) == len(hparams.hidden_sizes)
        self.feat_vec_len = feat_vec_len
        self.resnet_layer = MakeResNet(
            hparams.layers, hparams.kernel_sizes, feat_vec_len, hparams.hidden_sizes
        )

    def forward(self, X, lengths, **kwargs):
        # [Batch, feat_vec_len, Max_length] -> [Batch, hidden_sizes[-1], Max_length]
        return self.resnet_layer(X)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--layers",
            nargs="+",
            type=int,
            default=[2, 2, 2, 2],
            help="The number of basic blocks to be used in each layer. Default: 2 2 2 2 forms Resnet-18",
        )
        parser.add_argument(
            "--hidden_sizes",
            nargs="+",
            type=int,
            default=[256, 128, 64, 32],
            help="The size of the 1-D convolutional layers. Eg: 256 128 64 32",
        )
        parser.add_argument(
            "--kernel_sizes",
            nargs="+",
            type=int,
            default=[7, 7],
            help="Kernel sizes of the 2 convolutional layers that form the basic block of the Resnet. Default: 7 7",
        )
        return parser


class BiLSTM(torch.nn.Module):
    def __init__(self, feat_vec_len, hparams):
        super().__init__()
        self.feat_vec_len = feat_vec_len
        self.hidden_sizes = hparams.hidden_sizes
        self.depth = len(self.hidden_sizes)
        self.brnn = torch.nn.ModuleList([])
        for i in range(self.depth):
            self.brnn.append(
                torch.nn.LSTM(
                    input_size=feat_vec_len,
                    hidden_size=self.hidden_sizes[i],
                    num_layers=hparams.num_layers,
                    bias=True,
                    # dropout=hparams.dropout,
                    bidirectional=True,
                )
            )
            feat_vec_len = self.hidden_sizes[i]

    def forward(self, X, lengths, **kwargs):
        # [Batch, feat_vec_len, Max_length] -> [Max_length, Batch, feat_vec_len]
        output = X.transpose(1, 2).transpose(0, 1)

        # [Max_length, Batch, feat_vec_len] -> [Max_length, Batch, hidden_sizes[-1]]
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
            self.dropout(output)

        # [Max_length, Batch, hidden_sizes[-1]] -> [Batch, hidden_size[-1], Max_length]
        return output.transpose(0, 1).transpose(1, 2)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--hidden_sizes",
            nargs="+",
            type=int,
            default=[256, 128, 64, 32],
            help="The size of each stacked LSTM layers. Eg: 256 128 64 32",
        )
        parser.add_argument(
            "--num_layers",
            type=int,
            default=1,
            help="Number of LSTM units in each layer. Default: 1",
        )
        return parser
