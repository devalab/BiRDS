import math
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


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
        super(BasicBlock, self).__init__()
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
        self,
        layers,
        kernel_size,
        feat_vec_len,
        zero_init_residual=False,
        norm_layer=None,
    ):
        super(MakeResNet, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.start_planes = 64
        self.conv1 = torch.nn.Conv1d(
            in_channels=feat_vec_len,
            out_channels=self.start_planes,
            kernel_size=7,
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(self.start_planes)
        self.relu = torch.nn.ReLU(inplace=True)
        # self.maxpool = torch.nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(64, layers[0], kernel_size)
        self.layer2 = self._make_layer(128, layers[1], kernel_size)
        self.layer3 = self._make_layer(256, layers[2], kernel_size)
        self.layer4 = self._make_layer(512, layers[3], kernel_size)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each
        # residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                torch.nn.init.constant_(m.bn2.weight, 0)

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
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

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
    def __init__(self, feat_vec_len, resnet_layer="resnet6", num_units=64, dropout=0.0):
        super(ResNet, self).__init__()
        self.resnet_layer = globals()[resnet_layer](feat_vec_len=feat_vec_len)
        self.fc1 = torch.nn.Linear(512, num_units)
        self.act1 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(num_units, 1)
        # self.act2 = torch.nn.Sigmoid()

    def forward(self, X, lengths, **kwargs):
        # [Batch, feat_vec_len, Max_length] -> [Batch, 512, Max_length]
        X = self.resnet_layer(X)

        # [Batch, 512, Max_length] -> [Batch * Max_length, 512]
        X = X.transpose(1, 2)
        X = X.contiguous().view(-1, X.shape[2])

        # Run through linear and activation layers
        X = self.fc1(X)
        X = self.act1(X)
        X = self.dropout(X)
        X = self.fc2(X)
        # X = self.act2(X)

        # [Batch * Max_length, 1] -> [Batch, Max_length]
        X = X.view(-1, lengths[0])

        return X


class BiLSTM(torch.nn.Module):
    def __init__(self, feat_vec_len, hidden_sizes=[32, 1], num_layers=2, dropout=0.5):
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


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

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


class Transformer(torch.nn.Module):
    def __init__(
        self, feat_vec_len, ntoken=21, nhead=7, nhid=512, nlayers=2, dropout=0.5
    ):
        # ntoken: number of amino acids
        # nhead: the number of heads in the multiheadattention models
        # nhid: the dimension of the feedforward network model in torch.nn.TransformerEncoder
        # nlayers: the number of torch.nn.TransformerEncoderLayer in torch.nn.TransformerEncoder
        super(Transformer, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feat_vec_len, dropout)
        encoder_layers = torch.nn.TransformerEncoderLayer(
            feat_vec_len, nhead, nhid, dropout
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        # Final output which gives probability of the AA being in the binding site or not
        self.decoder = torch.nn.Linear(feat_vec_len, 1)

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


# Padding might be causing a problem


class UNetConvBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(
            torch.nn.Conv1d(in_size, out_size, kernel_size=3, padding=int(padding))
        )
        block.append(torch.nn.ReLU())
        if batch_norm:
            block.append(torch.nn.BatchNorm1d(out_size))

        block.append(
            torch.nn.Conv1d(out_size, out_size, kernel_size=3, padding=int(padding))
        )
        block.append(torch.nn.ReLU())
        if batch_norm:
            block.append(torch.nn.BatchNorm1d(out_size))

        self.block = torch.nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == "upconv":
            self.up = torch.nn.ConvTranspose1d(
                in_size, out_size, kernel_size=2, stride=2
            )
        elif up_mode == "upsample":
            self.up = torch.nn.Sequential(
                torch.nn.Upsample(mode="linear", scale_factor=2),
                torch.nn.Conv1d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        # diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y : (diff_y + target_size[0])]

    def pad(self, up, dim_2):
        if dim_2 == 0:
            return up
        return torch.cat([up, torch.zeros(up.shape[0], up.shape[1], dim_2)], 2)

    def forward(self, x, bridge):
        up = self.up(x)
        # crop1 = self.center_crop(bridge, up.shape[2:])
        up = self.pad(up, bridge.shape[2] - up.shape[2])
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out


class UNet(torch.nn.Module):
    def __init__(
        self,
        feat_vec_len,
        n_classes=1,
        depth=5,
        wf=6,
        padding=True,
        batch_norm=False,
        up_mode="upconv",
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            feat_vec_len (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ("upconv", "upsample")
        self.padding = padding
        self.depth = depth
        prev_channels = feat_vec_len
        self.down_path = torch.nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = torch.nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = torch.nn.Conv1d(prev_channels, n_classes, kernel_size=1)

    def forward(self, X, lengths, **kwargs):
        blocks = []
        for i, down in enumerate(self.down_path):
            X = down(X)
            if i != len(self.down_path) - 1:
                blocks.append(X)
                X = F.max_pool1d(X, 2)

        for i, up in enumerate(self.up_path):
            X = up(X, blocks[-i - 1])

        return self.last(X).view(-1, lengths[0])