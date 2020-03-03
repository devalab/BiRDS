import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import DEVICE

# Padding might be causing a problem

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv1d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm1d(out_size))

        block.append(nn.Conv1d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm1d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose1d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="linear", scale_factor=2),
                nn.Conv1d(in_size, out_size, kernel_size=1),
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
        return torch.cat(
            [up, torch.zeros(up.shape[0], up.shape[1], dim_2).to(DEVICE)], 2
        )

    def forward(self, x, bridge):
        up = self.up(x)
        # crop1 = self.center_crop(bridge, up.shape[2:])
        up = self.pad(up, bridge.shape[2] - up.shape[2])
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out


class UNet(nn.Module):
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
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv1d(prev_channels, n_classes, kernel_size=1)

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