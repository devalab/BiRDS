from argparse import ArgumentParser

import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3 convolution with padding"""
    padding = 1 + (dilation - 1)  # derived to ensure consistent size
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=True,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


def conv64x1(in_planes, out_planes, stride=1, groups=1):
    """64 convolution"""
    return nn.Conv1d(
        in_planes, out_planes, kernel_size=64, stride=stride, groups=groups, bias=True,
    )


class BasicBlock(nn.Module):
    def __init__(
        self, inplanes, planes, stride=1, downsample=None, norm_layer=None, dilation=1
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.project_down = conv1x1(128, 64, stride=1)
        self.project_up = conv1x1(64, 128, stride=1)
        self.bn64_1 = norm_layer(64)
        self.bn64_2 = norm_layer(64)
        self.bn128 = norm_layer(128)

        # dilations deal now with 64 incoming and 64 outcoming layers
        self.dilation = conv3x3(
            64, 64, stride, dilation=dilation
        )  # when the block is initialized, the only thing that changes is the dilation filter used!
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):

        identity = x

        # the deepmind basic block goes:

        # batchnorm
        out = self.bn128(x)

        # elu
        out = self.elu(out)

        # project down to 64
        out = self.project_down(out)

        # batchnorm
        out = self.bn64_1(out)

        # elu
        out = self.elu(out)

        # cycle through 4 dilations
        out = self.dilation(out)

        # batchnorm
        out = self.bn64_2(out)

        # elu
        out = self.elu(out)

        # project up to 128
        out = self.project_up(out)

        # identitiy addition
        out = out + identity

        return out


class ResNet(nn.Module):
    def __init__(self, feat_vec_len, hparams):
        super(ResNet, self).__init__()
        norm_layer = nn.BatchNorm1d
        self.inplanes = hparams.num_layers

        self.conv1 = conv1x1(feat_vec_len, hparams.num_layers, stride=1)
        self.conv2 = conv1x1(hparams.num_layers, hparams.output_size)

        self.proj_aux = conv1x1(hparams.num_layers, 83, stride=1)
        self.conv_aux = conv64x1(83, 83, stride=1, groups=1)

        self.elu = nn.ELU(inplace=True)

        self.bn1 = norm_layer(feat_vec_len)

        self.resnet_blocks = self._make_layer(
            hparams.num_layers, hparams.num_blocks, norm_layer=norm_layer
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if hparams.zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride), norm_layer(planes),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes

        # here I need to pass in the correct dilations 1,2,4,8
        dilations = [1, 2, 4, 8]

        for i, _ in enumerate(range(1, blocks)):
            layers.append(
                BasicBlock(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer,
                    dilation=dilations[i % 4],
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x, lengths):
        # fix input dimensions
        x = self.bn1(x)  # Why?
        x = self.conv1(x)
        # 1,128,L

        # propagate through RESNET blocks
        resnet_out = self.resnet_blocks(x)
        # renet_out has shape 1,128,L

        # fix output dimensions
        x = self.conv2(resnet_out)  # return 1,64,220

        # aux = self.proj_aux(resnet_out)
        # aux = self.conv_aux(self.proj_aux(resnet_out))
        # should we have elu / batchnorm(s) here??
        # aux_i = self.conv_aux(torch.transpose(aux, 2, 3))
        # aux_j = self.conv_aux(aux)

        return x

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument(
            "--num_blocks",
            type=int,
            default=220,
            help="The number of ResNet blocks to be used",
        )
        parser.add_argument(
            "--num_layers",
            type=int,
            default=128,
            help="The number of layers inside of ResNet",
        )
        parser.add_argument(
            "--output_size", type=int, default=64, help="Output dimension",
        )
        parser.add_argument(
            "--zero_init_residual", dest="zero_init_residual", action="store_true"
        )
        parser.add_argument(
            "--no_zero_init_residual", dest="zero_init_residual", action="store_false"
        )
        parser.set_defaults(zero_init_residual=False)
        return parser
