import torch
import math


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
        # Don't use += will cause inplace operation leading to error
        out = out + identity

        return out


class MakeResNet(torch.nn.Module):
    def __init__(self, layers, kernel_size, input_size, hidden_sizes, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.start_planes = hidden_sizes[0]
        self.conv1 = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=self.start_planes,
            kernel_size=7,
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(self.start_planes)
        self.relu = torch.nn.ReLU()
        self.depth = len(hidden_sizes)
        # self.maxpool = torch.nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.layers = torch.nn.ModuleList([])
        for i in range(self.depth):
            self.layers.append(self._make_layer(hidden_sizes[i], layers[i], kernel_size))

        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
        layers.append(BasicBlock(self.start_planes, out_planes, kernel_size, norm_layer, downsample))
        self.start_planes = out_planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.start_planes, out_planes, kernel_size, norm_layer))

        return torch.nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        for i in range(self.depth):
            x = self.layers[i](x)

        return x


class ResNet(torch.nn.Module):
    def __init__(self, input_size, hparams):
        super().__init__()
        assert len(hparams.layers) == len(hparams.hidden_sizes)
        self.use_ohe = hparams.use_ohe
        if hparams.use_ohe:
            self.input_size = input_size - 21 + hparams.embedding_size
            self.embedding = BERTEmbedding(vocab_size=21, embed_size=hparams.embedding_size)
        else:
            self.input_size = input_size
        self.resnet_layer = MakeResNet(hparams.layers, hparams.kernel_sizes, self.input_size, hparams.hidden_sizes)

    def forward(self, X, lengths, **kwargs):
        # [Batch, input_size, Max_length] -> [Batch, hidden_sizes[-1], Max_length]
        if self.use_ohe:
            ohe = X[:, :21].argmax(dim=1)
            ohe = self.embedding(ohe, segment_label=kwargs["segment_label"])
            out = torch.cat((ohe.transpose(1, 2), X[:, 21:]), dim=1)
        else:
            out = X

        out = self.resnet_layer(out)
        return out

    @staticmethod
    def add_class_specific_args(parser):
        parser.add_argument(
            "--embedding-size",
            type=int,
            default=32,
            help="Embedding size. Default: %(default)s",
        )
        parser.add_argument(
            "--layers",
            nargs="+",
            type=int,
            default=[2, 2, 2, 2, 2],
            help="The number of basic blocks to be used in each layer. Default: %(default)s",
        )
        parser.add_argument(
            "--hidden-sizes",
            nargs="+",
            type=int,
            default=[128, 256, 128, 64, 32],
            help="The size of the 1-D convolutional layers. Default: %(default)s",
        )
        parser.add_argument(
            "--kernel-sizes",
            nargs="+",
            type=int,
            default=[5, 5],
            help="Kernel sizes of the 2 convolutional layers of the basic block. Default: %(default)s",
        )
        return parser


class Detector(torch.nn.Module):
    def __init__(self, input_size, hparams):
        super().__init__()
        self.input_size = input_size
        layers = []
        for unit in hparams.detector_units:
            layers.append(torch.nn.Linear(input_size, unit))
            layers.append(torch.nn.LeakyReLU(0.2, True))
            layers.append(torch.nn.Dropout(hparams.dropout))
            input_size = unit
        self.detector = torch.nn.Sequential(*layers, torch.nn.Linear(input_size, 1))
        # torch.sigmoid will be done later in the loss function

    def forward(self, X, **kwargs):
        # [Batch, Max_len]
        return self.detector(X).squeeze(dim=2)

    @staticmethod
    def add_class_specific_args(parser):
        parser.add_argument(
            "--detector-units",
            metavar="UNIT",
            nargs="+",
            type=int,
            default=[8],
            help="The number of units in each layer of the detector. Default: %(default)s",
        )
        return parser


class GELU(torch.nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(torch.nn.Embedding):
    def __init__(self, vocab_size, embed_size=32):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class SegmentEmbedding(torch.nn.Embedding):
    def __init__(self, embed_size=32):
        super().__init__(17, embed_size, padding_idx=0)


class BERTEmbedding(torch.nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.2):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)
