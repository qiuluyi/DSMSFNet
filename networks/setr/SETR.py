import torch
import torch.nn as nn
from networks.setr.Transformer import TransformerModel
from networks.setr.PositionalEncoding import (
    FixedPositionalEncoding,
    LearnedPositionalEncoding,
)
from networks.setr.IntmdSequential import IntermediateSequential

__all__ = [
    'SETR_Naive_S',
    'SETR_Naive_L',
    'SETR_PUP_S',
    'SETR_PUP_L',
    'SETR_MLA_S',
    'SETR_MLA_L',
]


class SegmentationTransformer(nn.Module):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
            conv_patch_representation=False,
            positional_encoding_type="learned",
    ):
        super(SegmentationTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        if self.conv_patch_representation:
            self.conv_x = nn.Conv2d(
                self.num_channels,
                self.embedding_dim,
                kernel_size=(self.patch_dim, self.patch_dim),
                stride=(self.patch_dim, self.patch_dim),
                padding=self._get_padding(
                    'VALID', (self.patch_dim, self.patch_dim),
                ),
            )
        else:
            self.conv_x = None

    def encode(self, x):
        n, c, h, w = x.shape
        if self.conv_patch_representation:
            # combine embedding w/ conv patch distribution
            x = self.conv_x(x)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
        else:
            x = (
                x.unfold(2, self.patch_dim, self.patch_dim)
                    .unfold(3, self.patch_dim, self.patch_dim)
                    .contiguous()
            )
            x = x.view(n, c, -1, self.patch_dim ** 2)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
            x = self.linear_encoding(x)

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)

        return x, intmd_x

    def decode(self, x):
        raise NotImplementedError("Should be implemented in child class!!")

    def forward(self, x, auxillary_output_layers=None):
        encoder_output, intmd_encoder_outputs = self.encode(x)
        decoder_output = self.decode(
            encoder_output, intmd_encoder_outputs, auxillary_output_layers
        )

        if auxillary_output_layers is not None:
            auxillary_outputs = {}
            for i in auxillary_output_layers:
                val = str(2 * i - 1)
                _key = 'Z' + str(i)
                auxillary_outputs[_key] = intmd_encoder_outputs[val]

            return decoder_output, auxillary_outputs

        return decoder_output

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SETR_Naive(SegmentationTransformer):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            num_classes,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
            conv_patch_representation=False,
            positional_encoding_type="learned",
    ):
        super(SETR_Naive, self).__init__(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
        )

        self.num_classes = num_classes

    def decode(self, x, intmd_x, intmd_layers=None):
        x = self._reshape_output(x)
        x = nn.Conv2d(
            in_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1,
            padding=self._get_padding('VALID', (1, 1), ),
        )(x)
        x = nn.BatchNorm2d(self.embedding_dim)(x)
        x = nn.ReLU()(x)

        x = nn.Conv2d(
            in_channels=self.embedding_dim,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            padding=self._get_padding('VALID', (1, 1), ),
        )(x)
        x = nn.Upsample(scale_factor=self.patch_dim, mode='bilinear')(x)

        return x


class SETR_PUP(SegmentationTransformer):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            num_classes,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
            conv_patch_representation=False,
            positional_encoding_type="learned",
    ):
        super(SETR_PUP, self).__init__(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
        )

        self.num_classes = num_classes
        self.last = nn.Sigmoid()
        self.last_conv = nn.Conv2d(int(self.embedding_dim / 4), self.num_classes, kernel_size=1)

    def decode(self, x, intmd_x, intmd_layers=None):
        x = self._reshape_output(x)
        extra_in_channels = int(self.embedding_dim / 4)
        in_channels = [
            self.embedding_dim,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
        ]
        out_channels = [
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
        ]

        conv_layers = []
        upsample_layers = []

        for i, (in_channel, out_channel) in enumerate(
                zip(in_channels, out_channels)
        ):
            conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=self._get_padding('VALID', (1, 1), ),
                )
            )
            if i != 4:
                upsample_layers.append(
                    nn.Upsample(scale_factor=2, mode='bilinear')
                )
        for (conv_layer, upsample_layer) in zip(conv_layers, upsample_layers):
            x = conv_layer(x)
            x = upsample_layer(x)
        x = conv_layers[-1](x)
        x = self.last(self.last_conv(x))
        return x


class SETR_MLA(SegmentationTransformer):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            num_classes,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
            conv_patch_representation=False,
            positional_encoding_type="learned",
    ):
        super(SETR_MLA, self).__init__(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
        )

        self.num_classes = num_classes

    def decode(self, x, intmd_x, intmd_layers=None):
        assert intmd_layers is not None, "pass the intermediate layers for MLA"

        encoder_outputs = {}
        all_keys = []
        for i in intmd_layers:
            val = str(2 * i - 1)
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = intmd_x[val]
        all_keys.reverse()

        net1_in, _, net1_out = self._define_agg_net()
        temp_x = encoder_outputs[all_keys[0]]
        temp_x = self._reshape_output(temp_x)
        key0_intmd_in = net1_in(temp_x)
        key0_out = net1_out(key0_intmd_in)

        net2_in, net2_intmd, net2_out = self._define_agg_net()
        temp_x = encoder_outputs[all_keys[1]]
        temp_x = self._reshape_output(temp_x)
        key1_in = net2_in(temp_x)
        key1_intmd_in = key1_in + key0_intmd_in
        key1_intmd_out = net2_intmd(key1_intmd_in)
        key1_out = net2_out(key1_intmd_out)

        net3_in, net3_intmd, net3_out = self._define_agg_net()
        temp_x = encoder_outputs[all_keys[2]]
        temp_x = self._reshape_output(temp_x)
        key2_in = net3_in(temp_x)
        key2_intmd_in = key2_in + key1_intmd_in
        key2_intmd_out = net3_intmd(key2_intmd_in)
        key2_out = net3_out(key2_intmd_out)

        net4_in, net4_intmd, net4_out = self._define_agg_net()
        temp_x = encoder_outputs[all_keys[3]]
        temp_x = self._reshape_output(temp_x)
        key3_in = net4_in(temp_x)
        key3_intmd_in = key3_in + key2_intmd_in
        key3_intmd_out = net4_intmd(key3_intmd_in)
        key3_out = net4_out(key3_intmd_out)

        out = torch.cat((key0_out, key1_out, key2_out, key3_out), dim=1)
        out = nn.Conv2d(
            in_channels=self.embedding_dim,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            padding=self._get_padding('VALID', (1, 1), ),
        )(out)
        out = nn.Upsample(scale_factor=4, mode='bilinear')(out)

        return out

    # fmt: off
    def _define_agg_net(self):
        model_in = IntermediateSequential(return_intermediate=False)
        model_in.add_module(
            "layer_1",
            nn.Conv2d(
                self.embedding_dim, int(self.embedding_dim / 2), 1, 1,
                padding=self._get_padding('VALID', (1, 1), ),
            ),
        )

        model_intmd = IntermediateSequential(return_intermediate=False)
        model_intmd.add_module(
            "layer_intmd",
            nn.Conv2d(
                int(self.embedding_dim / 2), int(self.embedding_dim / 2), 3, 1,
                padding=self._get_padding('SAME', (3, 3), ),
            ),
        )

        model_out = IntermediateSequential(return_intermediate=False)
        model_out.add_module(
            "layer_2",
            nn.Conv2d(
                int(self.embedding_dim / 2), int(self.embedding_dim / 2), 3, 1,
                padding=self._get_padding('SAME', (3, 3), ),
            ),
        )
        model_out.add_module(
            "layer_3",
            nn.Conv2d(
                int(self.embedding_dim / 2), int(self.embedding_dim / 4), 3, 1,
                padding=self._get_padding('SAME', (3, 3), ),
            ),
        )
        model_out.add_module(
            "upsample", nn.Upsample(scale_factor=4, mode='bilinear')
        )
        return model_in, model_intmd, model_out
    # fmt: on


def SETR_Naive_S(dataset='cityscapes'):
    if dataset.lower() == 'cityscapes':
        img_dim = 768
        num_classes = 19
    elif dataset.lower() == 'ade20k':
        img_dim = 512
        num_classes = 150
    elif dataset.lower() == 'pascal':
        img_dim = 480
        num_classes = 59

    num_channels = 3
    patch_dim = 16
    aux_layers = None
    model = SETR_Naive(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=768,
        num_heads=12,
        num_layers=12,
        hidden_dim=3072,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )

    return aux_layers, model


def SETR_Naive_L(dataset='cityscapes'):
    if dataset.lower() == 'cityscapes':
        img_dim = 768
        num_classes = 19
    elif dataset.lower() == 'ade20k':
        img_dim = 512
        num_classes = 150
    elif dataset.lower() == 'pascal':
        img_dim = 480
        num_classes = 59

    num_channels = 3
    patch_dim = 16
    aux_layers = [10, 15, 20]
    model = SETR_Naive(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=1024,
        num_heads=16,
        num_layers=24,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )

    return aux_layers, model




def SETR_PUP_L(dataset='cityscapes'):
    if dataset.lower() == 'cityscapes':
        img_dim = 768
        num_classes = 19
    elif dataset.lower() == 'ade20k':
        img_dim = 512
        num_classes = 150
    elif dataset.lower() == 'pascal':
        img_dim = 480
        num_classes = 59

    num_channels = 3
    patch_dim = 16
    aux_layers = [10, 15, 20, 24]
    model = SETR_PUP(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=1024,
        num_heads=16,
        num_layers=24,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )

    return aux_layers, model





def SETR_MLA_L(dataset='cityscapes'):
    if dataset.lower() == 'cityscapes':
        img_dim = 768
        num_classes = 19
    elif dataset.lower() == 'ade20k':
        img_dim = 512
        num_classes = 150
    elif dataset.lower() == 'pascal':
        img_dim = 480
        num_classes = 59

    num_channels = 3
    patch_dim = 16
    aux_layers = [6, 12, 18, 24]
    model = SETR_MLA(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=1024,
        num_heads=16,
        num_layers=24,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )

    return aux_layers, model

