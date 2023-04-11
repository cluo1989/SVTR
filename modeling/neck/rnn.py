'''
Author: Cristiano-3 chunanluo@126.com
Date: 2023-04-11 10:45:54
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2023-04-11 17:36:47
FilePath: /SVTR/modeling/backbone/rnn.py
Description: 
'''
# coding: utf-8
import torch
from torch import nn
from modeling.backbone.svtr import (
    ConvBNLayer, Swish, Block, 
    trunc_normal_, zeros_, ones_
    )


class Im2Im(nn.Module):
    def __init__(self, in_channel, **kwargs):
        super().__init__()
        self.out_channel = in_channel

    def forward(self, x):
        return x


class Im2Seq(nn.Module):
    def __init__(self, in_channel, **kwargs):
        super().__init__()
        self.out_channel = in_channel

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.reshape(N, C, H*W)
        x = x.permute((0, 2, 1))
        return x


class EncoderWithRNN(nn.Module):
    def __init__(self, in_channel, **kwargs):
        super(EncoderWithRNN, self).__init__()
        hidden_size = kwargs.get('hidden_size', 256)
        self.out_channel = hidden_size*2
        self.lstm = nn.LSTM(in_channel, hidden_size, bidirectional=True, num_layers=2, batch_first=True)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x


class EncoderWithSVTR(nn.Module):
    def __init__(self,
        in_channel,
        dims=64,
        depth=2,
        hidden_dims=120,
        use_guide=False,
        num_heads=8,
        qkv_bias=True,
        mlp_ratio=2.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path=0.,
        qk_scale=None
        ):
        super(EncoderWithSVTR, self).__init__()
        self.depth = depth
        self.use_guide = use_guide
        self.conv1 = ConvBNLayer(in_channel, in_channel//8, padding=1, act=Swish)
        self.conv2 = ConvBNLayer(in_channel//8, hidden_dims, kernel_size=1, act=Swish)
        self.svtr_block = nn.ModuleList([
            Block(
                dim=hidden_dims,
                num_heads=num_heads,
                mixer='Global',
                HW=None,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer='Swish',
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer='nn.LayerNorm',
                epsilon=1e-5,
                prenorm=False
            ) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.conv3 = ConvBNLayer(hidden_dims, in_channel, kernel_size=1, act=Swish)
        self.conv4 = ConvBNLayer(2*in_channel, in_channel//8, padding=1, act=Swish)
        self.conv1x1 = ConvBNLayer(in_channel//8, dims, kernel_size=1, act=Swish)
        self.out_channel = dims
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward(self, x):
        if self.use_guide:
            z = x.clone()
            z.stop_gradient=True
        else:
            z = x

        h = z
        z = self.conv1(z)
        z = self.conv2(z)
        N,C,H,W = z.shape
        z = z.flatten(2).permute([0, 2, 1])
        for blk in self.svtr_block:
            z = blk(z)

        z = self.norm(z)
        z = z.reshape([-1, H, W, C]).permute([0, 3, 1, 2])
        z = self.conv3(z)
        z = torch.cat((h, z), dim=1)
        z = self.conv1x1(self.conv4(z))
        return z


class SequenceEncoder(nn.Module):
    def __init__(self, in_channel, encoder_type='rnn', **kwargs):
        super(SequenceEncoder, self).__init__()

        support_encoder_dict = {
            'reshape': Im2Seq,
            'rnn': EncoderWithRNN,
            'svtr': EncoderWithSVTR
        }
        assert encoder_type in support_encoder_dict, \
            f'{encoder_type} must in {support_encoder_dict.keys()}'

        self.encoder_reshape = Im2Seq(in_channel)
        self.out_channel = self.encoder_reshape.out_channel
        self.encoder_type = encoder_type

        if encoder_type == 'reshape':
            self.only_reshape = True
        else:
            self.encoder = support_encoder_dict[encoder_type](
                self.encoder_reshape.out_channel, **kwargs)
            self.out_channel = self.encoder.out_channel
            self.only_reshape = False

    def forward(self, x):
        if self.encoder_type != 'svtr':
            x = self.encoder_reshape(x)
            if not self.only_reshape:
                x = self.encoder(x)
            return x
        else:
            x = self.encoder(x)
            x = self.encoder_reshape(x)
            return x

if __name__ == '__main__':
    # svtrRNN = EncoderWithSVTR(56)
    # print(svtrRNN)
    print(SequenceEncoder(56, encoder_type='svtr'))
    