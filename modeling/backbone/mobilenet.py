'''
Author: Cristiano-3 chunanluo@126.com
Date: 2023-04-11 18:42:50
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2023-04-11 18:49:15
FilePath: /SVTR/modeling/backbone/mobilenet.py
Description: 
'''
# coding: utf-8
from torch import nn
from modeling.backbone.svtr import ConvBNLayer


class MobileNetV1Enhance(nn.Module):
    def __init__(self,
        in_channel=3,
        scale=0.5,
        last_conv_stride=1,
        last_pool_type='max',
        **kwargs
        ):
        super().__init__()
        self.scale = scale
        self.block_list = []
        