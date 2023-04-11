'''
Author: Cristiano-3 chunanluo@126.com
Date: 2023-03-28 17:56:37
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2023-04-10 18:12:55
FilePath: /SVTR/modeling/backbone/ctc_head.py
Description: 
'''
# coding: utf-8
from collections import OrderedDict

import torch
from torch import nn


class CTC(nn.Module):
    def __init__(self,
        in_channel,
        n_class,
        mid_channel=None,
        **kwargs
        ):
        super().__init__()

        if mid_channel == None:
            self.fc = nn.Linear(in_channel, n_class)
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_channel, mid_channel),
                nn.Linear(mid_channel, n_class)
                )
            
        self.n_class = n_class

    def forward(self, x, targets=None):
        return self.fc(x)


class MultiHead(nn.Module):
    def __init__(self, in_channel, **kwargs):
        super().__init__()
        self.out_c = kwargs.get('n_class')
        self.head_list = kwargs.get('head_list')
        self.gtc_head = 'sar'

        for idx, head_name  in enumerate(self.head_list):
            name = head_name
            if name == 'CTC':
                self.encoder_reshape