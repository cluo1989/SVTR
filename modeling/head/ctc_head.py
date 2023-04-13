'''
Author: Cristiano-3 chunanluo@126.com
Date: 2023-03-28 17:56:37
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2023-04-13 18:33:35
FilePath: /SVTR/modeling/backbone/ctc_head.py
Description: 
'''
# coding: utf-8
from torch import nn
from addict import Dict as AttrDict

from modeling.neck.rnn import Im2Seq, SequenceEncoder
from modeling.head.sar_head import SARHead


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

        for idx, head_name in enumerate(self.head_list):
            name = head_name
            if name == 'SARHead':
                sar_args = self.head_list[name]
                self.sar_head = eval(name)(in_channel=in_channel, out_channel=self.out_c, **sar_args)
                
            elif name == 'CTC':
                self.encoder_reshape = Im2Seq(in_channel)
                
                head_args = self.head_list[name]
                neck_args = self.head_list[name]['Neck']
                encoder_type = neck_args.pop('name')

                self.ctc_encoder = SequenceEncoder(in_channel=in_channel, encoder_type=encoder_type, **neck_args)
                self.ctc_head = eval(name)(in_channel=self.ctc_encoder.out_channel, n_class=self.out_c, **head_args)
            else:
                raise NotImplementedError(f'{name} is not supported in MultiHead yet')

    def forward(self, x, targets=None):
        ctc_encoder = self.ctc_encoder(x)
        ctc_out = self.ctc_head(ctc_encoder, targets)
        head_out = dict()
        head_out['ctc'] = ctc_out
        head_out['ctc_neck'] = ctc_encoder
        
        #return ctc_out  # infer

        # eval mode
        if not self.training:
            return ctc_out
        
        if self.gtc_head == 'sar':
            sar_out = self.sar_head(x, targets[1:])
            head_out['sar'] = sar_out
            return head_out
        else:
            return head_out


if __name__ == "__main__":
    config = AttrDict(
        head_list=AttrDict(
            CTC=AttrDict(Neck=AttrDict(name="svtr",dims=64, depth=2, hidden_dims=120, use_guide=True)),
            # SARHead=AttrDict(enc_dim=512,max_text_length=70)
        ),
        n_class=5963
    )

    multi = MultiHead(128, **config)
    print(multi)