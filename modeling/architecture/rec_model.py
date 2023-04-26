'''
Author: Cristiano-3 chunanluo@126.com
Date: 2023-04-11 17:15:33
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2023-04-24 16:06:31
FilePath: /SVTR/modeling/architecture/rec_model.py
Description: 
'''
# coding: utf-8
from torch import nn
from modeling.neck.rnn import SequenceEncoder, Im2Seq, Im2Im
from modeling.backbone.svtr import SVTRNet
from modeling.backbone.mobilenet import MobileNetV1Enhance
from modeling.head.ctc_head import CTC, MultiHead

backbone_dict = {'SVTRNet':SVTRNet, 'MobileNetV1Enhance':MobileNetV1Enhance}
neck_dict = {'RNN':SequenceEncoder, 'Im2Seq':Im2Seq, 'None':Im2Im}
head_dict = {'CTC':CTC, 'Multi':MultiHead}


class RecModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert 'in_channel' in config, 'in_channel must in model config'

        backbone_type = config.BACKBONE.pop('name')
        assert backbone_type in backbone_dict, f'backbone type must in {backbone_dict.keys()}'
        self.backbone = backbone_dict[backbone_type](config.in_channel, **config.BACKBONE)
        
        neck_type = config.NECK.pop('name')
        assert neck_type in neck_dict, f'neck type must in {neck_dict.keys()}'
        self.neck = neck_dict[neck_type](self.backbone.out_channel, **config.NECK)

        head_type = config.HEAD.pop('name')
        assert head_type in head_dict, f'head type must in {head_dict.keys()}'
        self.head = head_dict[head_type](self.neck.out_channel, **config.HEAD)

        self.name = f'RecModel_{backbone_type}_{neck_type}_{head_type}'

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    from addict import Dict as AttrDict

    config = AttrDict(
        in_channel=3,
        # backbone=AttrDict(type='MobileNetV1Enhance', scale=0.5, last_conv_stride=[1, 2], last_pool_type='avg'),
        backbone=AttrDict(type='SVTRNet', image_size=[32, 224]),
        neck=AttrDict(type='None'),
        head=AttrDict(
            type='Multi', 
            head_list=AttrDict(
                CTC=AttrDict(Neck=AttrDict(name="svtr", dims=64, depth=2, hidden_dims=120, use_guide=True)),
                SARHead=AttrDict(enc_dim=512,max_text_length=70)
            ),
            n_class=6625)
    )
    model = RecModel(config)
    print(model)
    
    from torchsummary import summary
    model.eval()
    summary(model, input_size=(3, 32, 224), batch_size=1)
    