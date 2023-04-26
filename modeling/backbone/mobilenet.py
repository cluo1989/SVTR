'''
Author: Cristiano-3 chunanluo@126.com
Date: 2023-04-11 18:42:50
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2023-04-25 16:28:31
FilePath: /SVTR/modeling/backbone/mobilenet.py
Description: 
'''
# coding: utf-8
import torch
from torch import nn
import torch.nn.functional as F
from modeling.backbone.svtr import ConvBNLayer


class HardSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=True) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel//reduction,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel//reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.hard_sigmoid = HardSigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hard_sigmoid(x)
        x = torch.mul(identity, x)
        return x


class DepthWiseSeparable(nn.Module):
    def __init__(self,
        num_channels,
        num_filters1,
        num_filters2,
        num_groups,
        stride,
        scale,
        dw_size=3,
        padding=1,
        use_se=False
        ):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(
            in_channel=num_channels,
            out_channel=int(num_filters1*scale),
            kernel_size=dw_size,
            stride=stride,
            padding=padding,
            groups=int(num_groups*scale),
            act=HardSwish
        )

        if use_se:
            self.se = SEModule(int(num_filters1*scale))

        self.pw_conv = ConvBNLayer(
            in_channel=int(num_filters1*scale),
            kernel_size=1,
            out_channel=int(num_filters2*scale),
            stride=1,
            padding=0,
            act=HardSwish
        )

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


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
        
        self.conv1 = ConvBNLayer(
            in_channel=in_channel,
            kernel_size=3,
            out_channel=int(32 * scale),
            stride=2,
            padding=1
        )

        conv2_1 = DepthWiseSeparable(
            num_channels=int(32 * scale),
            num_filters1=32,
            num_filters2=64,
            num_groups=32,
            stride=1,
            scale=scale
        )

        self.block_list.append(conv2_1)

        conv2_2 = DepthWiseSeparable(
            num_channels=int(64 * scale),
            num_filters1=64,
            num_filters2=128,
            num_groups=64,
            stride=1,
            scale=scale
        )

        self.block_list.append(conv2_2)

        conv3_1 = DepthWiseSeparable(
            num_channels=int(128 * scale),
            num_filters1=128,
            num_filters2=128,
            num_groups=128,
            stride=1,
            scale=scale
        )

        self.block_list.append(conv3_1)

        conv3_2 = DepthWiseSeparable(
            num_channels=int(128 * scale),
            num_filters1=128,
            num_filters2=256,
            num_groups=128,
            stride=(2, 1),
            scale=scale
        )

        self.block_list.append(conv3_2)

        conv4_1 = DepthWiseSeparable(
            num_channels=int(256 * scale),
            num_filters1=256,
            num_filters2=256,
            num_groups=256,
            stride=1,
            scale=scale
        )

        self.block_list.append(conv4_1)

        conv4_2 = DepthWiseSeparable(
            num_channels=int(256 * scale),
            num_filters1=256,
            num_filters2=512,
            num_groups=256,
            stride=(2, 1),
            scale=scale
        )

        self.block_list.append(conv4_2)

        for _ in range(5): # conv5_1-conv5_5
            conv5 = DepthWiseSeparable(
                num_channels=int(512 * scale),
                num_filters1=512,
                num_filters2=512,
                num_groups=512,
                stride=1,
                dw_size=5,
                padding=2,
                scale=scale,
                use_se=False
            )
            self.block_list.append(conv5)

        conv5_6 = DepthWiseSeparable(
            num_channels=int(512 * scale),
            num_filters1=512,
            num_filters2=1024,
            num_groups=512,
            stride=(2, 1),
            dw_size=5,
            padding=2,
            scale=scale,
            use_se=True
        )

        self.block_list.append(conv5_6)

        conv6 = DepthWiseSeparable(
            num_channels=int(1024 * scale),
            num_filters1=1024,
            num_filters2=1024,
            num_groups=1024,
            stride=last_conv_stride,
            dw_size=5,
            padding=2,
            scale=scale,
            use_se=True
        )

        self.block_list.append(conv6)
        self.block_list = nn.Sequential(*self.block_list)

        if last_pool_type == "avg":
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.out_channel = int(1024*scale)

    def forward(self, inputs):
        print(30*'-', inputs.shape)
        y = self.conv1(inputs)
        y = self.block_list(y)
        y = self.pool(y)
        return y


if __name__ == '__main__':
    from torchsummary import summary
    model = MobileNetV1Enhance()
    summary(model, input_size=(3, 32, 224), batch_size=1)

    arr = torch.rand((1,3,32,224))
    out = model(arr)
    print(out.size())
