#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-19

from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import  _ResLayer, _Stem
from .resnetSPNet import _ConvBatchNormReLU, _ResBlock

from collections import OrderedDict



class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])

class DeepLabV2(nn.Sequential):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """


    def __init__(self, n_classes, n_blocks, atrous_rates, class_emb,preload='eval',resnet="original"):
        super(DeepLabV2, self).__init__()
        self.resnet=resnet

        if class_emb is not None:
            self.emb_size = class_emb.shape[1]
            self.class_emb = torch.transpose(class_emb, 1, 0).float().cuda()

        if resnet=="spnet":
            self.layer1= nn.Sequential(
                    OrderedDict(
                        [
                            ("conv1", _ConvBatchNormReLU(3, 64, 7, 2, 3, 1)),
                            ("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True)),
                        ]
                    )
                )

            self.layer2= _ResBlock(n_blocks[0], 64, 64, 256, 1, 1)
            self.layer3=_ResBlock(n_blocks[1], 256, 128, 512, 2, 1)
            self.layer4= _ResBlock(n_blocks[2], 512, 256, 1024, 1, 2)
            self.layer5= _ResBlock(n_blocks[3], 1024, 512, 2048, 1, 4)

        else:
            self.layer1= _Stem(64)
            self.layer2=_ResLayer(n_blocks[0], 64,256, 1, 1)
            self.layer3= _ResLayer(n_blocks[1], 256, 512,2, 1)
            self.layer4= _ResLayer(n_blocks[2], 512, 1024, 1, 2)
            self.layer5= _ResLayer(n_blocks[3], 1024, 2048,1, 4)

        if preload=='eval':
            self.aspp= _ASPP(2048,n_classes, atrous_rates)
        else:
            self.aspp= _ASPP(2048,21, atrous_rates)


    def setClassEmb(self,class_emb):
        self.class_emb=torch.transpose(class_emb,1,0).float().cuda()

    def getClassEmb(self):
        return self.class_emb



    def forward(self, x):

        h=self.layer1(x)
        h=self.layer2(h)
        h=self.layer3(h)
        h=self.layer4(h)
        h=self.layer5(h)
        h = self.aspp(h)
        if hasattr(self, 'class_emb'):
            return (torch.matmul(h.permute(0, 2, 3, 1), self.class_emb.cuda()).permute(0, 3, 1, 2))

        return h

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                m.eval()



if __name__ == "__main__":
    class_emb = torch.randn(600,21)
    model = DeepLabV2(
        n_classes=21, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24],class_emb=class_emb
    )
    model.eval()
    image = torch.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
