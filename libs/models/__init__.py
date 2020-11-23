from __future__ import absolute_import

from .deeplabv2 import *
from .msc import MSC
from .msc_consistency import MSC_consistency


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def DeepLabV2_ResNet101_MSC(n_classes, class_emb=None,preload='eval',resnet="original",scales = []):
    return MSC(
        base=DeepLabV2(
            n_classes=n_classes, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24], class_emb=class_emb,preload=preload,resnet=resnet
        ),
        scales=scales,
    )



def DeepLabV2_ResNet101_consistency(n_classes, class_emb=None,preload='eval',resnet="original",scales = 0,mirroring=True):
    return MSC_consistency(
        base=DeepLabV2(
            n_classes=n_classes, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24], class_emb=class_emb,preload=preload,resnet=resnet),
        scales=scales,
        mirroring=mirroring
    )

