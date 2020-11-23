#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2018-03-26

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MSC(nn.Module):
    """
    Multi-scale inputs
    """

    def __init__(self, base, scales=[]):
        super(MSC, self).__init__()
        self.base = base
        self.scales=scales


    def forward(self, x):
        # Original
        logits = self.base(x)

        _, _, H, W = logits.shape
        interp = lambda l: F.interpolate(
            l, size=(H, W), mode="bilinear", align_corners=False
        )

        logits_pyramid = []
        if isinstance(self.scales, list):

            for p in self.scales:
                h = F.interpolate(x, scale_factor=p, mode="bilinear", align_corners=False)
                logits_pyramid.append(self.base(h))
        else:
            scales = np.random.uniform(low=0.5, high=1.75, size=(2,)).tolist()
            for p in scales:
                h = F.interpolate(x, scale_factor=p, mode="bilinear", align_corners=False)
                logits_pyramid.append(self.base(h))
        # Scaled


        # Pixel-wise max
        logits_all = [logits] + [interp(l) for l in logits_pyramid]
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0]

        if self.training:
            return [logits] + [interp(l) for l in logits_pyramid] + [logits_max]
        else:
            return logits_max
