#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MSC_consistency(nn.Module):
    """
    Multi-scale inputs with consitency
    """

    def __init__(self, base, scales=None,mirroring=True):
        super(MSC_consistency, self).__init__()
        self.base = base
        self.scales=scales
        self.mirroring=mirroring

    def forward(self, x):
        # Original
        logits = self.base(x)

        if self.mirroring==True:
            # Flipped
            x_flipped_1 = torch.flip(x, [3])
            logits_flipped_1 = torch.flip(self.base(x_flipped_1), [3])

        _, _, H, W = logits.shape
        interp = lambda l: F.interpolate(
            l, size=(H, W), mode="bilinear", align_corners=False
        )

        # Scaled
        logits_pyramid = []
        if self.scales is not None:
            if isinstance(self.scales, list):
                for p in self.scales:
                    h = F.interpolate(x, scale_factor=p, mode="bilinear", align_corners=False)
                    b = self.base(h)
                    logits_pyramid.append(b)
                    if self.mirroring==True:
                        # Flipped + scaled
                        h_flipped_1 = F.interpolate(x_flipped_1, scale_factor=p, mode="bilinear", align_corners=False)
                        a = torch.flip(self.base(h_flipped_1), [3])
                        logits_pyramid.append(a)
            else:
                # Random Scale
                scales = np.random.uniform(low=0.5, high=1.75, size=(2,)).tolist()
                for p in scales:
                    h = F.interpolate(x, scale_factor=p, mode="bilinear", align_corners=False)
                    b = self.base(h)
                    logits_pyramid.append(b)
                    if self.mirroring == True:
                        h_flipped_1 = F.interpolate(x_flipped_1, scale_factor=p, mode="bilinear", align_corners=False)
                        a = torch.flip(self.base(h_flipped_1), [3])
                        logits_pyramid.append(a)

        if self.mirroring==True:
            return [logits] + [logits_flipped_1]  + [interp(l) for l in logits_pyramid]
        else:
            return [logits] + [interp(l) for l in logits_pyramid]
