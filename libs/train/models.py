from libs.models import DeepLabV2_ResNet101_MSC,DeepLabV2_ResNet101_consistency
import torch

import torch.nn as nn
from collections import OrderedDict
import copy
from torch.nn.parallel import DistributedDataParallel
import numpy as np

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

        h = x
        for num, stage in enumerate(self.children()):
            if (num == 0):
                h = stage(x)
            else:
                h = h + stage(x)
        return h


def get_params(model, key,includeResnet=True):
    # For Dilated FCN

    if key == "1x":
        for m in model.named_modules():
            if "layer" in m[0] :  # 'resnet' instead of layer
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.Linear):
                    for p in m[1].parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == "10x":
        for m in model.named_modules():
            if "aspp" in m[0] or "classifier" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight
    # For conv bias in the ASPP module
    if key == "20x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].bias


class ModelManager:
    def __init__(self,
                 device,
                 resnet,
                 pseudolabeling,
                 spnetcheckpoint,
                 pretrained,
                 model_path,
                 CONFIG,
                 datamanager,
                 lr,
                 bn="BN",
                 continue_from=None,
                 mirroring=False,
                 scale=-1
                 ):

        _, _, seen_novel_classes, _, visible_classes, _, _ = datamanager.get_Classes()
        class_emb,to_ignore_class_emb, _ = datamanager.get_clsEmbs()

        #MODELS



        #pseudo label generator
        if scale==0:
            scales = 0
        elif scale==1:
            scales = [0.5,0.75]
        elif scale==2:
            scales = [1.5, 1.75]
        else:
            scales = None

        if scales is None and mirroring==False:
            model_old = DeepLabV2_ResNet101_MSC(class_emb.shape[1], class_emb[seen_novel_classes], preload='train',
                                resnet=resnet)
        else:
            model_old = DeepLabV2_ResNet101_consistency(class_emb.shape[1], class_emb[seen_novel_classes],
                                                          preload='train', resnet=resnet, scales=scales,mirroring=mirroring)

        # under training model
        if pseudolabeling>=0:
            model = DeepLabV2_ResNet101_MSC(class_emb.shape[1], class_emb[seen_novel_classes], preload='train',resnet=resnet)
        else:
            model = DeepLabV2_ResNet101_MSC(class_emb.shape[1], class_emb[visible_classes], preload='train',resnet=resnet)

        if not spnetcheckpoint:
            if pretrained == True:
                print("Loading pretrained")

                state_dict = torch.load(CONFIG.INIT_MODEL_O, map_location='cpu')
                model_old.load_state_dict(state_dict, strict=True)
                del state_dict
        model_old.base.aspp = _ASPP(2048, class_emb.shape[1], [6, 12, 18, 24])
        model.base.aspp = _ASPP(2048, class_emb.shape[1], [6, 12, 18, 24])
        if spnetcheckpoint == True:
            if model_path is None:
                model_path = CONFIG.INIT_INITIAL_CHECKPOINT
            p = torch.load(model_path, map_location='cpu')
            new_state_dict = OrderedDict()

            for k, v in p['state_dict'].items():
                name = k[7:]  # remove `module.`
                if resnet == 'spnet':
                    name = name.replace("scale", "base")  # 'scale'->base
                    name = name.replace("stages.", "")
                new_state_dict[name] = v


            model_old.load_state_dict(new_state_dict, strict=True)



        if continue_from is None:
            model.base = copy.deepcopy(model_old.base)

        if pseudolabeling >= 0:
            model.base.setClassEmb(class_emb[seen_novel_classes])
        else:
            model.base.setClassEmb(class_emb[visible_classes])

        model_old.base.setClassEmb(to_ignore_class_emb)


        self.model_old = DistributedDataParallel(model_old.to(device), device_ids=[torch.distributed.get_rank()])
        self.model = DistributedDataParallel(model.to(device), device_ids=[torch.distributed.get_rank()])

        if continue_from is not None:
            state_dict = torch.load(continue_from,map_location='cpu')
            self.model.load_state_dict(state_dict['state_dict'], strict=True)

        # Optimizer
        self.optimizer = {
            "sgd": torch.optim.SGD(
                # cf lr_mult and decay_mult in train.prototxt
                params=[
                    {
                        "params": get_params(self.model, key="1x"),
                        "lr": lr,
                        "weight_decay": CONFIG.WEIGHT_DECAY,
                    },
                    {
                        "params": get_params(self.model, key="10x"),
                        "lr": 10 * lr,
                        "weight_decay": CONFIG.WEIGHT_DECAY,
                    },
                    {
                        "params": get_params(self.model, key="20x"),
                        "lr": 20 * lr,
                        "weight_decay": 0.0,
                    }
                ],
                momentum=CONFIG.MOMENTUM,
            ),
            "adam": torch.optim.Adam(
                # cf lr_mult and decay_mult in train.prototxt
                params=[
                    {
                        "params": get_params(self.model, key="1x"),
                        "lr": lr,
                        "weight_decay": CONFIG.WEIGHT_DECAY,
                    },
                    {
                        "params": get_params(self.model, key="10x"),
                        "lr": 10 * lr,
                        "weight_decay": CONFIG.WEIGHT_DECAY,
                    },
                    {
                        "params": get_params(self.model, key="20x"),
                        "lr": 20 * lr,
                        "weight_decay": 0.0,
                    }
                ]
            )
            # Add any other optimizer
        }.get(CONFIG.OPTIMIZER)
        if continue_from is not None:
            if 'optimizer' in state_dict:
                self.optimizer.load_state_dict(state_dict['optimizer'])
            else:
                print("[continue from]  optimizer not defined")
            del state_dict


    def get_models(self):
            return self.model,self.model_old

    def get_optimizer(self):
            return self.optimizer