#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  _list-11-01
# Modified by: Subhabrata Choudhury

from __future__ import absolute_import, division, print_function


import inspect
import os
import random


import click
import numpy as np
import torch
import torch.distributed as distributed
import torch.nn as nn
import yaml
from addict import Dict
from libs.utils import DataManager

from libs.train import Trainer,ModelManager

os.environ["LC_ALL"] = "en_US.utf8"
os.environ["LANG"] = "en_US.utf8"
torch.autograd.set_detect_anomaly(True)


def setup(local_rank):
    # initialize the process group
    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = local_rank, torch.device(local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    # Set up random seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    return rank, world_size, device_id, device



@click.command()
@click.option("-c", "--config", type=str, required=True)
@click.option("--embedding", default='fastnvec')
@click.option("--inputmix", type=str, default='seen')
@click.option("--imagedataset", default='voc12')
@click.option("--experimentid", type=str)
@click.option("--local_rank", type=int, default=0)
@click.option("--bkg/--no-nkg", default=False)

@click.option("--batch_size", type=int, default=4)
@click.option("--lr", type=float, default=2.5e-4)
@click.option("--iter", type=int, default=2000)
@click.option("--save", type=int, default=500)
@click.option("--continue-from", type=str, default=None)


@click.option("--pretrained", is_flag=True)
@click.option("--spnetcheckpoint", is_flag=True)
@click.option("-m", "--model-path", type=str, required=False)
@click.option("--resnet", type=str, default='original')  # or spnet

@click.option("--pseudolabeling", type=int, default=-1)#OK

@click.option("--scale", type=int, default=-1)  #0 randomscale, 1 downscale, 2 upscale
@click.option("--mirroring", is_flag=True)


def main(config, embedding, inputmix, imagedataset, experimentid,
         local_rank, bkg, batch_size, lr, iter, save, continue_from, pretrained, spnetcheckpoint, model_path,resnet,
         pseudolabeling,scale,mirroring):

    rank, world_size, device_id, device = setup(local_rank)
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)

    # Configuration
    CONFIG = Dict(yaml.load(open(config), Loader=yaml.FullLoader))
    datadir = os.path.join('data/datasets', imagedataset)
    dm = DataManager(imagedataset,datadir,inputmix,embedding,device)

    # only one GPU save the logs
    if rank == 0:
        dm.generateSavepath(experimentid)
        dm.createDirectory(values, config, args)

    # CLASS LOADER
    dm.loadClasses(bkg)
    # cls_maps
    dm.loadClsMaps(bkg)
    # cls embeddings
    dm.loadClassEmbs()
    # DATA LOADING
    dm.loadData()

    # Dataset loader
    dm.loadDatasets(CONFIG, batch_size)

    dm.loadClassEmbs()

    #MODELS
    mm = ModelManager(
                 device,
                 resnet,
                 pseudolabeling,
                 spnetcheckpoint,
                 pretrained,
                 model_path,
                 CONFIG,
                 dm,
                 lr,
                 continue_from,
                 scale=scale,
                 mirroring=mirroring
                 )

    #CRITERION
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    criterion = criterion.to(device)

    #TensorBoard
    if rank == 0 :
            dm.generateTB(CONFIG.ITER_TB)

    #Trainer
    trainer = Trainer(mm,
                      criterion,
                      device,
                      save,
                      dm,
                      consistency=(scale>=0 or mirroring==True)
                      )
    #Training loop
    trainer.train(iter,lr,CONFIG.LR_DECAY,CONFIG.POLY_POWER,
                  pseudolabeling,
                  continue_from,
                  bkg
                  )


if __name__ == "__main__":
    main()