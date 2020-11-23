#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-03
# Modified by: Subhabrata Choudhury


from __future__ import absolute_import, division, print_function
import sys
import json
import os.path as osp

import os
import click

import numpy as np
import torch

import torch.nn.functional as F
import yaml
from addict import Dict

from tqdm import tqdm

from libs.datasets import get_dataset
from libs.models import DeepLabV2_ResNet101_MSC
from libs.utils import scores, scores_gzsl
import pickle
import re
import timeit

from torch.nn.parallel import DistributedDataParallel
import torch.distributed as distributed
import random
import math

from torch.utils.data.sampler import Sampler
from collections import OrderedDict



class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

os.environ["LC_ALL"]="en_US.utf8"
os.environ["LANG"]="en_US.utf8"

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
    
    return rank,world_size,device_id,device


@click.command()
@click.option("-c", "--config", type=str, required=True)
@click.option("--embedding", default='fastnvec')
@click.option("-m", "--model-path", type=str, required=True)
@click.option("-r", "--run", type=str, required=True)
@click.option("--imagedataset", default='cocostuff')
@click.option("--local_rank", type=int, default=0)
@click.option("--resnet",type=str,default='original')
@click.option("--bkg/--no-nkg", default=False)
def main(config, embedding,  model_path, run, imagedataset, local_rank,resnet,bkg):

    rank,world_size,device_id,device = setup(local_rank) 
    print("Local rank: {} Rank: {} World Size: {} Device_id: {} Device: {}".format(local_rank,rank,world_size,device_id,device))
    pth_extn = '.pth.tar'

    # Configuration
    CONFIG = Dict(yaml.load(open(config)))

    
    datadir = os.path.join('data/datasets', imagedataset)
    print("Split dir: ", datadir)
    savedir = osp.dirname(model_path)
    epoch = re.findall("checkpoint_(.*)\."+pth_extn[1:], osp.basename(model_path))[-1]



    if run == 'zlss' or run == 'flss':
        val = np.load(datadir + '/split/test_list.npy')
        visible_classes = np.load(datadir + '/split/novel_cls.npy')
        if bkg:
            visible_classes = np.asarray(
                np.concatenate([np.array([0]), visible_classes]), dtype=int)
    elif run == 'gzlss' or run == 'gflss':
        val = np.load(datadir + '/split/test_list.npy')

        vals_cls = np.asarray(np.concatenate([np.load(datadir+'/split/seen_cls.npy'), np.load(datadir+'/split/val_cls.npy')]), dtype=int)

        if bkg:
            vals_cls = np.asarray(
                np.concatenate([np.array([0]), vals_cls]), dtype=int)
        valu_cls = np.load(datadir + '/split/novel_cls.npy')
        visible_classes = np.concatenate([vals_cls, valu_cls])
    else:
        print("invalid run ", run)
        sys.exit()


    
    
    cls_map = np.array([255]*256)
    for i,n in enumerate(visible_classes):
        cls_map[n] = i


    if run == 'gzlss' or run == 'gflss':

        novel_cls_map = np.array([255]*256)
        for i,n in enumerate(list(valu_cls)):
            novel_cls_map[cls_map[n]] = i

        seen_cls_map = np.array([255]*256)
        for i,n in enumerate(list(vals_cls)):
            seen_cls_map[cls_map[n]] = i

    all_labels  = np.genfromtxt(datadir+'/labels_2.txt', delimiter='\t', usecols=1, dtype='str')

    print("Visible Classes: ", visible_classes)
    
    
    # Dataset 
    dataset = get_dataset(CONFIG.DATASET)(train=None, test=val,
        root=CONFIG.ROOT,
        split=CONFIG.SPLIT.TEST,
        base_size=CONFIG.IMAGE.SIZE.TEST,
        mean=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        warp=CONFIG.WARP_IMAGE,
        scale=None,
        flip=False,
    )
    
    random.seed(42)
    
    if  embedding == 'word2vec':
        class_emb = pickle.load(open(datadir+'/word_vectors/word2vec.pkl', "rb"))
    elif embedding == 'fasttext':
        class_emb = pickle.load(open(datadir+'/word_vectors/fasttext.pkl', "rb"))
    elif embedding == 'fastnvec':
        class_emb = np.concatenate([pickle.load(open(datadir+'/word_vectors/fasttext.pkl', "rb")), pickle.load(open(datadir+'/word_vectors/word2vec.pkl', "rb"))], axis = 1)
    else:
        print("invalid emb ", embedding)
        sys.exit() 

    class_emb = class_emb[visible_classes]
    class_emb = F.normalize(torch.tensor(class_emb), p=2, dim=1).cuda()


    print("Embedding dim: ", class_emb.shape[1])
    print("# Visible Classes: ", class_emb.shape[0])


    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.BATCH_SIZE.TEST,
        num_workers=CONFIG.NUM_WORKERS,
        shuffle=False,
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False),
	    pin_memory = True,
        drop_last=True
    )

    torch.set_grad_enabled(False)


    # Model 
    model = DeepLabV2_ResNet101_MSC(class_emb.shape[1], class_emb,resnet=resnet)



    state_dict = torch.load(model_path, map_location='cpu')
    model = DistributedDataParallel(model.to(device),device_ids=[rank])
    new_state_dict = OrderedDict()
    if resnet == 'spnet':
        for k, v in state_dict['state_dict'].items():
                name = k.replace("scale", "base")  # 'scale'->base
                name = name.replace("stages.", "")
                new_state_dict[name] = v
    else:
        new_state_dict=state_dict['state_dict']
    model.load_state_dict(new_state_dict)
    del state_dict
  
    model.eval()
    targets, outputs = [], []
    
    loader_iter = iter(loader)
    iterations = len(loader_iter)
    print("Iterations: {}".format(iterations))
    
    
    pbar = tqdm(
              loader, total=iterations, leave=False, dynamic_ncols=True,position=rank
       )
    for iteration in pbar:
        
        data,target,img_id = next(loader_iter)
        # Image
        data = data.to(device)
        # Forward propagation
        output = model(data)
        output = F.interpolate(output, size=data.shape[2:], mode="bilinear", align_corners = False)
        
        output = F.softmax(output, dim=1)
        target = cls_map[target.numpy()]
        

        remote_target=torch.tensor(target).to(device)
        if rank==0:
            remote_target=torch.zeros_like(remote_target).to(device)
            

        output = torch.argmax(output, dim=1).cpu().numpy()


        remote_output=torch.tensor(output).to(device)
        if rank==0:
            remote_output=torch.zeros_like(remote_output).to(device)

        for o, t in zip(output, target):
            outputs.append(o)
            targets.append(t)


        torch.distributed.reduce(remote_output, dst=0)
        torch.distributed.reduce(remote_target, dst=0)

        torch.distributed.barrier()
        
                   
        if rank==0:
            remote_output=remote_output.cpu().numpy()
            remote_target=remote_target.cpu().numpy()
            for o, t in zip(remote_output, remote_target):
                outputs.append(o)
                targets.append(t)
                
         
        

    if rank==0:

        if run == 'gzlss' or  run == 'gflss' :
            score, class_iou = scores_gzsl(targets, outputs, n_class=len(visible_classes), seen_cls=cls_map[vals_cls], unseen_cls=cls_map[valu_cls])
        else:
            score, class_iou = scores(targets, outputs, n_class=len(visible_classes))

        for k, v in score.items():
            print(k, v)

        score["Class IoU"] = {}
        for i in range(len(visible_classes)):
            score["Class IoU"][all_labels[visible_classes[i]]] = class_iou[i]

        name = ""
        name = model_path.replace(pth_extn,  "_" + run + ".json")

        if bkg ==True:
            with open(name.replace('.json',  '_bkg.json'), "w") as f:
                json.dump(score, f, indent=4, sort_keys=True)
        else:
            with open(name, "w") as f:
                json.dump(score, f, indent=4, sort_keys=True)
    
        print(score["Class IoU"])

    return


if __name__ == "__main__":
    print("Time Taken", timeit.timeit(main))
    exit()


