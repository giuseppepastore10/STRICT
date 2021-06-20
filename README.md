# A Closer Look at Self-training for Zero-Label Semantic Segmentation
##### Giuseppe Pastore, Fabio Cermelli, Yongqin Xian, Massimiliano Mancini, Zeynep Akata, Barbara Caputo. In L2ID 2021.

This is the code for the paper: ["A Closer Look at Self-training for Zero-Label Semantic Segmentation"](https://arxiv.org/abs/2104.11692) accepted at L2ID 2021workshop in the main conference CVPR.

In this paper, we address Generalized Zero Label Semantic Segmentation task: the goal is to segment both pixels of classes seen during training as well as pixels of unseen classes. Pixels of seen and unseen classes co-occur in practice but pixels of training images containing unseen classes are commonly ignored during training.
 
What we propose in this work is to capture the latent information about unseen classes by supervising the model with self-produced pseudo-labels for the unlabeled pixels. 

Generating accurate pseudo-labels for unseen classes in semantic segmentation is difficult because the accuracy on unseen classes is often much lower than in the supervised case, so we propose a simple, robust and highly scalable self-training pipeline based on filtering out the pseudo-labels that are inconsistent across multiple augmented masks and on the iteratively updating of the pseudo-label generator as its accuracy on unseen classes increase.

We demonstrated that our model surpasses all more complex strategies in GZLSS on PascalVOC12 and COCO-stuff datasets achieving the current state of the art.

![teaser](https://raw.githubusercontent.com/giuseppepastore10/STRICT/master/teaserFig.png)

# Requirements

This is a Pytorch implementation and we use the pytorch.distributed package.

This repository uses the following libraries:
...

To facilitate your work in installing all dependencies, we provide you the requirement (requirements.txt) file.

# How to download data

In this project we used the same data and data splits of [SPNet](https://github.com/subhc/SPNet):

[COCO-Stuff](https://github.com/nightrome/cocostuff#downloads).

[PascalVOC12](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)

We provide the scripts to download them in 'data/download_\<dataset_name\>.sh'.
The script takes no inputs but use it in the target directory (where you want to download data). 

# How to perform training and test

The PascalVOC12.sh and COCO-stuff.sh filles contain the commands for the training and test phase for the two datasets necessary to replicate the experiments reported in the paper, while in the config folder the default hyperparameters are specified.

### Run train.py
However, this is the basic command to run the training:

>python -m torch.distributed.launch --nproc_per_node=\<num_GPUs\> --master_port \port\> train.py --config \<config_folder\> --experimentid \<exp_name\> --imagedataset \<dataset\>  --spnetcheckpoint --pseudolabeling \<starting_iteration\> -m \<checkpoint model\> --iter \<num_iterations\>  --scale \<scaling configuration\> --mirroring --batch_size \<batch_size\>

>--config \<config_folder\>: indicates the config folder in which hyperparameters are specified; it can be:
- config/voc12/ZLSS.yaml for PascalVOC12
- config/coco/ZLSS.yaml for COCO-stuff
>--pseudolabeling: indicates the number of training iterations after which the model starts to be finetuned with self-produced pseudolabels.
>-m/--model-path: identify the directory of the checkpoint model to be finetuned; note that it indicates also the pseudo-label generator's backbone.
>--mirroring: abilitate the mirroring transformation;
>--scale \<scaling configuration\>: abilitate the scaling transformation: 
- scaling configuration=-1: no scaling transformations;
- scaling configuration=0: random scaling;
- scaling configuration=1: down scaling;
- scaling configuration=2: up scaling;


Pretrained models

### Run test.py

>python -m torch.distributed.launch --nproc_per_node=\<num_GPUs\> --master_port \port\> eval.py --config \<config_folder\> --imagedataset \<dataset\> --model-path \<testing_model\>-r gzlss


## Cite us
If you use this repository, please consider to cite

       @misc{pastore2021closer,
             title={A Closer Look at Self-training for Zero-Label Semantic Segmentation}, 
             author={Giuseppe Pastore and Fabio Cermelli and Yongqin Xian and Massimiliano Mancini and Zeynep Akata and Barbara Caputo},
             year={2021},
             eprint={2104.11692},
             archivePrefix={arXiv},
             primaryClass={cs.CV}
       }
