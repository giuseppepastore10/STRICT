import os
import time
import json
import shutil
import os.path as osp
import inspect
import numpy as np
import pickle
from libs.datasets import get_dataset
import torch
import random
import torch.distributed as dist
import math
from tensorboardX import SummaryWriter
from torchnet.meter import MovingAverageValueMeter
import torch.nn.functional as F

class MyDistributedSampler(torch.utils.data.Sampler):
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
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, seenset, novelset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.data_index = []
        for v in seenset:
            self.data_index.append([v, 0])
        for v, i in novelset:
            self.data_index.append([v, i + 1])

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.data_index) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.data_index), generator=g).tolist()
        else:
            indices = list(range(len(self.data_index)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter([self.data_index[i] for i in indices])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class DataManager:
    def __init__(self,
                 imagedataset,
                 datadir,
                 inputmix,
                 embedding,
                 device
                 ):
        self.imagedataset=imagedataset
        self.datadir=datadir
        self.inputmix=inputmix

        self.embedding = embedding
        self.device = device



    def generateSavepath(self,experimentid):
        # name the savedir, might add logs/ before the datetime for clarity
        if experimentid is None:
            savedir = time.strftime('%Y%m%d%H%M%S')
        else:
            savedir = experimentid

        self.savepath = os.path.join('logs', self.imagedataset, savedir)
        return self.savepath

    # getter method
    def get_savepath(self):
        return self.savepath

    def generateTB(self,period):
        self.writer = SummaryWriter(self.savepath + '/runs')
        self.loss_meter = MovingAverageValueMeter(20)
        self.tb=period


    def get_writer(self):
        return self.writer

    def createDirectory(self,values, config, args):
        try:
            os.makedirs(self.savepath)
            # print("Log dir:", savepath)
        except:
            pass

        # now join the path in save_screenshot:
        if os.path.exists(self.savepath + '/libs'):
            shutil.rmtree(self.savepath + '/libs')
        shutil.copytree('./libs/', self.savepath + '/libs')
        shutil.copy2(osp.abspath(inspect.stack()[0][1]), self.savepath)
        shutil.copy2(config, self.savepath)
        args_dict = {}
        for a in args:
            args_dict[a] = values[a]
        with open(self.savepath + '/args.json', 'w') as fp:
            json.dump(args_dict, fp)

    def loadClasses(self,bkg):
        self.seen_classes = np.load(self.datadir + '/split/seen_cls.npy')   #only the seen classes

        if bkg:
            self.seen_classes = np.asarray(
                np.concatenate([np.array([0]),self.seen_classes]), dtype=int) #seen classes + bkg

        self.novel_classes = np.load(self.datadir + '/split/novel_cls.npy')
        self.all_labels = np.genfromtxt(self.datadir + '/labels_2.txt', delimiter='\t', usecols=1, dtype='str')

        self.seen_classes = np.asarray(
            np.concatenate([self.seen_classes, np.load(self.datadir + '/split/val_cls.npy')]), dtype=int)
        self.seen_novel_classes = np.concatenate([self.seen_classes, self.novel_classes])
        self.to_ignore_classes = self.novel_classes

        if self.inputmix == 'seen':
            self.visible_classes = self.seen_classes
        else:
            self.visible_classes = self.seen_novel_classes

        print("Seen classes: ")
        print(self.seen_classes)
        print("all labels: ")
        print(self.all_labels)

        return self.seen_classes, self.novel_classes, self.seen_novel_classes, self.to_ignore_classes, self.visible_classes,self.all_labels

    def get_Classes(self):
        return self.seen_classes, self.novel_classes, self.seen_novel_classes, self.to_ignore_classes, self.visible_classes, self.all_labels, self.visibility_mask

    def loadData(self):

        self.train = np.load(self.datadir + '/split/train_list.npy')

        self.novelset = []
        self.seenset = []


        if self.inputmix == 'seen':
            self.seenset = range(self.train.shape[0])
        else:
            print("inputmix is not seen")
            exit()

        return self.train,self.seenset,self.novelset

    def get_data(self):
            return self.train,self.seenset, self.novelset


    def loadDatasets(self,CONFIG, bs):
        # Sampler
        sampler = MyDistributedSampler(self.seenset, self.novelset, num_replicas=torch.distributed.get_world_size(), rank=torch.distributed.get_rank())


        self.dataset = get_dataset(CONFIG.DATASET)(train=self.train, test=None,
                                              root=CONFIG.ROOT,
                                              transform=None,
                                              split=CONFIG.SPLIT.TRAIN,
                                              base_size=513,
                                              crop_size=CONFIG.IMAGE.SIZE.TRAIN,
                                              mean=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
                                              warp=CONFIG.WARP_IMAGE,
                                              scale=(0.5, 1.5),
                                              flip=True,
                                              visibility_mask=self.visibility_mask,
                                              )
        random.seed(42)
        # DataLoader
        self.loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=bs,
            num_workers=CONFIG.NUM_WORKERS,
            # num_workers = 1,
            sampler=sampler,
            pin_memory=True

        )
        return self.dataset, self.loader


    def get_datasets(self):

        return self.dataset, self.loader


    def loadClassEmbs(self):
        # Word embeddings
        if self.embedding == 'word2vec':
            self.class_emb = pickle.load(open(self.datadir + '/word_vectors/word2vec.pkl', "rb"))
        elif self.embedding == 'fasttext':
            self.class_emb = pickle.load(open(self.datadir + '/word_vectors/fasttext.pkl', "rb"))
        elif self.embedding == 'fastnvec':
            self.class_emb = np.concatenate([pickle.load(open(self.datadir + '/word_vectors/fasttext.pkl', "rb")),
                                        pickle.load(open(self.datadir + '/word_vectors/word2vec.pkl', "rb"))], axis=1)
        else:
            print("invalid emb ", self.embedding)
            exit()
        self.class_emb = F.normalize(torch.tensor(self.class_emb), p=2, dim=1).to(self.device)
        self.seen_class_emb = self.class_emb[self.seen_classes]
        self.to_ignore_class_emb = self.class_emb[self.to_ignore_classes]

        return self.class_emb,self.to_ignore_class_emb,self.seen_class_emb


    def get_clsEmbs(self):
            return self.class_emb, self.to_ignore_class_emb, self.seen_class_emb



    def loadClsMaps(self,bkg):


        self.seen_map = np.array([-1] * 256)
        for i, n in enumerate(list(self.seen_classes)):
            self.seen_map[n] = i

        self.all_map = np.array([-1] * 256)
        for i, n in enumerate(list(self.seen_classes)):
            self.all_map[n] = i
        for i, n in enumerate(self.to_ignore_classes,len(self.seen_classes)):
            self.all_map[n] = i

        self.inverse_map = np.array([-1] * 256)
        for i,n in enumerate(self.all_map):
                self.inverse_map[n]=i



        if bkg:
            for i, n in enumerate(self.to_ignore_classes):
                self.seen_map[n] = 0



        # viene usata per sapere quali predizioni sono unseen e quali no nel calcolo della percentuale
        self.cls_map_seen = np.array([0] * 256)
        for i, n in enumerate(self.to_ignore_classes):
            self.cls_map_seen[n] = 1

        self.cls_map = None
        self.cls_map = np.array([255] * 256)
        for i, n in enumerate(self.seen_classes):
            self.cls_map[n] = i



        # VISIBILITY MASK
        self.visibility_mask = {}
        self.visibility_mask[0] = self.seen_map.copy()



        print(self.visibility_mask[0])
        return self.seen_map,self.cls_map_seen,self.cls_map


    def getClsMaps(self):
        return self.seen_map, self.cls_map_seen, self.cls_map,self.inverse_map


    def savePerIteration(self, iter_loss, optimizer, model, iteration, save):

        self.loss_meter.add(iter_loss)
        # TensorBoard
        if iteration % self.tb == 0:
            self.writer.add_scalar("train_loss", self.loss_meter.value()[0], iteration)
            for i, o in enumerate(optimizer.param_groups):
                self.writer.add_scalar("train_lr_group{}".format(i), o["lr"], iteration)

        # Save a model (short term)
        if iteration>0 and iteration % save == 0:
            print(
                "\nIteration: {} \nSaving (short term) model (iteration,state_dict,optimizer) ...\n ".format(
                    iteration))
            with open(self.savepath + '/iteration.json', 'w') as fp:
                json.dump({'iteration':iteration}, fp)
            name = "checkpoint_current.pth.tar"
            if "voc" in self.savepath or iteration % 5000 == 0:
                name="checkpoint_{}.pth.tar".format(iteration)
            torch.save(
                {
                    'iteration': iteration,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                osp.join(self.savepath, name)
            )

    def saveFinal(self, optimizer, model):


        torch.save(
            {

                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            osp.join(self.savepath, "checkpoint_final.pth.tar")
        )


