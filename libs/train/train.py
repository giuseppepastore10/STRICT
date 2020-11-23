import numpy as np

import torch

from tqdm import tqdm
import cv2
import torch.nn.functional as F




def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter, max_iter, power):
    if iter % lr_decay_iter or iter > max_iter:
        return None
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    optimizer.param_groups[0]["lr"] = new_lr
    optimizer.param_groups[1]["lr"] = 10 * new_lr
    optimizer.param_groups[2]["lr"] = 20 * new_lr

def resize_target(target, size):
    new_target = np.zeros((target.shape[0], size, size), np.int32)
    for i, t in enumerate(target.numpy()):
        new_target[i, ...] = cv2.resize(t, (size,) * 2, interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(new_target).long()

class Trainer:
    def __init__(self,
                 modelManager,
                 criterion,
                 device,
                 save,
                 datamanager,
                 bn="BN",
                 consistency=False,
                 ):

        self.model , self.model_old = modelManager.get_models()
        self.optimizer = modelManager.get_optimizer()
        self.device = device
        self.criterion = criterion

        self.datamanager = datamanager
        self.seen_classes, self.novel_classes, self.seen_novel_classes, self.to_ignore_classes, self.visible_classes, self.all_labels, self.visibility_mask = self.datamanager.get_Classes()
        self.class_emb, self.to_ignore_class_emb, self.seen_classes_emb = self.datamanager.get_clsEmbs()
        self.seen_map, self.cls_map_seen, self.cls_map,self.inverse_map = self.datamanager.getClsMaps()
        self.map = self.cls_map_seen
        self.map[-1] = 0
        _,self.train_loader = self.datamanager.get_datasets()
        self.train_iter = iter(self.train_loader)
        self.consistency=consistency

        self.bn = bn

        #to include into the store class
        self.save = save


    def train(self,
              max_iter,
              lr,
              lr_decay,
              poly_power,
              pseudo=-1,
              continue_from=None,
              bkg=False
              ):


        self.bkg = bkg


        self.model_old.eval()
        self.model.train()




        if self.bn == 'BN':
            self.model_old.module.base.freeze_bn()
            self.model.module.base.freeze_bn()

        for p in self.model_old.parameters():
            p.requires_grad = False


        pbar = tqdm(
            range(1, max_iter + 1),
            total=max_iter,
            leave=True,
            dynamic_ncols=True,
            position=torch.distributed.get_rank()
        )

        if continue_from is not None:
            state_dict = torch.load(continue_from,map_location='cpu')

            iteration = state_dict['iteration']
            del state_dict

            # resume iteration

            for i in range(iteration):
                self.train_loader.sampler.set_epoch(iteration)

                try:
                    data = next(self.train_iter)
                    del data
                except:
                    self.train_iter = iter(self.train_loader)
                    data= next(self.train_iter)
                    del data
                    pbar.update(1)
        else:
            iteration = 0
        print("LR: ", lr)
        for cur_iter in range(iteration,max_iter):

            poly_lr_scheduler(
                optimizer=self.optimizer,
                init_lr=lr,
                iter=cur_iter,
                lr_decay_iter=lr_decay,
                max_iter=max_iter,
                power=poly_power,
            )


            self.train_loader.sampler.set_epoch(cur_iter)
            try:
                data= next(self.train_iter)
            except:
               self.train_iter = iter(self.train_loader)
               data = next(self.train_iter)

            images, labels = data


            if pseudo>=0 and cur_iter >= pseudo:
                loss=self.forwardStep(images, labels, True)
            else:
                loss=self.forwardStep(images, labels, False)

            # TO INVERT TO HAVE A SINGLE BAR
            # collect statistics from multiple processes
            #print("CUR_ITER: ", str(cur_iter),"Rank ",str(torch.distributed.get_rank())," Loss: %3.f" % float(loss))
            pbar.update(1)
            pbar.set_postfix(loss="%.3f" % float(loss))
            pbar.set_description("CUR ITER: "+str(cur_iter)+" Rank %1.f" %torch.distributed.get_rank())
            torch.distributed.barrier()

            if torch.distributed.get_rank() == 0:
                self.datamanager.savePerIteration( loss, self.optimizer, self.model, cur_iter, self.save)





        # SAVE FINAL MODEL
        if torch.distributed.get_rank()==0:
            self.datamanager.saveFinal(self.optimizer,self.model)




    # forward step
    def forwardStep(self, images, labels, pseudo):


        if self.train_loader is None or self.optimizer is None:
            print("Train loader and/or optimizer are/is None")
            exit()



        data = images.to(self.device)
        target = labels.to(self.device)
        self.optimizer.zero_grad()


        if pseudo:
            out_, tar_, loss = self.trainPseudolabeling(data, target)
        else:
            out_, tar_, loss = self.trainNoPseudolabeling(data, target)


        loss.backward()

        # Update weights with accumulated gradients
        self.optimizer.step()

        loss = torch.tensor(float(loss)).to(self.device)
        torch.distributed.all_reduce(loss)
        torch.distributed.barrier()


        loss / torch.distributed.get_world_size()
        return loss

    #training without pseudolabeling
    def trainNoPseudolabeling(self,data, target):
        # output of the new model
        self.model.train()

        out_ = next(iter(self.model(data)))
        target = resize_target(target.cpu(), out_.size(2))
        tar_ = torch.tensor(target).to(self.device)
        loss = self.criterion.forward(out_, tar_)
        return out_,tar_, loss

    # training with pseudolabeling
    def trainPseudolabeling(self, data, target):
        # output of the new model
        out_ = next(iter(self.model(data)))
        # Propagate forward
        output = self.model_old(data)


        if self.consistency:
            size = output[0].size(2)
        else:
            size = output.size(2)

        target = resize_target(target.cpu(),size)
        target_ = torch.tensor(target).to(self.device)


        if self.consistency:
            for u, outp in enumerate(output):
                outps = F.softmax(outp, dim=1)

                res_values_i, res_i = outps.max(dim=1)
                res_i = res_i + self.seen_classes.size
                if u == 0:
                    res1 = res_i
                    mask = res1 == res_i
                else:
                    mask = (res1==res_i) * mask
                del res_i

            if self.bkg == True:
                res1[~mask] = 0
            else:
                res1[~mask] = -1

            res = res1

        else:
            res_p = F.softmax(output, dim=1)
            res_values, res = res_p.max(dim=1)
            res = res + self.seen_classes.size

        if self.bkg == True:
            tar_ = torch.where(target_ == 0, res, target_)
        else:
            tar_ = torch.where(target_ == -1, res, target_)

        loss = self.criterion.forward(out_, tar_)
        return out_, tar_, loss










