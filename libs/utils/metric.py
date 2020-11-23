# Originally written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


def _fast_hist(label_true, label_pred, n_class,flag=False):
    if (flag==True):
        mask = (label_true >= 0) & (label_true < n_class)
    else:
        mask = (label_true > 0) & (label_true < n_class)

    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class,flag=False):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        if(lt.size > 0):
            hist += _fast_hist(lt.flatten(), lp.flatten(), n_class,flag)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    recall_cls = dict(zip(range(n_class), acc_cls))
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))
    precision_cls = np.diag(hist) / (hist.sum(axis=0) + 1e-3)
    precision_cls = dict(zip(range(n_class), precision_cls))


    return {
        "Overall Acc": acc,
        "Mean Acc": acc_cls,
        "FreqW Acc": fwavacc,
        "Mean IoU": mean_iu
    }, cls_iu, precision_cls,recall_cls

def scores_acc(label_trues, label_preds, n_class,flag=False):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        if(lt.size > 0):
            hist += _fast_hist(lt.flatten(), lp.flatten(), n_class,flag)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls_r = np.diag(hist) / hist.sum(axis=1)
    recall_cls = dict(zip(range(n_class), acc_cls_r))
    acc_cls = np.nanmean(acc_cls_r)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))
    precision_cls = np.diag(hist) / (hist.sum(axis=0) + 1e-3)
    precision_cls = dict(zip(range(n_class), precision_cls))

    return {
        "Overall Acc": acc,
        "Mean Acc": acc_cls,
        "FreqW Acc": fwavacc,
        "Mean IoU": mean_iu,
    }, cls_iu, precision_cls,recall_cls


