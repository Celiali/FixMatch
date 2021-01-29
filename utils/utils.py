from numpy.lib.twodim_base import mask_indices
import torch
import logging
import os
from datetime import datetime
import sys
from sklearn.metrics import confusion_matrix
import numpy as np

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262 """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262 """
    def __init__(self,):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    #
    # def __str__(self):
    #     fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    #     return fmtstr.format(**self.__dict__)


def setup_default_logging(params, string = 'Train', default_level=logging.INFO,
                          format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s"):
    output_dir = os.path.join(params.EXPERIMENT.log_path)
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger(string)

    def time_str(fmt=None):
        if fmt is None:
            fmt = '%Y-%m-%d_%H:%M:%S'
        return datetime.today().strftime(fmt)

    logging.basicConfig(  # unlike the root logger, a custom logger can’t be configured using basicConfig()
        filename=os.path.join(output_dir, f'{time_str()}.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=default_level)

    # print
    # file_handler = logging.FileHandler(filename=os.path.join(output_dir, f'{time_str()}.log'), mode='a')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(default_level)
    console_handler.setFormatter(logging.Formatter(format))
    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Calculate confusion matrix (row: true classes), precesion and recall
def save_cfmatrix(y_labeled, y_pseudo, mask, y_true_labeled, y_true_unlabeled, save_to=None, comment=None): 
    nclass = [i for i in range(10)]
    cfmatrix = {}
    cfmatrix['labled'] = confusion_matrix(y_true_labeled.cpu().detach(), y_labeled.cpu().detach(), labels=nclass)
    cfmatrix['unlabled_pre'] = confusion_matrix(y_true_unlabeled.cpu().detach(), y_pseudo.cpu().detach(), labels=nclass)
    valid_idx = np.where(mask.cpu()>0)
    if valid_idx[0].size > 0:
        cfmatrix['unlabled_after'] = confusion_matrix(y_true_unlabeled.cpu().detach()[valid_idx], y_pseudo.cpu().detach()[valid_idx], labels=nclass)
    else: 
        cfmatrix['unlabled_after'] = np.zeros((10, 10))
    for name, matrix in cfmatrix.items():
        f = open(save_to + 'cf_matrix_%s.txt'%name, 'ab')
        np.savetxt(f, matrix, fmt='%.2f', header=comment)
        f.close()

def nl_loss(y_pred, y_pseudo, q): 
    '''
    @y_pseudo: the predictions of unlabeled data with weak augmentations, one hot
    @y_pred: the output logits of unlabeled data with strong augmentations
    '''
    y_pred = torch.softmax(y_pred.detach_(), dim=-1)
    return (1 - torch.pow(torch.sum(y_pseudo * y_pred, axis=-1), q)) / q
