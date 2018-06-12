import torch
from torch.autograd import Variable
import os, errno
import numpy as np

def to_var(x,requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x,requires_grad=requires_grad)

def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)


def make_dir_if_not_exists(path):
    """Make directory if doesn't already exists"""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)



def to_one_hot(inp,num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    
    return Variable(y_onehot.cuda(),requires_grad=False)


def mixup_process(out, target_reweighted,lam):

    indices = np.random.permutation(out.size(0))
    out = out*lam.expand_as(out) + out[indices]*(1-lam.expand_as(out))
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam.expand_as(target_reweighted) + target_shuffled_onehot * (1 - lam.expand_as(target_reweighted))
    return out, target_reweighted
