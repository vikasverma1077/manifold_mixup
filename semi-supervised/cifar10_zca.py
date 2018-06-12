'''
Created on May 5, 2018
@author: vermavik
'''
import torch
from torch.autograd import Variable
import os, errno
import numpy as np
from scipy import linalg
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms


def ZCA(data, reg=1e-6):
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S) + reg)), U.T)
    whiten = np.dot(data - mean, components.T)
    return components, mean, whiten


def compute_zca(data_aug, batch_size,workers,dataset, data_target_dir):
    import numpy as np
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
      
    if data_aug==1:
            train_transform = transforms.Compose(
                                                 [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=2), transforms.ToTensor()])
    else:
        train_transform = transforms.Compose(
                                             [transforms.ToTensor()])
    
    train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
    num_classes = 10
    temp_data = train_data.train_data.astype(float)
    temp_data = temp_data.astype(float)
    temp_data[:,:,:,0] = ((temp_data[:,:,:,0] - 125.3))/(63.0)
    temp_data[:,:,:,1] = ((temp_data[:,:,:,1] - 123.0))/(62.1)
    temp_data[:,:,:,2] = ((temp_data[:,:,:,2] - 113.9))/(66.7)
    temp_data = temp_data.reshape(temp_data.shape[0],temp_data.shape[1]*temp_data.shape[2]*temp_data.shape[3])
    components, mean, whiten = ZCA(temp_data)
    np.save('../data/cifar10/zca_components', components)
    np.save('../data/cifar10/zca_mean', mean)
    
if __name__ == '__main__':
    compute_zca(data_aug=1, batch_size=32, workers=1, dataset='cifar10', data_target_dir="../data/cifar10/")
    