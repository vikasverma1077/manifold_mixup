import torch
from torch.autograd import Variable
import os, errno
import numpy as np
from scipy import linalg


import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from itertools import repeat, cycle

def to_var(x,requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x,requires_grad=requires_grad)

def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)


def apply_zca(data, zca_mean, zca_components):
        temp = data.numpy()
        shape = temp.shape
        temp = temp.reshape(-1, shape[1]*shape[2]*shape[3])
        temp = np.dot(temp - zca_mean, zca_components.T)
        temp = temp.reshape(-1, shape[1], shape [2], shape[3])
        data = torch.from_numpy(temp).float()
        return data
        

def to_one_hot(inp):
    y_onehot = torch.FloatTensor(inp.size(0), 10)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    
    return Variable(y_onehot.cuda(),requires_grad=False)


def load_data_subset(data_aug, batch_size,workers,dataset, data_target_dir, labels_per_class=100, valid_labels_per_class = 500):
    import numpy as np
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
      
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif dataset == 'svhn':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
    
    else:
        assert False, "Unknow dataset : {}".format(dataset)
    
    if data_aug==1:
        print ('data aug')
        if dataset == 'svhn':
            train_transform = transforms.Compose(
                                             [ transforms.RandomCrop(32, padding=2), transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                                            [transforms.ToTensor(), transforms.Normalize(mean, std)])
        else:    
            train_transform = transforms.Compose(
                                                 [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=2), transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                                                [transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        print ('no data aug')
        train_transform = transforms.Compose(
                                             [transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
                                            [transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'cifar100':
        train_data = datasets.CIFAR100(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif dataset == 'svhn':
        train_data = datasets.SVHN(data_target_dir, split='train', transform=train_transform, download=True)
        test_data = datasets.SVHN(data_target_dir, split='test', transform=test_transform, download=True)
        num_classes = 10
        
    elif dataset == 'stl10':
        train_data = datasets.STL10(data_target_dir, split='train', transform=train_transform, download=True)
        test_data = datasets.STL10(data_target_dir, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'imagenet':
        assert False, 'Do not finish imagenet code'
    else:
        assert False, 'Do not support dataset : {}'.format(dataset)

        
    n_labels = num_classes
    
    def get_sampler(labels, n=None, n_valid= None):
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))
        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices_valid = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n_valid] for i in range(n_labels)])
        indices_train = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[n_valid:n_valid+n] for i in range(n_labels)])
        indices_unlabelled = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[n_valid:] for i in range(n_labels)])
        indices_train = torch.from_numpy(indices_train)
        indices_valid = torch.from_numpy(indices_valid)
        indices_unlabelled = torch.from_numpy(indices_unlabelled)
        sampler_train = SubsetRandomSampler(indices_train)
        sampler_valid = SubsetRandomSampler(indices_valid)
        sampler_unlabelled = SubsetRandomSampler(indices_unlabelled)
        return sampler_train, sampler_valid, sampler_unlabelled
    
    if dataset == 'svhn':
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.labels, labels_per_class, valid_labels_per_class)
    else: 
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.train_labels, labels_per_class, valid_labels_per_class)
    
    labelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = train_sampler,  num_workers=workers, pin_memory=True)
    validation = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = valid_sampler,  num_workers=workers, pin_memory=True)
    unlabelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = unlabelled_sampler,  num_workers=workers, pin_memory=True)
    test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return labelled, validation, unlabelled, test, num_classes

if __name__ == '__main__':
    labelled, validation, unlabelled, test, num_classes  = load_data_subset(data_aug=1, batch_size=32,workers=1,dataset='cifar10', data_target_dir="/u/vermavik/data/DARC/cifar10", labels_per_class=100, valid_labels_per_class = 500)
    for (inputs, targets), (u, _) in zip(cycle(labelled), unlabelled):
        print input
