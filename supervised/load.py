import torch
from torchvision import datasets, transforms
from affine_transforms import Rotation, Zoom, Shear

def load_data(data_aug, batch_size,workers,dataset, data_target_dir):
    
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(dataset)
    
    if data_aug==1:
        train_transform = transforms.Compose(
                                             [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
                                            [transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        train_transform = transforms.Compose(
                                             [ transforms.ToTensor(),
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

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                         num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    
    return train_loader, test_loader, num_classes 


def load_data_subset(data_aug, batch_size,workers,dataset, data_target_dir, validation_fraction, labels_per_class=100):
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
    else:
        assert False, "Unknow dataset : {}".format(dataset)
    
    if data_aug==1:
        train_transform = transforms.Compose(
                                             [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
                                            [transforms.ToTensor(), transforms.Normalize(mean, std)])
    elif data_aug=="rotate_test":
        train_transform = transforms.Compose(
                                             [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
                                            [transforms.ToTensor(), transforms.Normalize(mean, std),Shear(2.0)])

    else:
        train_transform = transforms.Compose(
                                             [ transforms.ToTensor(),
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
    
    def get_sampler(labels, n=None, valid_percent=0.0):
        # Only choose digits in n_labels
        #print type(labels)
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))
        # Ensure uniform distribution of labels

        if n is not None:
            n_total = int(0.5 + n / (1.0 - valid_percent))
        else:
            n_total = n

        np.random.shuffle(indices)
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n_total] for i in range(n_labels)])
        #print indices.shape
        indices = torch.from_numpy(indices)
        np.random.shuffle(indices)

        if valid_percent > 0.0:
            valid_cutoff = n_total*n_labels - n*n_labels
            #print valid_cutoff
            #valid_cutoff = int(indices.shape[0]*valid_percent)
            print("n labels", n_labels)
            print("n", n)
            print("n_total", n_total)
            train_indices = indices[valid_cutoff:]
            valid_indices = indices[:valid_cutoff]
            print ("train indices", train_indices.shape)
            print ("valid indices", valid_indices.shape)
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(valid_indices)

            return train_sampler, valid_sampler
        else:

            sampler = SubsetRandomSampler(indices)
            return sampler
    
    #print type(train_data.train_labels)
    
    # Dataloaders for MNIST
    if validation_fraction > 0.0:
        train_sampler, valid_sampler = get_sampler(train_data.train_labels, labels_per_class, validation_fraction)
        valid = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler,  num_workers=workers, pin_memory=True)
    else:
        train_sampler = get_sampler(train_data.train_labels, labels_per_class, validation_fraction)
        valid = None

    labelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,  num_workers=workers, pin_memory=True)


    unlabelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=get_sampler(train_data.train_labels),  num_workers=workers, pin_memory=True)
    validation = torch.utils.data.DataLoader(test_data, batch_size=batch_size, sampler=get_sampler(test_data.test_labels), num_workers=workers, pin_memory=True)

    return labelled, valid, unlabelled, validation, num_classes
    

def load_mnist(data_aug, batch_size, test_batch_size, cuda, data_target_dir):

    if data_aug == 1:
        hw_size = 24
        transform_train = transforms.Compose([
                            transforms.RandomCrop(hw_size),                
                            transforms.ToTensor(),
                            Rotation(15),                                            
                            Zoom((0.85, 1.15)),       
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])
        transform_test = transforms.Compose([
                            transforms.CenterCrop(hw_size),                       
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])
    else:
        hw_size = 28
        transform_train = transforms.Compose([
                            transforms.ToTensor(),       
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])
        transform_test = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])
    
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}       
    
    
                
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_target_dir, train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_target_dir, train=False, transform=transform_test),
        batch_size=test_batch_size, shuffle=True, **kwargs)
    
    return train_loader, test_loader    



def load_mnist_subset(data_aug, batch_size, test_batch_size, cuda, data_target_dir, n_labels =10, labels_per_class=100):
    import numpy as np
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    #from utils import onehot

    
    
    if data_aug == 1:
        hw_size = 24
        transform_train = transforms.Compose([
                            transforms.RandomCrop(hw_size),                
                            transforms.ToTensor(),
                            Rotation(15),                                            
                            Zoom((0.85, 1.15)),       
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])
        transform_test = transforms.Compose([
                            transforms.CenterCrop(hw_size),                       
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])
    else:
        hw_size = 28
        transform_train = transforms.Compose([
                            transforms.ToTensor(),       
                            transforms.Normalize((0.5,), (0.5,))#transforms.Normalize((0.1307,), (0.3081,))
                       ])
        transform_test = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))#transforms.Normalize((0.1307,), (0.3081,))
                       ])
    
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}       
    
    
        
    mnist_train = datasets.MNIST(data_target_dir, train=True, download=True, transform=transform_train)
    mnist_valid = datasets.MNIST(data_target_dir, train=False, transform=transform_test)
    

    def get_sampler(labels, n=None):
        # Only choose digits in n_labels
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))

        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_labels)])
        #print indices
        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler
    
    
    #print type(mnist_train.train_labels)
    # Dataloaders for MNIST
    labelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, sampler=get_sampler(mnist_train.train_labels.numpy(), labels_per_class), **kwargs)
    unlabelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, sampler=get_sampler(mnist_train.train_labels.numpy()), **kwargs)
    validation = torch.utils.data.DataLoader(mnist_valid, batch_size=test_batch_size, sampler=get_sampler(mnist_valid.test_labels.numpy()) , **kwargs)

    return labelled, unlabelled, validation
