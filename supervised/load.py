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


def load_data_subset(data_aug, batch_size,workers,dataset, data_target_dir, labels_per_class=100, valid_labels_per_class = 500):
    ## copied from GibbsNet_pytorch/load.py
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
    elif dataset == 'mnist':
        pass
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
        elif dataset == 'mnist':
            hw_size = 24
            train_transform = transforms.Compose([
                                transforms.RandomCrop(hw_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                           ])
            test_transform = transforms.Compose([
                                transforms.CenterCrop(hw_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                           ])
        else:
            train_transform = transforms.Compose(
                                                 [transforms.RandomHorizontalFlip(),
                                                  transforms.RandomCrop(32, padding=2),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                                                [transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        print ('no data aug')
        if dataset == 'mnist':
            hw_size = 28
            train_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                           ])
            test_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                           ])

        else:
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
    elif dataset == 'mnist':
        train_data = datasets.MNIST(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.MNIST(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    #print ('svhn', train_data.labels.shape)
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
        # Only choose digits in n_labels
        # n = number of labels per class for training
        # n_val = number of lables per class for validation
        #print type(labels)
        #print (n_valid)
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))
        # Ensure uniform distribution of labels
        np.random.shuffle(indices)

        indices_valid = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n_valid] for i in range(n_labels)])
        indices_train = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[n_valid:n_valid+n] for i in range(n_labels)])
        indices_unlabelled = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:] for i in range(n_labels)])
        #print (indices_train.shape)
        #print (indices_valid.shape)
        #print (indices_unlabelled.shape)
        indices_train = torch.from_numpy(indices_train)
        indices_valid = torch.from_numpy(indices_valid)
        indices_unlabelled = torch.from_numpy(indices_unlabelled)
        sampler_train = SubsetRandomSampler(indices_train)
        sampler_valid = SubsetRandomSampler(indices_valid)
        sampler_unlabelled = SubsetRandomSampler(indices_unlabelled)
        return sampler_train, sampler_valid, sampler_unlabelled

    #print type(train_data.train_labels)

    # Dataloaders for MNIST
    if dataset == 'svhn':
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.labels, labels_per_class, valid_labels_per_class)
    elif dataset == 'mnist':
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.train_labels.numpy(), labels_per_class, valid_labels_per_class)
    else:
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.train_labels, labels_per_class, valid_labels_per_class)

    labelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = train_sampler,  num_workers=workers, pin_memory=True)
    validation = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = valid_sampler,  num_workers=workers, pin_memory=True)
    unlabelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = unlabelled_sampler,  num_workers=workers, pin_memory=True)
    test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return labelled, validation, unlabelled, test, num_classes


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
