from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from christorch.util import NumpyDataset

def get_iterators(bs):
    def preproc(x):
        return ((x/255.) - 0.5) / 0.5
    data = CIFAR10("./", train=True, download=True)
    X_train = data.train_data
    train_ds = NumpyDataset(
        X=X_train,
        ys=None,
        preprocessor=preproc,
        reorder_channels=True
    )
    loader_train = DataLoader(train_ds, bs, shuffle=True)
    return loader_train

def get_data(train=True):
    data = CIFAR10("./", train=train, download=True)
    if train:
        return data.train_data
    else:
        return data.test_data

if __name__ == '__main__':
    #loader = get_iterators(32)
    #for xx in loader:
    #    assert xx.size(1) == 3

    get_data(False)
    
    import pdb
    pdb.set_trace()
