import torch
import torch.nn as nn
from load import load_mnist_subset
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import to_one_hot, mixup_process
import random
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,2),
            nn.BatchNorm1d(2)) 

        self.h2 = nn.Sequential(
            nn.Linear(2,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,10))

        self.sm = nn.Softmax()

    def compute_h1(self,x):
        return self.h1(x)

    def compute_y(self,x,targets,mixup=False, visible_mixup=False):

        target_soft = to_one_hot(targets,10)


        if visible_mixup:
            lam = Variable(torch.from_numpy(np.array([np.random.beta(0.5,0.5)]).astype('float32')).cuda())
            x, target_soft = mixup_process(x, target_soft, lam=lam)

        h = self.h1(x)

        if mixup:
            lam = Variable(torch.from_numpy(np.array([np.random.beta(0.5,0.5)]).astype('float32')).cuda())
            h, target_soft = mixup_process(h, target_soft, lam=lam)

        y = self.sm(self.h2(h))

        return y, target_soft

net = Net()

net.cuda()

data_source_dir = '../data/mnist/'
bs = 256
bs_o = bs
train_loader, unlabelled, test_loader = load_mnist_subset(0, bs, bs, True, data_source_dir, n_labels = 10, labels_per_class = 5000)

def make_fig(h_state, labels,name="",clear=False):
    h_arr = h_state.data.cpu().numpy()
    label_arr = labels.data.cpu().numpy()
    plt.scatter(h_arr[:,0],h_arr[:,1],c=label_arr)
    if clear:
        plt.savefig('analytical_plots/scatter_plot_%s.png' % name)
        plt.clf()

opt = torch.optim.Adam(net.parameters(),lr=0.0001)

bce_loss = torch.nn.BCELoss()

def prune_by_label(inp,target):
    new_inp = []
    new_target = []

    for i in range(0,inp.size(0)):
        if target[i] < 10:
            new_inp.append(inp[i])
            new_target.append(target[i:i+1])

    return torch.cat(new_inp,0), torch.cat(new_target,0)


for epoch in range(0,30):

    for i, (inp, target) in enumerate(train_loader):

        inp, target = prune_by_label(inp,target)

        inp = Variable(inp.cuda())
        bs = inp.size(0)
        inp = inp.resize(bs, 784)
        target = Variable(target.cuda())

        h1 = net.compute_h1(inp)

        y, target_soft = net.compute_y(inp, target, mixup=True, visible_mixup=False)

        loss = bce_loss(y, target_soft)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if i == 0:
            clear = True
        else:
            clear = False

        if i % 50 == 0:
            make_fig(h1, target,clear=clear)


    print "epoch", epoch
    print loss


