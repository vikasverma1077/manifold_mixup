from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

from collections import OrderedDict
import cPickle as pickle
import math
import os
import sys
import time
import argparse
import datetime
from itertools import repeat, cycle

from utils import *
from networks.wide_resnet_manifold_mixup import *
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch Manifold Mixup Training')
parser.add_argument('--optimizer', type = str, default = 'sgd',
                        help='optimizer we are going to use. can be either adam of sgd')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--l2', default=0.0005, type=float, help='l2 decay')
parser.add_argument('--nesterov', action='store_true', help='nesterov in sgd')
parser.add_argument('--epochs', type = int, default = 10,
                        help='num of epochs')
parser.add_argument('--batch_size', default=100, type=int,
                        help='Batch size')
parser.add_argument('--mixup_sup', default=1, type=int, help='1 for mixup in supervised loss, 0 for no-mixup in supervised loss')
parser.add_argument('--mixup_usup', default=1, type=int, help='1 for mixup in unsupervised loss, 0 for no-mixup in unsupervised loss')

parser.add_argument('--mixup_sup_hidden', action='store_true', help='whether to apply mixup in hidden layer for Supervised loss')
parser.add_argument('--mixup_usup_hidden', action='store_true', help='whether to apply mixup in hidden layer for unsupervised loss')


parser.add_argument('--mixup_alpha_sup', type=float, default=0.1, help='alpha parameter for supervised mixup')
parser.add_argument('--mixup_alpha_usup', type=float, default=0.1, help='alpha parameter for unsupervised mixup')

parser.add_argument('--alpha_max', type=float, default=1.0, help='max value of alpha')
parser.add_argument('--alpha_max_at_factor', type=float, default=0.4, help='fraction of number of updates at which unsupervised loss coefficient becomes 1')

parser.add_argument('--net_type', default='WRN28_2', type=str, help='either of WRN28_2, WRN28_10')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/svhn]')
parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120, 160], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.2, 0.2, 0.2], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--exp_dir', type = str, default = 'exp',
                        help='folder where results are to be stored')
parser.add_argument('--data_dir', type = str, default = '../data/cifar10/',
                        help='data dir')
parser.add_argument('--job_id', type=str, default='')
parser.add_argument('--add_name', type=str, default='')

args = parser.parse_args()
print (args)
# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_val_acc = 0

start_epoch = 1
num_epochs = args.epochs
batch_size = args.batch_size

if args.dataset == 'cifar10':
    len_data = 4000
    num_updates = (45000/batch_size)*num_epochs ### 45000= 50000-5000
elif args.dataset == 'svhn':
    len_data = 1000
    num_updates = int((65937/batch_size)+1)*num_epochs #### 65937 = 73250- 7325
    print ('number of updates', num_updates)
alpha_max_at = int(num_updates*args.alpha_max_at_factor) ## the update where the alpha becomes 1
print (batch_size, num_updates, num_epochs )

def get_alpha_schedule(x):
  return math.exp(-5*np.power((1-x),2))*args.alpha_max
  
t = np.linspace(0.0, 1, alpha_max_at)
alpha =np.ones(num_updates)*args.alpha_max
for i in range(alpha_max_at):
    alpha[i] = get_alpha_schedule(t[i])


def onehot(k):
    """Converts a number to its one-hot or 1-of-k representation vector."""
    def encode(label):
        y = np.zeros(k)
        if label < k:
            y[label] = 1
        return y
    return encode

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

if args.dataset == 'cifar10':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, 2 ,'cifar10', data_source_dir, labels_per_class = 400, valid_labels_per_class =500)
    zca_components = np.load(args.data_dir +'zca_components.npy')
    zca_mean = np.load(args.data_dir +'zca_mean.npy')
if args.dataset == 'svhn':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, 2 ,'svhn', data_source_dir, labels_per_class = 100, valid_labels_per_class =732)

def getNetwork(args, num_classes):
    if args.net_type in ['WRN28_10', 'WRN28_2']:
        net = eval(args.net_type)(num_classes)
        file_name = str(args.net_type)
    else:
        print('Error : Network should be either Wide28_2 or WRN28_10')
        sys.exit(0)

    return net, file_name

print('| Building net type [' + args.net_type + ']...')
net, file_name = getNetwork(args, num_classes)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion_l = nn.CrossEntropyLoss()
criterion_u = nn.MSELoss()
softmax = nn.Softmax(dim =1)
m = nn.LogSoftmax(dim=1)
nll = nn.NLLLoss()

if args.optimizer== 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay = args.l2, nesterov= args.nesterov)
elif args.optimzer == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay = args.l2)

if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

result_path = os.path.join(args.exp_dir , 'out.txt')
filep = open(result_path, 'w',  buffering=0)

out_str = str(args)
filep.write(out_str + '\n')     

# Training
update_idx = -1

def train(epoch):
    net.train()
    train_loss = 0
    train_loss_sup = 0
    train_loss_usup = 0
    nll_loss =0
    correct = 0
    total = 0
    
    if args.optimizer == 'sgd':
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)
        print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, current_learning_rate))
    
    batch_idx = -1
    global update_idx
    
    def l1_loss(net):
        l1_reg = None
        for W in net.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
        return l1_reg
    
    for (inputs, targets), (u, _) in zip(cycle(trainloader), unlabelledloader):
        if inputs.shape[0]!= u.shape[0]:
            bt_size = np.minimum(inputs.shape[0], u.shape[0])
            inputs = inputs[0:bt_size]
            targets = targets[0:bt_size]
            u = u[0:bt_size]
        
        
        if args.dataset == 'cifar10':
            inputs = apply_zca(inputs, zca_mean, zca_components)
            u = apply_zca(u, zca_mean, zca_components)
        
        
        inputs_temp = inputs
        inputs_temp = inputs.cuda()
        inputs_temp = Variable(inputs_temp)
        
        batch_idx +=1
        update_idx +=1
        
        ## get the supervised loss: mixup or no-mixup based
        if args.mixup_sup:
                inputs, targets, u  = inputs.cuda(), targets.cuda(), u.cuda()
                inputs = Variable(inputs)
                targets = Variable(targets)
                u = Variable(u)
                outputs, target_a, target_b, lam = net(x=inputs, y= targets, mixup_hidden = args.mixup_sup_hidden, mixup_alpha=args.mixup_alpha_sup)
                lam = lam.data.cpu().numpy().item()
                loss_func = mixup_criterion(target_a, target_b, lam)
                loss_supervised = loss_func(criterion_l, outputs)
               
        else:
            if use_cuda:
                targets = targets.cuda()
                inputs = inputs.cuda()
                u = u.cuda()
             
            inputs = torch.autograd.Variable(inputs)
            targets = torch.autograd.Variable(targets)
            u = torch.autograd.Variable(u)
            
            outputs = net(inputs, mixup =False)
            loss_supervised = criterion_l(outputs, targets)
        
            
        ### the unsupervised loss###
        
        outputs_u = net(u, mixup = False)
        
        if args.mixup_usup:
                output, target_a, target_b, lam = net(x=u, y= outputs_u, mixup_hidden = args.mixup_usup_hidden, mixup_alpha=args.mixup_alpha_usup)
                mixedup_target = target_a*lam.expand_as(target_a) + target_b*(1-lam.expand_as(target_b))
                mixedup_target = Variable(mixedup_target.data)
                loss_unsupervised = criterion_u(output, mixedup_target)
                loss = loss_supervised + alpha[update_idx].item()*loss_unsupervised
        else:
            loss = loss_supervised
        
        l1_reg = l1_loss(net)    
        loss = loss + 0.001*l1_reg
        optimizer.zero_grad()
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.data[0]*targets.size(0)
        train_loss_sup += loss_supervised.data[0]*targets.size(0)
        train_loss_usup += loss_unsupervised.data[0]*targets.size(0)
        
        outputs = net(inputs_temp, mixup = False)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        nll_loss += nll(m(outputs), targets).data[0]*targets.size(0)
        
        outputs=net(u, mixup = False)
        
        unsup_alpha = alpha[update_idx].item()
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f  NLLLoss: %.4f Acc@1: %.3f unsup_alpha: %.6f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len_data//batch_size)+1, train_loss/total, nll_loss/total,  100.*correct/total, unsup_alpha ))
        sys.stdout.flush()
    
    filep.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f SupLoss: %.4f UsupLoss: %.4f NLLLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len_data//batch_size)+1, train_loss/total,train_loss_sup/total,train_loss_usup/total, nll_loss/total, 100.*correct/total)+'\n')    
        
best_test_acc =0.0      
def test(epoch):
    global best_val_acc
    global best_test_acc
    net.eval()
    test_loss = 0
    nll_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(validloader):
        if args.dataset == 'cifar10':
            inputs = apply_zca(inputs, zca_mean, zca_components)
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs, mixup = False)
        loss = criterion_l(outputs, targets)

        test_loss += loss.data[0]*targets.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        nll_loss += nll(m(outputs), targets).data[0]*targets.size(0)
    # Save checkpoint when best model
    acc = 100.*correct/total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f NLLLoss: %.4f Acc@1: %.2f%%" %(epoch, test_loss/total, nll_loss/total, acc))
    filep.write("\n| Validation Epoch #%d\t\t\tLoss: %.4f  NLLLoss: %.4f Acc@1: %.2f%%" %(epoch, test_loss/total, nll_loss/total, acc)+'\n')
    if acc > best_val_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
        state = {
                'net':net.module if use_cuda else net,
                'acc':acc,
                'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = args.exp_dir#'./checkpoint/'+args.dataset+os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, os.path.join(save_point, 'model.t7'))
        best_val_acc = acc
        #### get test accuracy at best valid acc##
        net.eval()
        test_loss = 0
        nll_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if args.dataset == 'cifar10':
                inputs = apply_zca(inputs, zca_mean, zca_components)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs, mixup=False)
            loss = criterion_l(outputs, targets)
    
            test_loss += loss.data[0]*targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            nll_loss += nll(m(outputs), targets).data[0]*targets.size(0)
        # Save checkpoint when best model
        acc = 100.*correct/total
        best_test_acc = acc
        best_nll_loss = nll_loss/total
        print("\n| Best Test Acc: Epoch #%d\t\t\tLoss: %.4f NLLLoss: %.4f Acc@1: %.2f%%" %(epoch, test_loss/total, best_nll_loss, acc))
        filep.write("\n| Best Test Acc: Epoch #%d\t\t\tLoss: %.4f NLLLoss: %.4f Acc@1: %.2f%%" %(epoch, test_loss/total, best_nll_loss, acc)+'\n')
        


print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(args.optimizer))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)
    print("\n| Best Test Acc:  Acc@1: %.2f%%" %(best_test_acc))
    filep.write("\n| Best Test Acc:  Acc@1: %.2f%%" %(best_test_acc)+'\n')
    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))
    
filep.close()