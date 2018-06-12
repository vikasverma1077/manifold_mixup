#!/usr/bin/env python
import sys
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable, grad
from torchvision.utils import save_image
import os
slurm_name = os.environ["SLURM_JOB_ID"]
from utils import to_var
from torch.autograd import grad
from torch.nn.modules import NLLLoss
import random
from load import *
from resnet import *
from preact_resnet import *

from analytical_helper_script import run_test_with_mixup
from attacks import run_test_adversarial
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='cifar10', choices=['mnist','cifar10', 'cifar100'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--epochs', type=int, default=1200, metavar='E',
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=128, metavar='B',
                    help='batch size')
parser.add_argument('--labels_per_class', type=int, default=5000, metavar='NL',
                    help='labels_per_class')

parser.add_argument('--mixup', action='store_true', default=False, 
                    help='whether to use mixup or not')
parser.add_argument('--mixup_alpha', type=float, default=1.0, help='alpha parameter for mixup')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--schedule', type=int, nargs='+', default=[400, 800], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--mixup_hidden', action='store_true', default=False)

parser.add_argument('--model_type', type=str, default='PreActResNet18', choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'PreActResNet18', 'PreActResNet34', 'PreActResNet152'])
parser.add_argument('--fraction_validation', type=float, default=0.0, help = 'fraction of data to be used as validation, value can be between 0 and 1')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--initial_channels', type=int, default=64, choices=(16,64))
parser.add_argument('--nesterov', type=bool, default=True)
parser.add_argument('--run_analytical', type=bool, default=False)
parser.add_argument('--run_adversarial_attacks', type=bool, default=False)
parser.add_argument('--data_transform_type', default=1, choices = (1, 'rotate_test'),help = ' 1 for usual data augmentation and rotate_test for rotation')
parser.add_argument('--save_model', action='store_true', default=False)
parser.add_argument('--model_name', type = str, default = 'model', help = 'name of the model')
parser.add_argument('--exp_dir', type = str, default = 'temp', help = 'experiment dir where plots are saved')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
dataname = args.dataname

def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
  
if dataname=='cifar10':
    C =  eval(args.model_type)(args.mixup_hidden,args.initial_channels,10)
elif dataname=='cifar100':
    C =  eval(args.model_type)(args.mixup_hidden,args.initial_channels,100)
    
if torch.cuda.is_available():
    C = C.cuda()

if dataname=="cifar10":
    IMAGE_LENGTH = 32
    NUM_CHANNELS = 3
    data_source_dir = '../data/cifar10/'
    train_loader, valid_loader, _ , test_loader, num_classes = load_data_subset(args.data_transform_type, args.batch_size, 2 ,'cifar10', data_source_dir, args.fraction_validation, labels_per_class = args.labels_per_class)
    validation_loader = test_loader

elif dataname=="cifar100":
    IMAGE_LENGTH = 32
    NUM_CHANNELS = 3
    data_source_dir = '../data/cifar100/'
    train_loader, valid_loader, _ , test_loader, num_classes = load_data_subset(1, args.batch_size, 2 ,'cifar100', data_source_dir, args.fraction_validation, labels_per_class = args.labels_per_class)
    validation_loader = test_loader

def plot(exp_dir, train_loss_list,  train_acc_list, test_loss_list, test_acc_list):
    
    plt.plot(np.asarray(train_loss_list), label='train_loss')
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'train_loss.png' ))
    plt.clf()
    
    plt.plot(np.asarray(train_acc_list), label='train_acc')
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'train_acc.png' ))
    plt.clf()
    
    plt.plot(np.asarray(test_loss_list), label='test_loss')
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'test_loss.png' ))
    plt.clf()
    
    plt.plot(np.asarray(test_acc_list), label='test_acc')
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'test_acc.png' ))
    plt.clf()


def mixup_data(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

criterion = torch.nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
optimizer = torch.optim.SGD(C.parameters(), lr=0.001, momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov)

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

bce_loss = torch.nn.BCELoss()
softmax = torch.nn.Softmax(dim=1)

best_val_acc = 0
best_test_acc = 0


if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

for epoch in range(args.epochs):
    current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)
    train_loss = 0
    correct =0
    total =0
    C.train()
    for i, (input, target) in enumerate(train_loader):
            if args.mixup:
                lam = mixup_data(args.mixup_alpha)
                if args.cuda:
                    input, target = input.cuda(), target.cuda()
                    lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
                input, target, lam = Variable(input), Variable(target), Variable(lam)
                output, reweighted_target = C(input, lam, target)
                loss = bce_loss(softmax(output), reweighted_target)#mixup_criterion(target_a, target_b, lam)
            else:
                if args.cuda:
                    target = target.cuda()
                    input = input.cuda()
                input = torch.autograd.Variable(input)
                target = torch.autograd.Variable(target)
                output = C(input)
                loss = criterion(output, target)

            C.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]*target.size(0)
            _, pred = torch.max(output.data, 1)
            if args.mixup:
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()## TODO: target label sum of two target_a and target_b   
            else:
                correct += pred.eq(target.data.view_as(pred)).cpu().sum() 
            total += target.size(0)
    train_loss = train_loss/total
    train_loss_list.append(train_loss)
    print epoch, "======epoch========"
    print 'Train:loss= {:.3f} accuracy={:.3f}%'.format(train_loss,  100.*correct/total)
    
    C.eval()
    
    def run_from_loader(loader,loss_lst=[], acc_lst=[]):
        t_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            
            output = C(data)
            loss = criterion(output, target)

            t_loss += loss.data[0]*target.size(0) # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            total += target.size(0)

        t_loss /= total
        loss_lst.append(t_loss)
        t_accuracy = 100. * correct / total
        acc_lst.append(t_accuracy)

        return t_loss, t_accuracy


    if args.fraction_validation > 0.0:
        valid_loss, valid_accuracy = run_from_loader(valid_loader)
        out_str = 'Valid: loss={:.4f}, accuracy={:.3f}%'.format(valid_loss, valid_accuracy)
        print out_str
    test_loss, test_accuracy = run_from_loader(test_loader, test_loss_list, test_acc_list)
    out_str = 'Test: loss={:.4f}, accuracy={:.3f}%'.format(test_loss, test_accuracy)
    print out_str

    if args.fraction_validation > 0.0:
        if valid_accuracy > best_val_acc:
            best_val_acc = valid_accuracy
            best_test_acc = test_accuracy
    
        out_str = 'At best val accuracy={:.3f}%'.format(best_val_acc)
        print out_str
        out_str = 'Best Test accuracy={:.3f}%'.format(best_test_acc)
        print out_str
    else:
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
        out_str = 'Best Test accuracy={:.3f}%'.format(best_test_acc)
        print out_str
 
    if args.run_analytical:
        run_test_with_mixup(args, C, test_loader,mix_rate=0.5,mix_layer=0)
        run_test_with_mixup(args, C, test_loader,mix_rate=0.5,mix_layer=2)
    if args.run_adversarial_attacks:
        run_test_adversarial(args, C, test_loader, 10, 'fgsm', {'eps' : 0.01})
        run_test_adversarial(args, C, test_loader, 10, 'fgsm', {'eps' : 0.03})

    if args.save_model:
        torch.save(C, 'saved_models/%s.pt' % args.model_name)
        
    if epoch%2==0:
        if args.mixup:
            plot(args.exp_dir, train_loss_list,  train_acc_list, test_loss_list, test_acc_list)
        else:
            plot(args.exp_dir, train_loss_list,  train_acc_list, test_loss_list, test_acc_list)
    

