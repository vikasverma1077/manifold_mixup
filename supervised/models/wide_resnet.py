### dropout has been removed in this code. original code had dropout#####
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys, os
import numpy as np
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import to_one_hot, mixup_process, get_lambda
from load_data import per_image_standardization
act = torch.nn.ReLU()



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.conv1(act(self.bn1(x)))
        out = self.conv2(act(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    
    def __init__(self, depth, widen_factor, num_classes, per_img_std= False, stride = 1):
        super(Wide_ResNet, self).__init__()
        self.num_classes = num_classes
        self.per_img_std = per_img_std
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet_v2 depth should be 6n+4'
        n = int((depth-4)/6)
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0], stride = stride)
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)
    
    """
    ## Modified WRN architecture###
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet_v2 depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor
        #self.mixup_hidden = mixup_hidden

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.bn1 = nn.BatchNorm2d(nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        #self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)
    """
    def forward(self, x, target= None, mixup=False, mixup_hidden=False, mixup_alpha=None):
        #print x.shape
        if self.per_img_std:
            x = per_image_standardization(x)
        
        if mixup_hidden:
            layer_mix = random.randint(0,2)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None   
        
        out = x
        
        if mixup_alpha is not None:
            lam = get_lambda(mixup_alpha)
            lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
            lam = Variable(lam)
        
        if target is not None :
            target_reweighted = to_one_hot(target,self.num_classes)
            
        if layer_mix == 0:
                out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = self.conv1(out)
        out = self.layer1(out)
        
        
        if layer_mix == 1:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = self.layer2(out)

        if layer_mix == 2:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        
        out = self.layer3(out)
        if  layer_mix == 3:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = act(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        if target is not None:
            return out, target_reweighted
        else: 
            return out
        
                  
        
def wrn28_10(num_classes=10, dropout = False, per_img_std = False, stride = 1):
    #print ('this')
    model = Wide_ResNet(depth=28, widen_factor=10, num_classes=num_classes, per_img_std = per_img_std, stride = stride)
    return model

def wrn28_2(num_classes=10, dropout = False, per_img_std = False, stride = 1):
    #print ('this')
    model = Wide_ResNet(depth =28, widen_factor =2, num_classes = num_classes, per_img_std = per_img_std, stride = stride)
    return model



if __name__ == '__main__':
    net=Wide_ResNet(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())
