'''
-Takes a classifier model as input as well as the targets.  

-Run one step of FGSM.  

-Report test accuracy


'''

from torch.autograd import Variable, grad
import torch
import numpy as np
from resnet import to_one_hot

def to_var(x,requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def fgsm(classifier, x, loss_func,attack_params):
    epsilon = attack_params['eps']
    #x_diff = 2 * 0.025 * (to_var(torch.rand(x.size())) - 0.5)
    #x_diff = torch.clamp(x_diff, -epsilon, epsilon)
    x_adv = to_var(x.data)

    c_pre = classifier(x_adv)
    loss = loss_func(c_pre) # gan_loss(c, is_real,compute_penalty=False)
    nx_adv = x_adv + epsilon*torch.sign(grad(loss, x_adv,retain_graph=False)[0])
    x_adv = to_var(nx_adv.data)

    return x_adv

def pgd(classifier, x, loss_func,attack_params):
    epsilon = attack_params['eps']
    #x_diff = 2 * 0.025 * (to_var(torch.rand(x.size())) - 0.5)
    #x_diff = torch.clamp(x_diff, -epsilon, epsilon)
    x_adv = to_var(x.data)

    for i in range(0, attack_params['iter']):
        c_pre = classifier(x_adv)
        loss = loss_func(c_pre) # gan_loss(c, is_real,compute_penalty=False)
        nx_adv = x_adv + epsilon*torch.sign(grad(loss, x_adv,retain_graph=False)[0])
        x_adv = to_var(nx_adv.data)

    return x_adv

def run_test_adversarial(cuda, C, loader, num_classes, attack_type, attack_params):
    correct = 0
    total = 0

    loss = 0.0
    softmax = torch.nn.Softmax()
    bce_loss = torch.nn.BCELoss()

    epsilon = attack_params['eps']

    for batch_idx, (data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        if attack_type == "fgsm":
            adv_data = fgsm(C, data, lambda pred: bce_loss(softmax(pred), to_one_hot(target, num_classes)), attack_params)
        else:
            adv_data = pgd(C, data, lambda pred: bce_loss(softmax(pred), to_one_hot(target, num_classes)), attack_params)

        output = C(adv_data)
        #loss = criterion(output, target)

        # t_loss += loss.data[0]*target.size(0) # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        total += target.size(0)

    #t_loss /= total
    t_accuracy = 100. * correct / total

    print "Test with", attack_type, epsilon, "accuracy", t_accuracy

