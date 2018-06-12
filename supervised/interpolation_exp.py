#!/usr/bin/env python

import torch
from torch.autograd import Variable, grad
from load import *
from torchvision.utils import save_image

from analytical_helper_script import run_test_with_mixup
from affine_transforms import Rotation, Zoom, Shear

from attacks import run_test_adversarial
import numpy as np

torch.manual_seed(123)

#C_b_10 = torch.load('saved_models/176323.pt')
#C_b_10.eval()
#visible mixup
#C_v_100 = torch.load('saved_models/cifar_100_visible.pt')
#C_v_100.eval()
#hidden mixup
#C_h_100 = torch.load('saved_models/cifar_100_hidden.pt')
#C_h_100.eval()

#C_v_10 = torch.load('saved_models/cifar_10_visible.pt')
#C_v_10.eval()

C_nm_10 = torch.load('saved_models/cifar_10_nm.pt')
C_nm_10.eval()

#C_h_10 = torch.load('saved_models/cifar_10_hidden.pt')
#C_h_10.eval()

#C_nm_100 = torch.load('saved_models/cifar_100_nm.pt')
#C_nm_100.eval()

#C_v_100_a2 = torch.load('saved_models/cifar_100_visible_a2.pt')
#C_v_100_a2.eval()

#deform_transforms = [Rotation(45)]
#data_source_dir = '/u/vermavik/data/DARC/cifar10/'
#train_loader, valid_loader, unlabelled_loader, test_loader, num_classes = load_data_subset("deform_test", 32, 2 ,'cifar100', data_source_dir, 0.0, labels_per_class = 5000,deform_transforms=deform_transforms)


#print "visible mixup"
#run_test_with_mixup(True, C_v, test_loader,mix_rate=1.0,mix_layer=0, num_trials=2)
#run_test_adversarial(True, C_v, test_loader, num_classes=100, attack_type='fgsm', attack_params={'eps':0.03})

#print "hidden mixup"
#run_test_with_mixup(True, C_h, test_loader,mix_rate=1.0,mix_layer=0, num_trials=2)
#run_test_adversarial(True, C_h, test_loader, num_classes=100, attack_type='fgsm', attack_params={'eps':0.03})

#run_test_adversarial(True, C, test_loader, num_classes=10, attack_type='pgd', attack_params={'eps':0.03, 'eps_iter' : 0.01, 'iter': 2})


def test_battery(model, dataname): 

    if dataname == "cifar10":
        data_source_dir = '../data/cifar10/'
        classes = 10
    elif dataname == "cifar100":
        data_source_dir = '../data/cifar100/'
        classes = 100

    train_loader, valid_loader, unlabelled_loader, test_loader, num_classes = load_data_subset(0, 128, 2 ,dataname, data_source_dir, 0.0, labels_per_class = 5000)

    #run_test_adversarial(True, model, test_loader, num_classes=num_classes, attack_type='pgd', attack_params={'eps':0.03, 'eps_iter': 0.01, 'iter': 200})

    interpolation_tests(test_loader, model, classes)
    #return
    #adversarial_example_tests(test_loader, model, classes)
    #interpolation_tests(test_loader, model, classes)

    return

    res_lst = []
    shear_lst = [0.5, 1.0, 2.0, 2.5, 3.0]
    for shearing in shear_lst:
        deform_transform = [Shear(shearing)]
        acc = deformation_tests(model, dataname, data_source_dir, deform_transform)
        print "shearing", shearing, acc
        #res_lst.append(acc)

    print "shearing exp"
    print shear_lst
    print res_lst

    res_lst = []
    rot_lst = [20.0, 40.0, 60.0, 80.0]
    for rotation in rot_lst:
        deform_transform = [Rotation(rotation)]
        acc = deformation_tests(model, dataname, data_source_dir, deform_transform)
        print "rot", rotation, acc
        #res_lst.append(acc)

    print "rotation exp"
    print rot_lst
    print res_lst

    res_lst = []
    zoom_lst = [0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.6, 1.8]
    for zoom in zoom_lst:
        deform_transform = [Zoom((zoom,zoom))]
        acc = deformation_tests(model, dataname, data_source_dir, deform_transform)
        print "zoom", zoom, acc
        #res_lst.append(acc)

    print "zoom exp"
    print zoom_lst
    print res_lst

def adversarial_example_tests(loader, model, classes):

    for eps in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
        run_test_adversarial(True, model, loader, num_classes=classes, attack_type='fgsm', attack_params={'eps':eps})

    #for iterations in [1,2,3]:
    #    run_test_adversarial(True, model, loader, num_classes=classes, attack_type='pgd', attack_params={'eps':0.03, 'eps_iter': 0.01, 'iter': iterations})

def interpolation_tests(loader, model, classes):

    for layer in [0,1,2]:
        loss_lst = []
        for lamb in np.arange(0.0,1.0,0.02).tolist() + [1.0]:
            #print layer, lamb
            acc, loss = run_test_with_mixup(True, model, loader, mix_rate=lamb, mix_layer=layer, num_trials=1)
            loss = loss.data.cpu().numpy().tolist()[0]
            loss_lst.append(loss)
    
        print "layer", layer, loss_lst


def deformation_tests(model, dataname, data_source_dir, deform_transform):

    train_loader, valid_loader, unlabelled_loader, test_loader, num_classes = load_data_subset("deform_test", 128, 2,dataname, data_source_dir, 0.0, labels_per_class = 5000,deform_transforms=deform_transform)

    acc, loss = run_test_with_mixup(True, model, test_loader,mix_rate=1.0,mix_layer=0, num_trials=1)

    return acc


print "Baseline CIFAR10"
test_battery(C_nm_10, 'cifar10')

#print "Visible Mixup CIFAR 10"
#test_battery(C_v_10, 'cifar10')

#print "Hidden Mixup CIFAR 10"
#test_battery(C_h_10, 'cifar10')


#print "Baseline no-mixup CIFAR100"
#test_battery(C_nm_100, 'cifar100')

#print "Visible Mixup CIFAR100"
#test_battery(C_v_100, 'cifar100')

#print "Hidden Mixup CIFAR100"
#test_battery(C_h_100, 'cifar100')

raise Exception('done')

batches = []

def denorm(inp):
    return (inp+2.0)/4.0

for i, (train, target) in enumerate(train_loader):

    print train.size()

    train = Variable(train.cuda())

    print train.min(), train.max()

    batches.append(train)

    if i > 0:
        break

print len(batches)

interp = (batches[0] + batches[1])/2.0

h1 = C.compute_h2(batches[0])
h2 = C.compute_h2(batches[1])

print h1.size()

h1_interp = 0.5*h1 + 0.5*h2

starting_x = Variable(batches[0].data, requires_grad=True)

for iteration in range(0,200):
    curr_h = C.compute_h2(starting_x)
    loss = torch.sum((h1_interp - curr_h)**2) / h1_interp.size(0)
    g = grad(loss, starting_x)[0]
    new_x = starting_x - 1.0 * g / g.norm(2)
    starting_x = Variable(new_x.data, requires_grad=True)
    print "loss", loss

print C(starting_x)[0:10]

save_image(denorm(batches[0].data), 'interpolation_images/batch1.png')
save_image(denorm(batches[1].data), 'interpolation_images/batch2.png')
save_image(denorm(interp.data), 'interpolation_images/visible_interp.png')
save_image(denorm(starting_x.data), 'interpolation_images/hidden_interp.png')


#print C(train).max(1)
#print target





