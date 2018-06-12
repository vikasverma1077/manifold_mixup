#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.autograd import Variable, grad
from load import *
from torchvision.utils import save_image

from analytical_helper_script import run_test_with_mixup

from attacks import run_test_adversarial

C = torch.load('saved_models/176359.pt') ## Put the model path here
C.eval()

data_source_dir = '../data/cifar10/'
bs = 32
train_loader, valid_loader, unlabelled_loader, test_loader, num_classes = load_data_subset(0, bs, 2 ,'cifar10', data_source_dir, 0.0, labels_per_class = 5000)

decoder_func = nn.Sequential(
        nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=1),
        nn.LeakyReLU(0.02),
        nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=1),
        nn.LeakyReLU(0.02),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128, 64, kernel_size=5, padding=2, stride=1),
        nn.LeakyReLU(0.02),
        nn.Conv2d(64, 32, kernel_size=5, padding=2, stride=1),
        nn.LeakyReLU(0.02),
        nn.Conv2d(32, 3, kernel_size=5, padding=2, stride=1)
        )

decoder_func.cuda()

opt = torch.optim.Adam(decoder_func.parameters(), lr = 0.0001)

def denorm(inp):
    return ((inp+2.0)/4.0).clamp(0.0,1.0)

train = None

for epoch in range(0,200):

    losses = []

    for i, (train_t, target) in enumerate(train_loader):

        last_batch = train

        train = Variable(train_t.cuda())

        h_val = C.compute_h2(train)

        x_rec = decoder_func(h_val + 0.05 * Variable(torch.randn(h_val.size()).cuda()))

        loss = torch.sum((x_rec - train)**2) / train.size(0)

        losses.append(loss)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 300 == 1:
            print epoch, i
            print sum(losses)/len(losses)

            lambs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

            all_hi = Variable(torch.zeros(bs*len(lambs),3,32,32).cuda())
            all_xi = Variable(torch.zeros(bs*len(lambs),3,32,32).cuda())

            for i in range(len(lambs)):
                lamb = lambs[i]
                h_interp = C.compute_h2(last_batch)*lamb + h_val*(1-lamb)
                x_interp_h = decoder_func(h_interp)
                x_interp_x = last_batch*lamb + train*(1-lamb)
                
                #hilst.append(denorm(x_interp_h.data))
                #xilst.append(denorm(x_interp_x.data))

                for j in range(bs):
                    all_hi[j*len(lambs) + i] = denorm(x_interp_h[j])
                    all_xi[j*len(lambs) + i] = denorm(x_interp_x[j])

            all_hi = torch.stack(all_hi).data
            all_xi = torch.stack(all_xi).data

            save_image(all_hi, 'interpolation_images/lower_noise/x_interp_h.png',nrow=len(lambs))
            save_image(all_xi, 'interpolation_images/lower_noise/x_interp_x.png',nrow=len(lambs))

            #save_image(denorm(last_batch.data), 'interpolation_images/original_2.png')
            #save_image(denorm(train.data), 'interpolation_images/original.png')
            #save_image(denorm(x_interp_h.data), 'interpolation_images/x_interp_h.png')
            #save_image(denorm(x_rec.data), 'interpolation_images/rec.png')
#save_image(denorm(interp.data), 'interpolation_images/visible_interp.png')
#save_image(denorm(starting_x.data), 'interpolation_images/hidden_interp.png')


#print C(train).max(1)
#print target





