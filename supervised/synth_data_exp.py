'''
1.  Generate random attribute set for training.  
2.  Generate attribute set for testing.  
3.  Synthesize the train and test images with a drawing library.  

'''

print "running script"

from PIL import Image, ImageDraw
import random
random.seed(42.0)
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from classifier_synth import Classifier
classifier = Classifier()
classifier.cuda()

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mixup', action='store_true', default=False)
parser.add_argument('--mixup_hidden', action='store_true', default=False)

args = parser.parse_args()
print args

print "header done"

def make_attributes(train_mode):
    xpos = random.uniform(30.0, 150.0)
    ypos = random.uniform(20.0, 50.0)
    xbot_noise = random.uniform(-30.0, 5.0)
    ychange_noise = random.uniform(-10.0,10.0)

    #was doing -80, -78, 78, 80.  -30, -20.  20, 30.  
    #xlen 0,1 vs. 99,100.  

    if train_mode:

        if random.uniform(0,1) < 0.5:
            rand_angle = random.uniform(-70.0,-50.0)
        else:
            rand_angle = random.uniform(50.0,80.0)

        if random.uniform(0,1) < 0.5:
            label = 0.0
            xlen = random.uniform(0.0,20.0)
            #rand_angle = random.uniform(-70.0,-50.0)
        else:
            label = 1.0
            xlen = random.uniform(40.0,100.0)
            #rand_angle = random.uniform(50.0,80.0)
    else:

        rand_angle = random.uniform(-30.0, 30.0)

        if random.uniform(0,1) < 0.5:
            label = 0.0
            xlen = random.uniform(0.0,20.0)
            #rand_angle = random.uniform(-30.0,-20.0)
        else:
            label = 1.0
            xlen = random.uniform(40.0,100.0)
            #rand_angle = random.uniform(-10.0,30.0)


    ylen = 200.0

    linewidth = random.randint(15,35)

    return (xpos, ypos, xbot_noise, ychange_noise, xlen, ylen, linewidth, rand_angle, label)

def make_image(attributes): 
    canvas = (256, 256)
    scale = 5
    thumb = (64,64)

    #print "new image"
    im = Image.new('RGBA', canvas, (255, 255, 255, 255))
    #print "draw image"
    draw = ImageDraw.Draw(im)

    xpos, ypos, xbot_noise, ychange_noise, xlen, ylen, linewidth, randangle, label = attributes


    draw.line([(xpos,ypos),(xpos+xlen,ypos+ychange_noise)], fill='red', width=linewidth)
    draw.line([(xpos+xlen,ypos+ychange_noise),(xpos+xlen+xbot_noise,ypos+ylen)], fill='red', width=linewidth)

    im = im.rotate(randangle)

    #print "thumbnail"
    im.thumbnail(thumb)
    #print "convert"
    im = im.convert("L")

    #print "to numpy"
    z = np.array(im) / 255.0

    z = torch.from_numpy(z.reshape((1,1,64,64)).astype('float32'))

    return z

def make_batch(train_mode, mixup, hidden_mixup, mixup_func):

    zlst = []
    target_lst = []
    for i in range(0,128):

        lamb = mixup_func()

        if mixup:

            attr1 = make_attributes(train_mode=True)
            attr2 = make_attributes(train_mode=True)

            target1 = attr1[-1]
            target2 = attr2[-1]
            target = target1*lamb + target2*(1-lamb)

            if hidden_mixup:
                attr_average = map(lambda a: int(lamb*a[0]+(1-lamb)*a[1]), zip(attr1,attr2))
                zlst.append(make_image(attr_average))
            else:
                zlst.append(lamb*make_image(attr1) + (1-lamb)*make_image(attr2))

        else: 
            attr = make_attributes(train_mode)
            target = attr[-1]
            zlst.append(make_image(attr))
        
        target_lst.append(target)

    z = torch.cat(zlst, 0)
    
    target = Variable(torch.from_numpy(np.array(target_lst).astype('float32')).cuda())
    
    return z, target

opt = torch.optim.Adam(classifier.parameters(), lr = 0.0001)

test_acc_lst = []

bce = nn.BCELoss()

if False:
    from torchvision.utils import save_image
    save_image(make_batch(True, False, False, lambda: random.uniform(0.0,1.0))[0], 'train_original.png')
    save_image(make_batch(True, True, False, lambda: random.uniform(0.0,1.0))[0], 'train_mixup_visible.png')
    save_image(make_batch(True, True, True, lambda: random.uniform(0.0,1.0))[0], 'train_mixup_latent.png')
    save_image(make_batch(False, False, False, lambda: random.uniform(0.0,1.0))[0], 'test_original.png')


for iter in range(0,1000):

    z, target = make_batch(True, args.mixup, args.mixup_hidden, lambda: random.uniform(0.0,1.0))

    classifier.train()
    c = classifier(Variable(z.cuda()))

    
    loss = bce(c, target)
    train_acc = (c > 0.5).eq(target > 0.5).cpu().double().mean()
    classifier.zero_grad()
    loss.backward()
    opt.step()

    z, target = make_batch(False, False, False, lambda: random.uniform(0,1))
    classifier.eval()
    c = classifier(Variable(z.cuda()))
    loss = bce(c, target)

    test_acc = (c > 0.5).eq(target > 0.5).cpu().double().mean()
    test_acc_lst.append(test_acc)
    if iter % 20 == 0:
        print "iteration", iter
        print "test loss", loss
        print "test acc", sum(test_acc_lst) / len(test_acc_lst)
        test_acc_lst = []



