
from torch.autograd import Variable
import torch
import numpy as np
from utils import to_one_hot

criterion = torch.nn.CrossEntropyLoss()

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def run_test_with_mixup(cuda, C, loader,mix_rate,mix_layer,num_trials=1):
    correct = 0
    total = 0

    loss = 0.0
    softmax = torch.nn.Softmax()
    bce_loss = torch.nn.CrossEntropyLoss()#torch.nn.BCELoss()

    lam = np.array(mix_rate)
    lam = Variable(torch.from_numpy(np.array([lam]).astype('float32')).cuda())

    for i in range(0,num_trials):
        for batch_idx, (data, target) in enumerate(loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)

            output,reweighted_target = C(data, lam=lam, target=target, layer_mix=mix_layer)

            '''take original with probability lam.  First goal is to recover the target indices for the other batch.  '''

            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().numpy().sum()
            total += target.size(0)

            '''These are the original targets in a one-hot space'''
            target1_onehot = to_one_hot(target,10)

            #print lam
            #print reweighted_target[0:3]

            target2 = (reweighted_target - target1_onehot*(lam)).max(1)[1]

            #print "reweighted target should put probability", lam, "on first set of indexed-values"
            #print target[0:3]
            #print target2[0:3]

            loss += mixup_criterion(target, target2, lam)(bce_loss,output) * target.size(0)

    #t_loss /= total
    t_accuracy = 100. * correct / total
    
    average_loss = (loss / total)

    #print "Test with mixup", mix_rate, "on layer", mix_layer, ', loss: ', average_loss
    #print "Accuracy", t_accuracy

    return t_accuracy, average_loss

