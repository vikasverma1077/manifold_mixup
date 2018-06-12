from torch.autograd import Variable
import torch
import numpy as np

def run_test_with_mixup(cuda, C, loader,mix_rate,mix_layer):
    correct = 0
    total = 0

    loss = 0.0
    softmax = torch.nn.Softmax()
    bce_loss = torch.nn.BCELoss()


    lam = np.array(mix_rate)
    lam = Variable(torch.from_numpy(np.array([lam]).astype('float32')).cuda())

    for batch_idx, (data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        output,reweighted_target = C(data, lam=lam, target=target, layer_mix=mix_layer)
        #loss = criterion(output, target)

        # t_loss += loss.data[0]*target.size(0) # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        total += target.size(0)
        
        loss += bce_loss(softmax(output), reweighted_target)

    #t_loss /= total
    t_accuracy = 100. * correct / total
    
    average_loss = loss / total

    print "Test with mixup", mix_rate, "on layer", mix_layer, ', loss: ', average_loss
    print "Accuracy", t_accuracy
        


