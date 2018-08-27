import torch
from torch import nn
from torchvision.utils import save_image

def pp_interp(net, alpha):
    """
    Only works with model_resnet_preproc.py as your
      architecture!!!
    """
    conv2d = net.d.preproc
    deconv2d = nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1)
    deconv2d = deconv2d.cuda()
    deconv2d.weight = conv2d.weight

    gz1 = net.sample(bs=128)
    gz2 = net.sample(bs=128)

    #alpha = net.sample_lambda(gz1.size(0))
    gz_mix = alpha*gz1 + (1.-alpha)*gz2

    save_image(gz1*0.5 + 0.5, filename="gz1.png")
    save_image(gz2*0.5 + 0.5, filename="gz2.png")
    save_image(gz_mix*0.5 + 0.5, filename="gz_mix.png")

    # Ok, do the mixup in hidden space.

    gz1_h = conv2d(gz1)
    gz2_h = conv2d(gz2)
    #alpha = 0.05
    gz_mix_h = alpha*gz1_h + (1.-alpha)*gz2_h
    gz_mix_h_dec = deconv2d(gz_mix_h)
    save_image(gz_mix_h_dec*0.5 + 0.5, filename="gz_mix_h_dec.png")

    print(conv2d.weight == deconv2d.weight)

    
    import pdb
    pdb.set_trace()
    
    
