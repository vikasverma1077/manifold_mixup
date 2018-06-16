import numpy as np
import torch
from torch.autograd import Variable, grad
import imp
import os
import argparse
import glob
from mugan import MUGAN
from skimage.io import imsave

def dump_samples_to_disk(folder, how_many, bs):
    if not os.path.exists(folder):
        os.makedirs(folder)
    train_size = how_many
    for b in range( (train_size // bs) + 1 ):
        print("Generating fake samples -- iteration %i" % b)
        samples_ = net.sample(bs).data.cpu().numpy()*0.5 + 0.5
        samples_ = (samples_*255.).astype("uint8")
        if b == 0:
            samples = samples_
        else:
            samples = np.vstack((samples, samples_))
    samples = samples[0:train_size]
    assert len(samples) == train_size
    for i in range(samples.shape[0]):
        img = samples[i]
        imsave(arr=img.swapaxes(0,1).swapaxes(1,2),
               fname="%s/%i.png" % (folder,i))

def compute_fid_cifar10(use_samples=True, bs=512, use_tf=False):
    """
    Compute the FID score between the dataset and samples
    from the model.
    use_samples: compute FID using samples from our GAN. If
      this is set to `False` then we use the training set of
      CIFAR10.
    bs: batch size
    """
    import fid_score
    from iterators import cifar10
    train_size = 50000 # cifar10 train size
    if use_samples:
        # Generate a bunch of samples of the same # as the size of
        # the training set.
        samples = None
        for b in range( (train_size // bs) + 1 ):
            print("Generating fake samples -- iteration %i" % b)
            samples_ = net.sample(bs).data.cpu().numpy()*0.5 + 0.5
            samples_ = (samples_*255.).astype("uint8")
            if b == 0:
                samples = samples_
            else:
                samples = np.vstack((samples, samples_))
        samples = samples[0:train_size]
        assert len(samples) == train_size
    else:
        samples = cifar10.get_data(train=True)
        samples = samples.transpose((0,3,1,2))
    print("Loading entire test set...")
    cifar10_test = cifar10.get_data(train=False)
    if not use_tf:
        cifar10_test = cifar10_test.transpose((0, 3, 1, 2))
        score = fid_score.calculate_fid_given_imgs(
            imgs1=samples,
            imgs2=cifar10_test,
            batch_size=bs,
            cuda=True,
            dims=2048
        )
        print("FID score: %f" % score)
    else:
        from tf.fid import calculate_fid_given_imgs
        samples_tf = samples.swapaxes(1, 2).swapaxes(2, 3)
        score = calculate_fid_given_imgs(
            samples_tf,
            "tf/cifar-10-fid.npz")
        print("TF FID score: %f" % score)

def compute_inception_cifar10(use_samples=True,
                              how_many=50000,
                              bs=512,
                              seed=0,
                              dump_only=False,
                              use_tf=False):
    """
    Compute Inception score.
    """
    import inception_score
    from iterators import cifar10
    train_size = how_many # cifar10 train size
    # Generate a bunch of samples of the same # as the size of
    # the training set.
    rs = np.random.RandomState(seed)
    if use_samples:
        samples = None
        for b in range( (train_size // bs) + 1 ):
            print("Generating fake samples -- iteration %i" % b)
            samples_ = net.sample(bs, seed=rs.randint(1000000)).data.cpu().numpy()
            if b == 0:
                samples = samples_
            else:
                samples = np.vstack((samples, samples_))
        samples = samples[0:train_size]
        assert len(samples) == train_size
    else:
        samples = cifar10.get_data(train=True)
        samples = samples.transpose((0, 3, 1, 2))
        samples = ((samples/255.) - 0.5) / 0.5
    # Expects samples in range [-1, 1]
    if dump_only:
        # Don't compute scores, just dump the samples
        # to disk.
        samples_uint8 = ((samples*0.5) + 0.5)*255.
        samples_uint8 = samples_uint8.astype(np.uint8)
        np.save("%s.%i" % (args.name, seed), samples_uint8)
    else:
        if not use_tf:
            score_mu, score_std = inception_score.inception_score(
                imgs=samples,
                batch_size=64,
                resize=True,
                splits=10
            )
            print("PyTorch Inception score: %f +/- %f" % (score_mu, score_std))
        else:
            from tf.inception import get_inception_score
            # This one actually expects samples in [0,255].
            samples_tf = (((samples*0.5)+0.5)*255.).astype(np.uint8)
            samples_tf = samples_tf.swapaxes(1, 2).swapaxes(2, 3)
            samples_tf = [ samples_tf[i] for i in range(len(samples_tf)) ]
            inception_score_mean, inception_score_std = get_inception_score(samples_tf)
            print("TF bugged Inception Score: Mean = {} \tStd = {}.".format(
                inception_score_mean, inception_score_std))

    
'''
Process arguments.
'''
def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--name', type=str, default="deleteme")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--loss', type=str, default='jsgan')
    parser.add_argument('--z_dim', type=int, default=62)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--mixup', type=str, default=None)
    parser.add_argument('--mixup_ff', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--dnorm', type=float, default=0.0)
    parser.add_argument('--update_g_every', type=int, default=1)
    # Iterator returns (it_train_a, it_train_b, it_val_a, it_val_b)
    parser.add_argument('--iterator', type=str, default="iterators/mnist.py")
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--interactive', type=str, default=None)
    parser.add_argument('--network', type=str, default="networks/mnist.py")
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--save_images_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--last_epoch', type=int, default=None)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    return args

args = parse_args()
# Dynamically load network module.
net_module = imp.load_source('network', args.network)
gen_fn, disc_fn = getattr(net_module, 'get_network')(args.z_dim)
# Dynamically load iterator module.
itr_module = imp.load_source('iterator', args.iterator)
itr = getattr(itr_module, 'get_iterators')(args.batch_size)

gan_class = MUGAN
gan_kwargs = {
    'gen_fn': gen_fn,
    'disc_fn': disc_fn,
    'z_dim': args.z_dim,
    'loss': args.loss,
    'mixup': args.mixup,
    'mixup_ff': args.mixup_ff,
    'alpha': args.alpha,
    'dnorm': args.dnorm,
    'update_g_every': args.update_g_every,
    'opt_d_args': {'lr':args.lr, 'betas':(args.beta1, args.beta2)},
    'opt_g_args': {'lr':args.lr, 'betas':(args.beta1, args.beta2)},
    'use_cuda': False if args.cpu else True
}
net = gan_class(**gan_kwargs)
#net.alpha = args.alpha

if args.resume is not None:
    if args.resume == 'auto':
        # autoresume
        model_dir = "%s/%s/models" % (args.save_path, args.name)
        # List all the pkl files.
        files = glob.glob("%s/*.pkl" % model_dir)
        # Make them absolute paths.
        files = [ os.path.abspath(key) for key in files ]
        if len(files) > 0:
            # Get creation time and use that.
            latest_model = max(files, key=os.path.getctime)
            print("Auto-resume mode found latest model: %s" %
                  latest_model)
            net.load(latest_model, legacy=args.legacy)
    else:
        ignore_d = True if args.interactive is not None else False
        net.load(args.resume, legacy=args.legacy,
                 ignore_d=ignore_d)
    if args.last_epoch is not None:
        net.last_epoch = args.last_epoch
if args.interactive is not None:
    how_many = 5000*10
    if 'inception' in args.interactive:
        if args.interactive == 'inception':
            # Compute the Inception score and output
            # mean and std.
            compute_inception_cifar10(how_many=how_many)
        elif args.interactive == 'inception_tf':
            compute_inception_cifar10(how_many=how_many, use_tf=True)
        elif args.interactive == 'inception_both':
            compute_inception_cifar10(how_many=how_many)
            compute_inception_cifar10(how_many=how_many, use_tf=True)
    elif args.interactive == 'dump':
        # Dump the images to disk.
        compute_inception_cifar10(how_many=how_many,
                                  seed=0,
                                  dump_only=True)
    elif args.interactive == 'dump_to_disk':
        dump_samples_to_disk(folder="img_dump", how_many=50000, bs=512)
    elif args.interactive == 'fid_tf':
        compute_fid_cifar10(use_tf=True)
    elif args.interactive == 'free':        
        import pdb
        pdb.set_trace()
else:
    net.train(
        itr=itr,
        epochs=args.epochs,
        model_dir="%s/%s/models" % (args.save_path, args.name),
        result_dir="%s/%s" % (args.save_path, args.name),
        append=True if args.resume is not None else False,
        save_every=args.save_every
    )
