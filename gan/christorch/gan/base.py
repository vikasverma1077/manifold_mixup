import torch
import os
import time
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from torch.autograd import Variable
from torch import optim
from skimage.io import imsave
from christorch import util

class GAN:
    """
    Base model for GAN models
    """
    def __init__(self,
                 gen_fn,
                 disc_fn,
                 z_dim,
                 opt_g=optim.Adam,
                 opt_d=optim.Adam,
                 opt_d_args={'lr':0.0002, 'betas':(0.5, 0.999)},
                 opt_g_args={'lr':0.0002, 'betas':(0.5, 0.999)},
                 dnorm=None,
                 handlers=[],
                 scheduler_fn=None,
                 scheduler_args={},
                 use_cuda='detect'):
        assert use_cuda in [True, False, 'detect']
        if use_cuda == 'detect':
            use_cuda = True if torch.cuda.is_available() else False
        self.z_dim = z_dim
        self.dnorm = dnorm
        self.g = gen_fn
        self.d = disc_fn
        optim_g = opt_g(filter(lambda p: p.requires_grad,
                               self.g.parameters()), **opt_g_args)
        optim_d = opt_d(filter(lambda p: p.requires_grad,
                               self.d.parameters()), **opt_d_args)
        self.optim = {
            'g': optim_g,
            'd': optim_d,
        }
        self.scheduler = {}
        if scheduler_fn is not None:
            for key in self.optim:
                self.scheduler[key] = scheduler_fn(
                    self.optim[key], **scheduler_args)
        self.handlers = handlers
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.g.cuda()
            self.d.cuda()
        self.last_epoch = 0

    def _get_stats(self, dict_, mode):
        stats = OrderedDict({})
        for key in dict_.keys():
            stats[key] = np.mean(dict_[key])
        return stats

    def sample_z(self, bs, seed=None):
        """Return a sample z ~ p(z)"""
        if seed is not None:
            rnd_state = np.random.RandomState(seed)
            z = torch.from_numpy(
                rnd_state.normal(0, 1, size=(bs, self.z_dim))
            ).float()
        else:
            z = torch.from_numpy(
                np.random.normal(0, 1, size=(bs, self.z_dim))
            ).float()
        if self.use_cuda:
            z = z.cuda()
        z = Variable(z)
        return z

    def mse(self, prediction, target):
        if not hasattr(target, '__len__'):
            target = torch.ones_like(prediction)*target
            if prediction.is_cuda:
                target = target.cuda()
            target = Variable(target)
        return torch.nn.MSELoss()(prediction, target)

    def _train(self):
        self.g.train()
        self.d.train()

    def _eval(self):
        self.g.eval()
        self.d.eval()
        
    def train_on_instance(self, z, x):
        self._train()
        # Train the generator.
        self.optim['g'].zero_grad()
        fake = self.g(z)
        d_fake = self.d(fake)
        gen_loss = self.mse(d_fake, 1)
        gen_loss.backward()
        self.optim['g'].step()
        # Train the discriminator.
        self.optim['d'].zero_grad()
        d_fake = self.d(fake.detach())
        d_real = self.d(x)
        d_loss = self.mse(d_real, 1) + self.mse(d_fake, 0)
        d_loss.backward()
        self.optim['d'].step()
        losses = {
            'g_loss': gen_loss.data.item(),
            'd_loss': d_loss.data.item()
        }
        outputs = {
            'x': x.detach(),
            'gz': fake.detach(),
        }
        return losses, outputs

    def sample(self, bs, seed=None):
        """Return a sample G(z)"""
        self._eval()
        with torch.no_grad():
            z_batch = self.sample_z(bs, seed=seed)
            gz = self.g(z_batch)
        return gz

    def visualise(self, gz_batch, out_file):
        """Save samples g(z) to disk"""
        h, w = gz_batch.shape[-2], gz_batch.shape[-1]
        vis_dim = int(np.floor(np.sqrt(gz_batch.shape[0])))
        vis = np.zeros((vis_dim*h, vis_dim*w, 3))
        c = 0
        for i in range(vis_dim):
            for j in range(vis_dim):
                vis[i*h:(i+1)*h, j*w:(j+1)*w, :] = util.convert_to_rgb(
                    gz_batch[c])
                c += 1
        imsave(arr=vis, fname=out_file)
    
    def prepare_batch(self, X_batch):
        X_batch = X_batch.float()
        if self.use_cuda:
            X_batch = X_batch.cuda()
        X_batch = Variable(X_batch)
        return X_batch

    def train(self,
              itr,
              epochs,
              model_dir,
              result_dir,
              append=False,
              save_every=1,
              scheduler_fn=None,
              scheduler_args={},
              verbose=True):
        for folder_name in [model_dir, result_dir]:
            if folder_name is not None and not os.path.exists(folder_name):
                os.makedirs(folder_name)
        f_mode = 'w' if not append else 'a'
        f = None
        if result_dir is not None:
            f = open("%s/results.txt" % result_dir, f_mode)
        for epoch in range(self.last_epoch, epochs):
            # Training
            epoch_start_time = time.time()
            if verbose:
                pbar = tqdm(total=len(itr))
            train_dict = OrderedDict({'epoch': epoch+1})
            for b, X_batch in enumerate(itr):
                X_batch = self.prepare_batch(X_batch)
                Z_batch = self.sample_z(X_batch.size()[0])
                losses, outputs = self.train_on_instance(Z_batch, X_batch,
                                                         iter=b+1)
                for key in losses:
                    this_key = 'train_%s' % key
                    if this_key not in train_dict:
                        train_dict[this_key] = []
                    train_dict[this_key].append(losses[key])
                pbar.update(1)
                pbar.set_postfix(self._get_stats(train_dict, 'train'))
                # Process handlers.
                for handler_fn in self.handlers:
                    handler_fn(losses, (Z_batch, X_batch), outputs,
                               {'epoch':epoch+1, 'iter':b+1, 'mode':'train'})
            if verbose:
                pbar.close()
            # Step learning rates.
            for key in self.scheduler:
                self.scheduler[key].step()
            all_dict = train_dict
            for key in all_dict:
                all_dict[key] = np.mean(all_dict[key])
            for key in self.optim:
                all_dict["lr_%s" % key] = \
                    self.optim[key].state_dict()['param_groups'][0]['lr']
            all_dict['time'] = \
                time.time() - epoch_start_time
            str_ = ",".join([str(all_dict[key]) for key in all_dict])
            print(str_)
            if f is not None:
                if (epoch+1) == 1 and not append:
                    # If we're not resuming, then write the header.
                    f.write(",".join(all_dict.keys()) + "\n")
                f.write(str_ + "\n")
                f.flush()
            if (epoch+1) % save_every == 0 and model_dir is not None:
                self.save(filename="%s/%i.pkl" % (model_dir, epoch+1),
                          epoch=epoch+1)
            # Save some visualisations. We fix the z for this one
            # so we can monitor the evolution over several epochs.
            bs = itr.batch_size
            gz_batch = self.sample(bs, seed=42).data.cpu().numpy()
            self.visualise(gz_batch,
                           out_file="%s/samplef_%i.png" % (result_dir, epoch+1))
            
        if f is not None:
            f.close()

    def save(self, filename, epoch, legacy=False):
        if legacy:
            torch.save(
                (self.g.state_dict(),
                 self.d.state_dict()),
                filename)
        else:
            dd = {}
            dd['g'] = self.g.state_dict()
            dd['d'] = self.d.state_dict()
            for key in self.optim:
                dd['optim_' + key] = self.optim[key].state_dict()
            dd['epoch'] = epoch
            torch.save(dd, filename)

    def load(self, filename, legacy=False, ignore_d=False):
        """
        ignore_d: if `True`, then don't load in the
          discriminator.
        """
        if not self.use_cuda:
            map_location = lambda storage, loc: storage
        else:
            map_location = None
        if legacy:
            g, d = torch.load(filename,
                              map_location=map_location)
            self.g.load_state_dict(g)
            if not ignore_d:
                self.d.load_state_dict(d)
        else:
            dd = torch.load(filename,
                            map_location=map_location)
            self.g.load_state_dict(dd['g'])
            if not ignore_d:
                self.d.load_state_dict(dd['d'])
            for key in self.optim:
                if ignore_d and key == 'd':
                    continue
                self.optim[key].load_state_dict(dd['optim_'+key])
            self.last_epoch = dd['epoch']
            
