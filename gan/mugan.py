import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable, grad
from christorch.gan.base import GAN

def bce(prediction, target):
    if not hasattr(target, '__len__'):
        target = torch.ones_like(prediction)*target
        if prediction.is_cuda:
            target = target.cuda()
        target = Variable(target)
    loss = torch.nn.BCELoss()
    if prediction.is_cuda:
        loss = loss.cuda()
    return loss(prediction, target)

def jsgan_d_fake_loss(d_fake):
    return bce(d_fake, 0)

def jsgan_d_real_loss(d_real):
    return bce(d_real, 1)

def jsgan_g_loss(d_fake):
    return jsgan_d_real_loss(d_fake)

def wgan_d_fake_loss(d_fake):
    return torch.mean(d_fake)

def wgan_d_real_loss(d_real):
    return -torch.mean(d_real)

def wgan_g_loss(d_fake):
    return wgan_d_real_loss(d_fake)

def hinge_d_fake_loss(d_fake):
    return nn.ReLU()(1.0 + d_fake).mean()

def hinge_d_real_loss(d_real):
    return nn.ReLU()(1.0 - d_real).mean()

def hinge_g_loss(d_fake):
    return -torch.mean(d_fake)

class MUGAN(GAN):
    """
    JSGAN with the option of adding mixup.
    Inherits from the base class `GAN` in christorch but adds
      some extra args to init such as mixup and expontentially
      decreasing LR schedule.
    """
    def __init__(self,
                 loss='jsgan',
                 mixup=None,
                 mixup_ff=False,
                 lr_sched=None, alpha=0.2,
                 update_g_every=1.,
                 *args, **kwargs):
        """
        loss: which loss function to use? Can choose between
          'jsgan', 'wgan', and 'hinge'.
        mixup: which mixup to do. Can be None, 'pixel', or 'hidden'.
        mixup_ff: whether or not to also generate fake-fake mixes as
          well.
        lr_sched:
        alpha: mixup parameter, s.t. lambda ~ Beta(alpha,alpha)
          and lambda := min(lambda, 1-lambda).
        update_g_every: update the generator every this many
          iterations.
        """
        if loss not in ['jsgan', 'wgan', 'hinge']:
            raise Exception("uknown loss")
        if loss == 'jsgan':
            self.g_loss = jsgan_g_loss
            self.d_loss_real = jsgan_d_real_loss
            self.d_loss_fake = jsgan_d_fake_loss
        elif loss == 'wgan':
            self.g_loss = wgan_g_loss
            self.d_loss_real = wgan_d_real_loss
            self.d_loss_fake = wgan_d_fake_loss
        else:
            self.g_loss = hinge_g_loss
            self.d_loss_real = hinge_d_real_loss
            self.d_loss_fake = hinge_d_fake_loss
        if mixup not in [None, 'pixel', 'hidden', 'vh1', 'vh2']:
            raise Exception("mixup must be either None, 'pixel', or 'hidden'")
        else:
            # Backwards compatibility: if 'hidden',
            # assume it's vh1.
            if mixup == 'hidden':
                self.mixup = 'vh1'
            else:
                self.mixup = mixup
        if lr_sched not in [None, 'exp']:
            raise Exception("lr_sched must be either None or 'exp'")
        self.mixup_ff = mixup_ff
        self.lr_sched = lr_sched
        self.alpha = 0.2
        self.norm_verbosity = 0 # if == 1, then monitor fake/mix norms too
        if lr_sched is not None:
            self.scheduler = {}
            self.scheduler['g'] = optim.lr_scheduler.ExponentialLR(
                self.optim['g'], gamma=0.99)
            self.scheduler['d'] = optim.lr_scheduler.ExponentialLR(
                self.optim['d'], gamma=0.99)
        self.update_g_every = update_g_every
        super(MUGAN, self).__init__(*args, **kwargs)
        # TODO :FIX
        if self.dnorm is None:
            self.dnorm = 0.
        
    def grad_norm(self, d_out, x):
        ones = torch.ones(d_out.size())
        if self.use_cuda:
            ones = ones.cuda()
        grad_wrt_x = grad(outputs=d_out, inputs=x,
                          grad_outputs=ones,
                          create_graph=True,
                          retain_graph=True,
                          only_inputs=True)[0]
        g_norm = (grad_wrt_x.view(
            grad_wrt_x.size()[0], -1).norm(2, 1)**2).mean()
        return g_norm    
    
    def _train_on_instance(self, z, x, **kwargs):
        """Training method for when mixup=False"""
        self._train()
        x.requires_grad = True # for dnorm
        # Train the generator.
        self.optim['g'].zero_grad()
        fake = self.g(z)
        _, d_fake = self.d(fake)
        gen_loss = self.g_loss(d_fake)
        if (kwargs['iter']-1) % self.update_g_every == 0:
            gen_loss.backward()
            self.optim['g'].step()
        # Train the discriminator.
        self.optim['d'].zero_grad()
        _, d_fake = self.d(fake.detach())
        _, d_real = self.d(x)
        d_loss = self.d_loss_real(d_real) + self.d_loss_fake(d_fake)
        d_loss.backward()
        self.optim['d'].step()
        ##################################
        # Also compute the gradient norm.
        # Grad norm for D_REAL
        _, d_real = self.d(x)
        g_norm_x = self.grad_norm(d_real, x)
        if self.dnorm > 0.:
            self.optim['d'].zero_grad()
            (g_norm_x*self.dnorm).backward()
            self.optim['d'].step()
        self.optim['d'].zero_grad()
        ##################################
        losses = {
            'g_loss': gen_loss.data.item(),
            'd_loss': d_loss.data.item(),
            'd_real_norm': g_norm_x.data.item()
        }
        outputs = {
            'x': x.detach(),
            'gz': fake.detach(),
        }
        return losses, outputs

    def sample_lambda(self, bs):
        alphas = []
        for i in range(bs):
            alpha = np.random.beta(self.alpha, self.alpha)
            alpha = min(alpha, 1.-alpha)
            alphas.append(alpha)
        alphas = np.asarray(alphas).reshape((bs,1,1,1))
        alphas = torch.from_numpy(alphas).float()
        if self.use_cuda:
            alphas = alphas.cuda()
        return alphas
    
    def _train_on_instance_mixup(self, z, x, **kwargs):
        """Perform mixup in the pixel space"""
        self._train()
        x.requires_grad = True # for dnorm
        # Train the generator.
        self.optim['g'].zero_grad()
        alpha = self.sample_lambda(x.size(0))
        fake = self.g(z)
        xz = Variable(alpha*x.data + (1.-alpha)*fake.data)
        if self.mixup_ff:
            perm = torch.randperm(fake.size(0)).view(-1).long()
            fake_perm = fake[perm]
            xz_ff = Variable(alpha*fake.data + (1.-alpha)*fake_perm.data)
        _, d_fake = self.d(fake)
        gen_loss = self.g_loss(d_fake)
        if (kwargs['iter']-1) % self.update_g_every == 0:
            gen_loss.backward()
            self.optim['g'].step()
        # Train the discriminator.
        self.optim['d'].zero_grad()
        _, d_xz = self.d(xz.detach())
        _, d_real = self.d(x)
        _, d_fake = self.d(fake.detach())
        d_loss = self.d_loss_fake(d_xz) + self.d_loss_real(d_real) + \
                 self.d_loss_fake(d_fake)
        if self.mixup_ff:
            _, d_xz_ff = self.d(xz_ff.detach())
            d_loss += self.d_loss_fake(d_xz_ff)
        d_loss.backward()
        self.optim['d'].step()
        ##################################
        # Also compute the gradient norm.
        # Grad norm for D_REAL
        _, d_real = self.d(x)
        g_norm_x = self.grad_norm(d_real, x)
        if self.dnorm > 0.:
            self.optim['d'].zero_grad()
            (g_norm_x*self.dnorm).backward()
            self.optim['d'].step()
        self.optim['d'].zero_grad()
        ##################################
        losses = {
            'g_loss': gen_loss.data.item(),
            'd_loss': d_loss.data.item(),
            'd_real_norm': g_norm_x.data.item(),
        }
        outputs = {
            'x': x.detach(),
            'gz': fake.detach(),
        }
        return losses, outputs

    def _train_on_instance_mixup_hidden(self, z, x, **kwargs):
        """Perform mixup in the hidden states"""
        self._train()
        x.requires_grad = True # for dnorm
        # Train the generator.
        self.optim['g'].zero_grad()
        fake = self.g(z)
        _, d_fake = self.d(fake)
        #gen_loss = self.bce(d_fake, 1)
        gen_loss = self.g_loss(d_fake)
        gen_loss.backward()
        self.optim['g'].step()
        # Train the discriminator.
        self.optim['d'].zero_grad()
        hs_reals, d_real = self.d(x)
        hs_fakes, d_fake = self.d(fake.detach())
        #d_loss = self.bce(d_fake, 0) + self.bce(d_real, 1)
        d_loss = self.d_loss_fake(d_fake) + self.d_loss_real(d_real)
        # Do the mix.
        alpha = self.sample_lambda(x.size(0))
        # Sample the index. If it is == 0, do first hidden
        # layer, else if 1, do second hidden, else do pixel.
        if self.mixup == 'vh1':
            nc = 2 # sample in (0,1)
        else:
            nc = 3 # sample in (0,1,2)
        idx = np.random.randint(0, nc)
        if idx < (nc-1):
            # Do mixup on either first or second hidden layer,
            # (which is determined by the idx sampled)
            hs_mix = alpha*hs_reals[idx] + (1.-alpha)*hs_fakes[idx]
            d_xz = self.d.partial_forward(hs_mix, idx)
            if self.mixup_ff:
                # If fake-fake mixes are enabled
                perm = torch.randperm(hs_fakes[idx].size(0)).view(-1).long()
                hs_fakes_perm = hs_fakes[idx][perm]
                hs_ff_mix = alpha*hs_fakes[idx] + (1.-alpha)*hs_fakes_perm
                d_xz_ff = self.d.partial_forward(hs_ff_mix, idx)
        else:
            # Do mixup in the pixels.
            xz = Variable(alpha*x.data + (1.-alpha)*fake.data)
            _, d_xz = self.d(xz)
            if self.mixup_ff:
                # If fake-fake mixes are enabled
                perm = torch.randperm(fake.size(0)).view(-1).long()
                fake_perm = fake[perm]
                xz_ff = Variable(alpha*fake.data + (1.-alpha)*fake_perm.data)
                _, d_xz_ff = self.d(xz_ff)
        d_loss += self.d_loss_fake(d_xz)
        if self.mixup_ff:
            d_loss += self.d_loss_fake(d_xz_ff)
        d_loss.backward()
        self.optim['d'].step()
        ##################################
        # Also compute the gradient norm.
        # Grad norm for D_REAL
        _, d_real = self.d(x)
        g_norm_x = self.grad_norm(d_real, x)
        if self.dnorm > 0.:
            self.optim['d'].zero_grad()
            (g_norm_x*self.dnorm).backward()
            self.optim['d'].step()
        self.optim['d'].zero_grad()
        ##################################
        losses = {
            'g_loss': gen_loss.data.item(),
            'd_loss': d_loss.data.item(),
            'd_real_norm': g_norm_x.data.item()
        }
        outputs = {
            'x': x.detach(),
            'gz': fake.detach(),
        }
        return losses, outputs
    
    def train_on_instance(self, z, x, **kwargs):
        if self.mixup == 'pixel':
            return self._train_on_instance_mixup(
                z, x, **kwargs)
        elif self.mixup is not None and self.mixup.startswith('vh'):
            return self._train_on_instance_mixup_hidden(
                z, x, **kwargs)
        else:
            return self._train_on_instance(
                z, x, **kwargs)
