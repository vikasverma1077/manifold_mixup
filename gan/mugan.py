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
    loss = torch.nn.BCELoss()
    if prediction.is_cuda:
        loss = loss.cuda()
    target = target.view(-1, 1)
    return loss(prediction, target)

def jsgan_d_fake_loss(d_fake, label=0):
    return bce(d_fake, label)

def jsgan_d_real_loss(d_real, label=1):
    return bce(d_real, label)

def jsgan_g_loss(d_fake, label=1):
    return bce(d_fake, label)

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
        if mixup in ['pixel', 'vh1', 'vh2']:
            if mixup == 'pixel':
                self.mixup = [-1]
            elif mixup == 'vh1':
                self.mixup = [0]
            elif mixup == 'vh2':
                self.mixup = [0, 1]
        elif mixup is None:
            self.mixup = None
        else:
            self.mixup = [int(x) for x in mixup.split(',')]
        
        if lr_sched not in [None, 'exp']:
            raise Exception("lr_sched must be either None or 'exp'")
        #self.mixup_ff = mixup_ff
        self.lr_sched = lr_sched
        self.alpha = alpha
        if lr_sched is not None:
            self.scheduler = {}
            self.scheduler['g'] = optim.lr_scheduler.ExponentialLR(
                self.optim['g'], gamma=0.99)
            self.scheduler['d'] = optim.lr_scheduler.ExponentialLR(
                self.optim['d'], gamma=0.99)
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

    def get_orig_mixup_mixes(self, real, fake):
        """
        Used for mixup_orig only.
        """
        ones = torch.ones(real.size(0), 1)
        zeros = torch.zeros(fake.size(0), 1)
        if self.use_cuda:
            ones = ones.cuda()
            zeros = zeros.cuda()
        # do fake-fake
        ff_perm1 = torch.randperm(fake.size(0)).view(-1).long()
        ff_perm2 = torch.randperm(fake.size(0)).view(-1).long()
        d1_f, d2_f = fake[ff_perm1], fake[ff_perm2]
        alpha = torch.randn(d1_f.size(0), 1).uniform_(0, self.alpha)
        if self.use_cuda:
            alpha = alpha.cuda()
        mix_ff = alpha.view(-1,1,1,1)*d1_f + (1.-alpha.view(-1,1,1,1))*d2_f
        label_ff = zeros
        # do real-fake
        rf_perm1 = torch.randperm(real.size(0)).view(-1).long()
        rf_perm2 = torch.randperm(real.size(0)).view(-1).long()
        d1_rf, d2_rf = real[rf_perm1], fake[rf_perm2]
        alpha2 = torch.randn(rf_perm1.size(0), 1).uniform_(0, self.alpha)
        if self.use_cuda:
            alpha2 = alpha2.cuda()
        mix_rf = alpha2.view(-1,1,1,1)*d1_rf + (1.-alpha2.view(-1,1,1,1))*d2_rf
        label_rf = alpha2*ones
        # do fake-real
        fr_perm1 = torch.randperm(real.size(0)).view(-1).long()
        fr_perm2 = torch.randperm(real.size(0)).view(-1).long()
        d1_fr, d2_fr = fake[fr_perm1], real[fr_perm2]
        alpha3 = torch.randn(fr_perm1.size(0), 1).uniform_(0, self.alpha)
        if self.use_cuda:
            alpha3 = alpha3.cuda()
        mix_fr = alpha3.view(-1,1,1,1)*d1_fr + (1.-alpha3.view(-1,1,1,1))*d2_fr
        label_fr = (1.-alpha3)*ones

        return (torch.cat((mix_ff, mix_rf, mix_fr), dim=0),
                torch.cat((label_ff, label_rf, label_fr), dim=0))
    
    def _train_on_instance_mixup_hidden(self, z, x, **kwargs):
        """Perform mixup in the hidden states"""
        self._train()
        x.requires_grad = True # for dnorm
        # --------------------
        # Train the generator.
        # --------------------
        self.optim['g'].zero_grad()
        fake = self.g(z)
        hs_reals, d_real = self.d(x)
        hs_fakes, d_fake = self.d(fake)
        # Sample a hidden layer.
        idx = np.random.choice(self.mixup)
        if idx == -1:
            xz, ll = self.get_orig_mixup_mixes(x, fake)
            _, d_xz = self.d(xz)
        else:
            h_xz, ll = self.get_orig_mixup_mixes(hs_reals[idx],
                                                 hs_fakes[idx])
            d_xz = self.d.partial_forward(h_xz, idx)
        gen_loss = -self.g_loss(d_xz, ll) + self.g_loss(d_fake)
        if (kwargs['iter']-1) % self.update_g_every == 0:
            gen_loss.backward()
            self.optim['g'].step()
        # ------------------------
        # Train the discriminator.
        # ------------------------
        self.optim['d'].zero_grad()
        hs_reals, d_real = self.d(x)
        hs_fakes, d_fake = self.d(fake.detach())
        # Sample a hidden layer.
        idx = np.random.choice(self.mixup)
        if idx == -1:
            xz, ll = self.get_orig_mixup_mixes(x, fake.detach())
            _, d_xz = self.d(xz)
        else:
            h_xz, ll = self.get_orig_mixup_mixes(hs_reals[idx],
                                                 hs_fakes[idx])
            d_xz = self.d.partial_forward(h_xz, idx)
        d_loss = self.d_loss_fake(d_xz, ll) + \
                 self.d_loss_fake(d_fake) + \
                 self.d_loss_real(d_real)
        d_loss.backward()
        self.optim['d'].step()
        ##################################
        # Also compute the gradient norm.
        # Grad norm for D_REAL
        _, d_real = self.d(x)
        if self.dnorm > 0.:
            g_norm_x = self.grad_norm(d_real, x)
            self.optim['d'].zero_grad()
            (g_norm_x*self.dnorm).backward()
            self.optim['d'].step()
        else:
            g_norm_x = torch.FloatTensor([-1.])
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
        if self.mixup is None:
            return self._train_on_instance(
                z, x, **kwargs)
        else:
            return self._train_on_instance_mixup_hidden(
                z, x, **kwargs)
