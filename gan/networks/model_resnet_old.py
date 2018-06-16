# ResNet generator and discriminator
from torch import nn
import torch.nn.functional as F

from networks.spectral_normalization import SpectralNorm
import numpy as np


channels = 3


class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, spec_norm=False):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x: x

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.spec_norm(self.conv1),
                nn.ReLU(),
                self.spec_norm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.spec_norm(self.conv1),
                nn.ReLU(),
                self.spec_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                self.spec_norm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            # if in_channels == out_channels:
            #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            # else:
            #     self.bypass = nn.Sequential(
            #         SpectralNorm(nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)),
            #         nn.AvgPool2d(2, stride=stride, padding=0)
            #     )


    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, spec_norm=False):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x: x

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            self.spec_norm(self.conv1),
            nn.ReLU(),
            self.spec_norm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.spec_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

GEN_SIZE=128
DISC_SIZE=128

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)

        self.model = nn.Sequential(
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z):
        return self.model(self.dense(z).view(-1, GEN_SIZE, 4, 4))

class Discriminator(nn.Module):
    def __init__(self, spec_norm=False, sigmoid=False):
        super(Discriminator, self).__init__()

        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x : x

        self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, DISC_SIZE,
                                           stride=2, spec_norm=spec_norm),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE,
                                      stride=2, spec_norm=spec_norm),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE,
                                      spec_norm=spec_norm),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE,
                                      spec_norm=spec_norm),
                nn.ReLU(),
                nn.AvgPool2d(8),
            )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = self.spec_norm(self.fc)
        if sigmoid:
            self.sigm = nn.Sigmoid()
        self.sigmoid = sigmoid

    def forward(self, x):
        pre_fc = self.model(x).view(-1,DISC_SIZE)
        result = self.fc(pre_fc)
        if self.sigmoid:
            return pre_fc, self.sigm(result)
        else:
            return pre_fc, result

    def partial_forward(self, x):
        if self.sigmoid:
            return self.sigm(self.fc(x))
        else:
            return self.fc(x)
        

def get_network(z_dim):
    gen = Generator(z_dim)
    disc = Discriminator(sigmoid=True)
    return gen, disc

if __name__ == '__main__':
    gen, disc = get_network(128)
    print(gen)
    print(disc)
