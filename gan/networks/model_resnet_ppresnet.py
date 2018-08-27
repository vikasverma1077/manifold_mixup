from torch import nn
from networks.model_resnet import (FirstResBlockDiscriminator,
                                   Generator,
                                   ResBlockDiscriminator,
                                   DISC_SIZE,
                                   SpectralNorm,
                                   channels)


class Discriminator(nn.Module):
    """
    This discriminator differs from the one in model_resnet
      in that we have a preprocessor conv right before the
      main model.
    """
    def __init__(self, spec_norm=False, sigmoid=False):
        super(Discriminator, self).__init__()

        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x : x

        #self.preproc = nn.Conv2d(channels, 16, 3, stride=1, padding=1)
        self.preproc = ResBlockDiscriminator(3, 16, stride=1,
                                             spec_norm=spec_norm)
        
        self.model = nn.Sequential(
            FirstResBlockDiscriminator(16, DISC_SIZE, stride=2,
                                       spec_norm=spec_norm),
            ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2,
                                  spec_norm=spec_norm),
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
        preproc = self.preproc(x)
        return preproc, self.partial_forward(preproc)

    def partial_forward(self, preproc, idx=-1):
        pre_fc = self.model(preproc).view(-1,DISC_SIZE)
        result = self.fc(pre_fc)
        if self.sigmoid:
            return self.sigm(result)
        else:
            return result

def get_network(z_dim):
    gen = Generator(z_dim)
    disc = Discriminator(sigmoid=True)
    return gen, disc

if __name__ == '__main__':
    a,b = get_network(62)
    print(a)
    print(b)
