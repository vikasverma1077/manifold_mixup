import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()


        self.c1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02),
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02),
            nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(256)
            )

        self.c2 = nn.Linear(256*4*4, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        h = self.c1(x)
        h = h.resize(128, 256*4*4)
        h = self.c2(h)
        h = self.sigmoid(h)

        h = h.resize(128)

        return h


