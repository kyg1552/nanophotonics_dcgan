import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn

# Generator Network
class _netG(nn.Module):
    def __init__(self, noise_z, spectrum_z):
        super(_netG, self).__init__()
        
        self.noise = nn.Sequential(
            # Test
            nn.ConvTranspose2d(noise_z, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),  
            # shape: batch_size x 512 x 4 x 4
        )
        self.Spectrum = nn.Sequential(
            # Test
            nn.ConvTranspose2d(spectrum_z, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # shape: batch_size x 512 x 4 x 4
        )
        self.gn = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # shape: batch_size x 512 x 8 x 8
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # shape: batch_size x 256 x 16 x 16

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # shape: batch_size x 128 x 32 x 32

            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # shape: batch_size x 1 x 64 x 64
        )

    def forward(self, data):
        output1 = self.noise(data[0])
        output2 = self.Spectrum(data[1])
        output = torch.cat([output1, output2], 1)
        output = self.gn(output) # -1 ~ 1
        
        return output # shape: batch_size x 1 x 64 x 64

# Discriminator Network
class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()

        self.x_gn_z = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # shape: batch_size x 64 x 32 x 32
        )
        self.spectrum = nn.Sequential(
            nn.Conv2d(200, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # shape: batch_size x 64 x 32 x 32
        )
        self.dn = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # shape: batch_size x 256 x 16 x 16

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # shape: batch_size x 512 x 8 x 8

            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # shape: batch_size x 1024 x 1 x 1

            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # shape: batch_size x  x 1 x 1 x 1
        )

    def forward(self, data):
        output1 = self.x_gn_z(data[0])
        output2 = self.spectrum(data[1])
        output3 = torch.cat([output1, output2], 1)
        output = self.dn(output3)
        output = output.view(-1, 1).squeeze(1)
        return output # shape: batch_size