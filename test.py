import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from skimage import filters
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

from model import _netG
from utils import getspectrumset, createDirectory

root = '../test_pattern_result'
createDirectory(root)

parser = argparse.ArgumentParser(description='DCGAN Nano Photonics Model Test, Create Pattern Image from Spectrum')
parser.add_argument('--file', type=str, help='Spectrum file path')
parser.add_argument('--model', type=str, help='Model file path')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(f'Device: {device}')

noise_z = 100 # 노이즈 벡터의 크기
spectrum_z = 200 # Spectrum 벡터 크기

netG = _netG(noise_z, spectrum_z).to(device)
netG.load_state_dict(torch.load(args.model))
netG.eval()

test_spectrum_file_path = args.file
spectrum = getspectrumset(test_spectrum_file_path, spectrum_z)

with torch.no_grad():
    fixed_noise = torch.randn(1, noise_z, 1, 1, device=device)

    spectrumgn = torch.FloatTensor(spectrum)
    spectrumgn.resize_(1, spectrum_z, 1, 1)
    spectrumGN = Variable(spectrumgn).to(device)

    pdf64 = netG((fixed_noise, spectrumGN[0].unsqueeze(0)))
    pdf49 = F.interpolate(pdf64, size=(49, 49))

    pdf64 = pdf64.squeeze().cpu().detach()
    pdf49 = pdf49.squeeze().cpu().detach()


    file_name = os.path.basename(test_spectrum_file_path).split('.')
    file_name = file_name[0]

    # Binarization using OTSU Algorithm
    grayscale_output = pdf64.add(1).div(2).mul(255).add_(0.5).clamp_(0, 255).byte().numpy() # convert grayscale
    threshold = filters.threshold_otsu(grayscale_output)
    threshold_grayscale_output = grayscale_output > threshold
    threshold_grayscale_output = threshold_grayscale_output.astype(np.uint8)
    plt.imsave(f'{root}/{file_name}64.png', pdf64, cmap=cm.gray)

    grayscale_output = pdf49.add(1).div(2).mul(255).add_(0.5).clamp_(0, 255).byte().numpy() # convert grayscale
    threshold = filters.threshold_otsu(grayscale_output)
    threshold_grayscale_output = grayscale_output > threshold
    threshold_grayscale_output = threshold_grayscale_output.astype(np.uint8)
    plt.imsave(f'{root}/{file_name}49.png', pdf49, cmap=cm.gray)