# 참고 블로그: https://kangbk0120.github.io/articles/2017-08/dcgan-pytorch
# 참고 논문: https://www.degruyter.com/document/doi/10.1515/nanoph-2019-0117/html

# DCGAN Nano Photonics
from __future__ import print_function # 

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from glob import glob
from tqdm import tqdm
from datetime import datetime
import random

from model import _netG, _netD
from utils import *

niter = 1000 # 학습 에폭 수

noise_z = 100 # 노이즈 벡터의 크기
spectrum_z = 200 # Spectrum 벡터 크기
nc = 1 # Image 채널의 수
lr = 0.0002 # Learning Rate
beta1 = 0.5 # Adam Optimization Beta Parameter
imageSize = 64 # 만들어지는 이미지의 크기
batchSize = 64 # 미니배치의 크기, 논문에서는 64 or 128로 학습 실험함.
rho = 0.5

outf = "../checkpoint" # 모델 저장할 폴더
createDirectory(outf)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(f'Device: {device}')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(imageSize, imageSize))
])

############ 데이터 불러오는 곳 시작 ##################################################################################################
#####################################################################################################################################
# 데이터셋 폴더
root = '../dataset'

# Spectrum, Design Image 데이터 불러오기
circle_spectrum_file_path = os.path.join(root, 'circle/spectrum/*.txt')
circle_spectrumset = getspectrumset(circle_spectrum_file_path, spectrum_z)
circle_pattern_imgs = glob(os.path.join(root, 'circle/pattern/*.png'))

ring_spectrum_file_path = os.path.join(root, 'ring/spectrum/*.txt')
ring_spectrumset = getspectrumset(ring_spectrum_file_path, spectrum_z)
ring_pattern_imgs = glob(os.path.join(root, 'ring/pattern/*.png'))

cross_spectrum_file_path = os.path.join(root, 'cross/spectrum/*.txt')
cross_spectrumset = getspectrumset(cross_spectrum_file_path, spectrum_z)
cross_pattern_imgs = glob(os.path.join(root, 'cross/pattern/*.png'))

min_data_num = min([len(circle_spectrumset), len(ring_spectrumset), len(cross_spectrumset)])

circle_spectrumset, circle_pattern_imgs = zip(*random.sample(list(zip(circle_spectrumset, circle_pattern_imgs)), min_data_num))
ring_spectrumset, ring_pattern_imgs = zip(*random.sample(list(zip(ring_spectrumset, ring_pattern_imgs)), min_data_num))
cross_spectrumset, cross_pattern_imgs = zip(*random.sample(list(zip(cross_spectrumset, cross_pattern_imgs)), min_data_num))

spectrumset = circle_spectrumset + ring_spectrumset + cross_spectrumset
pattern_imgs = circle_pattern_imgs + ring_pattern_imgs + cross_pattern_imgs

############ 데이터 불러오는 곳 끝 ################################################################################################
##################################################################################################################################

dataloader = []
for spectrum, img in zip(spectrumset, pattern_imgs):
    spectrum = torch.FloatTensor(spectrum)
    img = transform(img_loader(img))
    dataloader.append([img, spectrum])

trainloader = DataLoader(dataset=dataloader, batch_size=batchSize, shuffle=True)

netG = _netG(noise_z, spectrum_z).to(device)
netG.apply(weights_init)

netD = _netD().to(device)
netD.apply(weights_init)

criterion = nn.BCELoss()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

design_label = 1
pdf_label = 0

fixed_noise = torch.randn(1, noise_z, 1, 1, device=device)

upsample = nn.Upsample(scale_factor=imageSize)

################ 학습 시작 ########################################
for epoch in range(1, niter+1):
    for i, data in enumerate(tqdm(trainloader, desc=f"Epoch: {epoch}", ascii=True)):
        ## Training discriminator
        netD.zero_grad()
        design_pattern, spectrumin = data[0].to(device), data[1].to(device)
        batch_size = design_pattern.size(0)

        spectrumdn = spectrumin
        spectrumdn.resize_(batch_size, spectrum_z, 1, 1) # 200 x 64 x 64
        spectrumdn = upsample(spectrumdn)
        spectrumDN = Variable(spectrumdn).to(device)

        spectrumgn = spectrumin
        spectrumgn.resize_(batch_size, 200, 1, 1)
        spectrumGN = Variable(spectrumgn).to(device)

        label = torch.full((batch_size,), design_label).type(torch.FloatTensor).to(device)

        output = netD((design_pattern, spectrumDN)) # spectrumDN shape: 64 x 64 x 200
        lossD_design = criterion(output, label)
        lossD_design.backward()
        D_x = output.mean().item()

        # train with pdf
        noise = torch.randn(batch_size, noise_z, 1, 1, device=device)

        pdf = netG((noise, spectrumGN))
        label.fill_(pdf_label)
        output = netD((pdf.detach(), spectrumDN))
        lossD_pdf = criterion(output, label)
        lossD_pdf.backward()
        D_G_z1 = output.mean().item()

        # 논문에서는 rho를 이용하여 design과 pdf 사이의 중요도 조정
        lossD = ((1 - rho) * lossD_design) + (rho * lossD_pdf)
        optimizerD.step()

        ## Training Generator
        netG.zero_grad()
        label.fill_(design_label)
        output = netD((pdf, spectrumDN))
        lossG = criterion(output, label)
        lossG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

    result_dict = {"epoch":epoch,"loss_D":lossD.item(),"loss_G":lossG.item(),"score_D":D_x,"score_G1":D_G_z1,"score_G2":D_G_z2}
    print(result_dict)
    
    # 100 에폭마다 모델 저장
    if epoch % 100 == 0:
        day = datetime.now().strftime('%Y%m%d')
        torch.save(netG.state_dict(), f'{outf}/netG_{day}.pth')
        torch.save(netD.state_dict(), f'{outf}/netD_{day}.pth')
        print(f'Model Save, Epoch:{epoch}')

################ 학습 끝 ########################################