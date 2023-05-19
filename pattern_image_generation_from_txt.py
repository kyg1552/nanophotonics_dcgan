import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import os 
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

from utils import createDirectory

parser = argparse.ArgumentParser(description='Converting Pattern file from .txt to .png')
parser.add_argument('--file', type=str, help='Pattern .txt file path')
args = parser.parse_args()

# 패턴 txt 파일들이 있는 경로 작성해줘야함.
pattern_file_path = args.file

# 바이너리 이미지 파일이 저장될 폴더 지정
pattern_img_path = os.path.join(pattern_file_path, 'imgs') 

# 바이너리 이미지 파일이 저장될 폴더 생성(없으면 자동으로 생성됨)
createDirectory(pattern_img_path)

# 패턴 txt 파일 불러오기
pattern_files = glob(os.path.join(pattern_file_path, '*.txt'))
pattern_len = 49

### 패턴 파일(.txt)을 binary 이미지(.png)로 만들어서 저장
### 3.566612576103e+00 -> 1
### 1.450195599381e+00 -> 0
### 2.0 기준으로 이하는 0, 초과는 1로 처리

for pattern in pattern_files:
    file_name = os.path.basename(pattern).split('.')
    file_name = file_name[0]

    with open(pattern, 'r') as f:
        pattern = f.readlines()
        # pattern = pattern[-(pattern_len+1):-1] #  +1과 -1 는 밑에 1칸 개행이 있기 때문
        pattern = pattern[1:-1]

    patterns = []
    for pat in pattern:
        pat = pat.strip().split(' ')
        data = []
        for i in range(pattern_len):
            if float(pat[i]) < 2.:
                data.append(0.)
            else:
                data.append(1.)
        patterns.append(data)

    # 바이너리 이미지 파일(.png)로 저장
    patterns = np.array(patterns).astype(np.uint8)
    plt.imsave(os.path.join(pattern_img_path, f'{file_name}.png'), patterns, cmap=cm.gray)