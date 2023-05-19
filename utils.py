import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from PIL import Image
from glob import glob
import os
import shutil

def createDirectory(dir):
    try:
        if not os.path.exists(dir):
            print(f"Make Directory: {dir}")
            os.makedirs(dir)
    except:
        print("Error: Failded to create the directory")

# Binary Image Loader
def img_loader(path):
    return Image.open(path).convert('1')

def getspectrumset(path, spectrum_z):
    spectrum_files = glob(path)  # 스펙트럼 파일 있는 경로에 모든 파일 불러오기
    spectrumset = []
    for spectrum_file in spectrum_files: # txt 파일을 하나씩 불러오기
        spectrum_error_file = os.path.basename(spectrum_file)
        with open(spectrum_file, 'r') as f:
            spectrum = f.readlines()
        
        spectrum = spectrum[1:-1] # 맨 윗줄 설명하는 정보 줄과 맨 밑 개행 줄 제거
        if len(spectrum) != spectrum_z:
            print(f"Can't Load Spectrum file:{spectrum_error_file}")
            return None
        
        spectrum = [float(s.strip().split(', ')[1]) for s in spectrum]
        spectrumset.append(spectrum)

    return spectrumset

# Deep Neural Networks weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def delerrordata(spectrum_path, pattern_path, error_spectrum_path, error_pattern_path, spectrum_z):
    spectrum_files = glob(os.path.join(spectrum_path, '*.txt'))  # 스펙트럼 파일 있는 경로에 모든 파일 불러오기
    pattern_files = glob(os.path.join(pattern_path, '*.txt')) # 패턴 파일 있는 경로에 모든 파일 불러오기
    error_files = []

    for spectrum_file, pattern_file in zip(spectrum_files, pattern_files): # txt 파일을 하나씩 불러오기
        spectrum_error_file = os.path.basename(spectrum_file)
        pattern_error_file = os.path.basename(pattern_file)
        
        with open(spectrum_file, 'r') as f:
            spectrum = f.readlines()
        
        spectrum = spectrum[1:-1] # 맨 윗줄 설명하는 정보 줄과 맨 밑 개행 줄 제거
        if len(spectrum) != spectrum_z:
            print(f"Can't Load Spectrum file:{spectrum_error_file}")
            return None
        
        spectrum = [float(s.strip().split(', ')[1]) for s in spectrum]
        for v in spectrum:
            if v > 1:
                error_files.append(spectrum_error_file)
                error_spectrum = os.path.join(spectrum_path, spectrum_error_file)
                error_pattern = os.path.join(pattern_path, pattern_error_file)

                shutil.move(error_spectrum, error_spectrum_path)
                shutil.move(error_pattern, error_pattern_path)

    print(f"총 발산 파일 수: {len(error_files)}")
    print(f"발산한 파일명: {error_files}")
