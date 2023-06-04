## Band pass filter design operating in IR band using machine learning GAN algorithm

### How to use
  1. Dataset 구성
      * dataset 폴더 생성
      * dataset 폴더에 각 패턴별로 spectrum, pattern 폴더 생성하고 데이터 셋 저장
      * dataset/{pattern_name}/spectrum
      * dataset/{pattern_name}/pattern
  2. 발산한 파일 제거
      * delerror.py 안에 dataset root 폴더(각 패턴별 폴더), 작성해주고 ```python delerror.py```으로 발산한 파일 제거 진행
  3. 패턴 txt 파일 -> png 파일로 변환
      * dataset 폴더에 저장된 각 패턴별 폴더에서 ```python pattern_image_generation_from_txt.py --file '../dataset/{pattern_name}/pattern'```와 같이 실행하면 --file에 입력한 경로에 imgs 폴더에 패턴 이미지들이 생성된다.
  4. 학습
      * 학습 시작 전 train.py의 niter(에폭 수), dataload 부분에 각 패턴별 데이터 경로 수정한 후 ```python train.py``` 실행
  5. 테스트
      * 실행 예시 ```python test.py --file ../test/star.txt --model ../checkpoint/netG_20220820.pth```와 같이 테스트할 spectrum.txt파일을 불러와서 실행
