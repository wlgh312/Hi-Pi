import warnings
warnings.filterwarnings('ignore')
import librosa #리브로사 모듈 가져오기
import librosa.display
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import os

from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras.models import load_model
from PIL.Image import core as _imaging

#모델 로드
to_Train = False#훈련하지않고 예측한다는 의미
base_dir_out = '/home/parkgeonwoo/work/soundRecog'#이번 프로젝트가 모두 담겨있는 폴더
#outDir = '{}/data/cnn_CL4'.format(base_dir_out)#모델이  저장되어있 경로
fnmodel=base_dir_out+'/model4.h5'#모델의 경로+모델이름
#print(fnmodel)
if not to_Train:
    if not os.path.isfile(fnmodel):#모델 파일이 없을 경우
        to_Train = True
        print('Model file {} not exists !!'.format(fnmodel))
        print('Setting to_Train --> {}'.format(to_Train))

    else:#모델 파일이 있을 경우
        print('Model file Okay : {}  !!'.format(fnmodel))
        #model.load(fnmodel)

#%%time
model = load_model(fnmodel)#모델 로드
print('Loaed a model {}.. okay'.format(fnmodel))

#소리 인식 및 예측
while True:
    print("-----------------------------")
    key = input("Enter a key for start(q) : ")
    if key=='q':
        print("Start!")
        #for i in range(3):
        os.system("arecord -D plughw:2,0 -d 2 --format=S16_LE --rate=44100 /home/parkgeonwoo/work/soundRecog/audio/test1/mic.wav")
        #arecord -D plughw:2,0 -d 2 --format=S16_LE --rate=44100 ./work/soundRecog/audio/test1/mic.wav
        audio_path = '/home/parkgeonwoo/work/soundRecog/audio/test1/mic.wav' #내가 마이크를 통해 녹음한 오디오를 저장한 경로를 저장
        y, sr = librosa.load(audio_path, sr=44100)
        print("Making img...")
        min_level_db = -100
        def _normalize(S):
          return np.clip((S - min_level_db) / -min_level_db, 0, 1)

        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
        log_S = librosa.amplitude_to_db(S, ref=np.max)
        norm_S = _normalize(log_S)
        plt.figure(figsize=(4, 4))
        librosa.display.specshow(norm_S, sr=sr)
        plt.tight_layout()
        fig = plt.gcf()
        plt.show()
        fig.savefig('/home/parkgeonwoo/work/soundRecog/audio/test1/mic.png' )#spectrogram으로 저장

        datagen=ImageDataGenerator(rescale=1./255.)
        pred_generator = datagen.flow_from_directory(base_dir_out+'/audio',#test폴더안에 폴더가 두개이상이어야됨(확인할 이미지가 담겨있는 폴더말고 빈폴더를 하나 더 만들면 됩니다.)
                                           batch_size=1,
                                           class_mode = 'categorical',
                                           target_size = (64, 64))
        STEP_SIZE_VALID=int(np.ceil(pred_generator.n/pred_generator.batch_size))
        v = model.predict_generator(pred_generator, steps=STEP_SIZE_VALID)
        pred = np.argmax(v)
        #결과 출력
        print("  /\___/\ ")
        print(" / |   | \ ")
        print("(    ^    )")
        print("   m    m")
        print(os.system("date +%Y-%m-%d_%P%I:%M"), end="")
        if pred == 0:#물 마시는 소리
          print("\033[96m"+"Your cat drinks water!"+"\033[0m")
        elif pred == 1:#밥 먹는 소리
          print("\033[96m"+"Your cat eats feed!"+"\033[0m")
        elif pred == 2:#화장실 소리
          print("\033[96m"+"Your cat goes to the toilet!"+"\033[0m")
        elif pred == 3:#잡음
          print("\033[96m"+"It's noise.."+"\033[0m")
    else:
        print("Wrong key input...retry")
