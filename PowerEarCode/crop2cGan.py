import numpy as np
import pandas as pd
import librosa.display
from scipy.fftpack import fft,fftshift
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import wave
import os
import numpy as np
import random
from scipy.io import wavfile
data_count_train = 0
data_count_test = 0
if __name__ == '__main__':
    crop_slice = 32385
    hop_len = int(crop_slice / 10)
    real_A = librosa.load('./Germany/Tina Weinmayer_final.wav', sr=16000)[0]
    real_B = librosa.load('./Germany/Germany_afternoise_final.wav', sr=16000)[0]
    assert (len(real_A) == len(real_B))
    x=61
    train_dataset_len = len(real_A)-16000*x
    real_A_train = real_A[-1*train_dataset_len:]
    real_B_train = real_B[-1*train_dataset_len:]
    train_dataset = []
    for i in range(0,int((len(real_A_train)-crop_slice)/hop_len)):
        data_set = np.hstack((real_A_train[i*hop_len:i*hop_len+crop_slice], real_B_train[i*hop_len:i*hop_len+crop_slice]))
        assert (len(data_set)==crop_slice*2)
        train_dataset.append(data_set)


    real_A_test = real_A[:int(x*16000)]
    real_B_test = real_B[:int(x*16000)]
    assert(len(real_A_test) == len(real_B_test))
    test_dataset = []
    for i in range(0, int(len(real_A_test) / crop_slice)):
        data_set = np.hstack((real_A_test[i * crop_slice:i * crop_slice + crop_slice], real_B_test[i * crop_slice:i * crop_slice + crop_slice]))
        assert (len(data_set) == crop_slice * 2)
        test_dataset.append(data_set)

    random.shuffle(train_dataset)
    print("The size of traing dataset:", len(train_dataset))
    if not os.path.exists("./Germany/train/"):
        os.makedirs("./Germany/train/")
    for i in range(len(train_dataset)):
        wavfile.write("./Germany/train/{}.wav".format(data_count_train + 1), 16000, train_dataset[i])
        data_count_train += 1
    print("The training set has been saved to './train/xx.wav'")

    real_A_test = real_A[:int(x*16000)]
    real_B_test = real_B[:int(x*16000)]
    assert(len(real_A_test) == len(real_B_test))
    test_dataset = []
    for i in range(0, int(len(real_A_test) / crop_slice)):
        data_set = np.hstack((real_A_test[i * crop_slice:i * crop_slice + crop_slice], real_B_test[i * crop_slice:i * crop_slice + crop_slice]))
        assert (len(data_set) == crop_slice * 2)
        test_dataset.append(data_set)
    print("The size of test dataset:", len(test_dataset))
    if not os.path.exists("./Germany/test/"):
        os.makedirs("./Germany/test/")
    for i in range(len(test_dataset)):
        wavfile.write("./Germany/test/{}.wav".format(data_count_test+1), 16000, test_dataset[i])
        data_count_test += 1
    print("The test set has been saved to './test/xx.wav'")
