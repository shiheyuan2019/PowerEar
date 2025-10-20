# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pywt
import csv
import wave
import struct
from scipy.signal import butter, lfilter, freqz,filtfilt, convolve
from scipy.fft import fft
import scipy.io.wavfile as wf
import numpy.fft as nf
from scipy.signal import wiener
import pandas as pd


def passband(audio,lowcut,highcut,Fs):
    nyq = 0.5 * Fs
    low = lowcut / nyq
    high = highcut / nyq
    order = 5
    b, a = butter(order, [low, high], btype='band')
    x_filtered = filtfilt(b, a, audio)
    return x_filtered


def lowpass_filter(audio, sr, cutoff_freq):
    nyquist_rate = sr / 2.0
    norm_cutoff_freq = cutoff_freq / nyquist_rate
    b, a = butter(5, norm_cutoff_freq, 'lowpass')
    filtered_audio = filtfilt(b, a, audio)
    return filtered_audio

    
def highpass_filter(audio, sr, cutoff):
    nyquist = sr / 2.0
    normal_cutoff = cutoff / nyquist
    
    b, a = butter(5, normal_cutoff, btype='highpass', analog=False)
    
    filtered_audio = filtfilt(b, a, audio)
    
    return filtered_audio

def wiener_filter(data,mySize):
    filtered_speech = wiener(data, mySize)
    return filtered_speech


def moving_average_filter(audio,window_size):
    window = np.ones(window_size)/float(window_size)
    audio_filtered = convolve(audio, window, mode='valid')
    return audio_filtered

def txt_file_list_data(data):
    global Amplit, Fs
    time = data[0]
    Amplit = data[1]
    Fs = 1 / (time[len(time) - 1] / len(time))
    print(Fs)
    print(Amplit)
    return Amplit, Fs



def create_wav_function(Amplit, Fs,file_name):
    f = wave.open(file_name, mode='wb')
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(Fs)
    f.setnframes(len(Amplit))
    f.setcomptype('NONE', 'not compressed')
    waveData = Amplit * 32767 / (max(abs(Amplit)))
    print(waveData)
    waveData = waveData.astype(np.short)
    for k in waveData:
        f.writeframesraw(struct.pack('h', k))
    f.close()
    
def convert_into_wav(filename,wavname):
    data = pd.read_csv(filename, header=None,skiprows = 1)
    txt_file_list_data(data)
    create_wav_function(Amplit, Fs,wavname)
    print('convert_into_wav Finished')
    
file_name  = 'D:/powerEar_guide/audioSet/E_FM/test.csv'

t = []
val = []

with open(file_name, encoding='utf-8', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for k, row in enumerate(spamreader):
        if k < 5:
            continue
        t.append(float(row[0]))
        val.append(float(row[1]))


dt = t[10] - t[9] 

t = [x - t[0] for x in t]

Fs = int(1.0 / dt)
NFFT = 1024
print('sampling_rate:',Fs)

x = val  


lowpass_param=15
highpass_param=8192
mySize_param=90
moving_average_params=90
#different filter here

#x_filter=passband(x,lowpass_param,highpass_param,Fs)
#x_filtered=lowpass_filter(x,Fs,highpass_param)
#x_wiener = wiener_filter(x_filter, mySize_param)
#x_filtered = moving_average_filter(x_wiener, moving_average_params)

x_filtered=x
df=pd.DataFrame(x_filtered,t[:len(x_filtered)])
print(len(x_filtered))
print(len(t))
df.to_csv('filtered_data.csv')
convert_into_wav("filtered_data.csv",file_name.split('.')[0]+"_filtered.wav")







