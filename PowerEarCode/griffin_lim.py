# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import librosa
import soundfile as sf
import os


def griffin_lim(magnitude_spectrogram, n_iter=100, hop_length=127, win_length=510):
    phase = np.exp(2j * np.pi * np.random.rand(*magnitude_spectrogram.shape))
    complex_spectrogram = magnitude_spectrogram * phase
    for i in range(n_iter):
        audio = librosa.istft(complex_spectrogram, hop_length=hop_length, win_length=win_length)
        complex_spectrogram = librosa.stft(audio, n_fft=win_length, hop_length=hop_length)
        phase = np.exp(1j * np.angle(complex_spectrogram))
        complex_spectrogram = magnitude_spectrogram * phase
    audio = librosa.istft(complex_spectrogram, hop_length=hop_length, win_length=win_length)
    return audio
    
folder_path1="./results/iPhone14/test/images"#单一设备

folder_path2="./MCD/iPhone14/fake"
if not os.path.exists(folder_path2):
    os.makedirs(folder_path2)
for i in range(1, 31):  
    img_name = f'{i}_encoded.png'
    img_path = os.path.join(folder_path1, img_name)

    if os.path.exists(img_path):
        img = Image.open(img_path)
        gray_img = img.convert('L')
        img_array = np.array(gray_img)
        db_res = (img_array / 255) * (80) - 80
        spec = 10 ** (db_res / 20) * 60
        audio = griffin_lim(spec)
        wav_name = f'{i}.wav'
        sr = 16000
        sf.write(os.path.join(folder_path2, wav_name), audio, sr)
        print(f'Processed {img_name} to {wav_name}')

folder_path2="./MCD/iPhone14/input"
if not os.path.exists(folder_path2):
    os.makedirs(folder_path2)
for i in range(1, 31):  
    img_name = f'{i}_input.png'
    img_path = os.path.join(folder_path1, img_name)

    if os.path.exists(img_path):
        img = Image.open(img_path)
        gray_img = img.convert('L')
        img_array = np.array(gray_img)
        db_res = (img_array / 255) * (80) - 80
        spec = 10 ** (db_res / 20) * 60
        audio = griffin_lim(spec)
        wav_name = f'{i}.wav'
        sr = 16000
        sf.write(os.path.join(folder_path2, wav_name), audio, sr)
        print(f'Processed {img_name} to {wav_name}')

folder_path2="./MCD/iPhone14/real"
if not os.path.exists(folder_path2):
    os.makedirs(folder_path2)
for i in range(1, 31): 
    img_name = f'{i}_ground truth.png' 
    img_path = os.path.join(folder_path1, img_name)

    if os.path.exists(img_path):
        img = Image.open(img_path)
        gray_img = img.convert('L')
        img_array = np.array(gray_img)
        db_res = (img_array / 255) * (80) - 80
        spec = 10 ** (db_res / 20) * 60
        audio = griffin_lim(spec)

        wav_name = f'{i}.wav'
        sr = 16000
        sf.write(os.path.join(folder_path2, wav_name), audio, sr)
        print(f'Processed {img_name} to {wav_name}')