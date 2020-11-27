import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

def FFT(x):
    fft = np.fft.fft(x)
    spec = np.abs(fft)
    return spec[:len(spec)//2]

def sin(f):
    fs=200
    length = 1
    N = fs*length
    t = np.arange(0, 1, 1/N)
    s = np.sin(2*np.pi*f*t)
    return s

def STFT(x, win_length, hop_length, fs):
    n_fft = int((win_length/1000)*fs)
    hop_length = int((hop_length/1000)*fs)
    spec = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
    return np.abs(spec)

#audio, fs = librosa.load('../audio/OSR_us_000_0010_8k.wav')
#sp = STFT(x=audio, win_length=25, hop_length=10, fs=fs)
