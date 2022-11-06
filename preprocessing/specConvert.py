import librosa
import pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa.display


def getSpectrogram(waveform_path):
    data, samplerate = librosa.load(waveform_path)
    time_sec = (len(data)/samplerate)
    step = time_sec/len(data)
    i = 0
    time_divion = []
    while i <= time_sec-step:
        time_divion.append(i)
        i = i+step
    four_sec_step_number = (4*len(time_divion))/time_sec
    fig, ax = plt.subplots()
    mel_feat = librosa.feature.melspectrogram(y=data, sr=44000)
    power = librosa.power_to_db(mel_feat, ref=np.max)
    plt.axis('off')
    librosa.display.specshow(power)
    plt.savefig(
        f'C:/Users/Roshni/OneDrive/Desktop/Sem7/Project 1/ECG UI/spectrograms/Spectrogram.png')
