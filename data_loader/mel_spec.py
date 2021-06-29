import cv2
import torch
import numpy as np
import random
import sys

import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa.display
import numpy as np
import soundfile as sf
from  soundfile import SoundFile
import glob 
import tqdm
import cv2
from pathlib import Path
from .audio_preprocessing import remove_silent

def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)
    
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V

def audio2image(audio, original_sr, audio_transfroms=None):    
    #melspectrogram
    if audio_transfroms is not None:
        audio, original_sr = audio_transfroms(audio, original_sr)
    n_fft = 512   
    hop_length=int(len(audio)/256)

    mel_spect = librosa.feature.melspectrogram(y=audio, sr=original_sr, n_fft=n_fft, hop_length=hop_length)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max).astype(np.float32)
    image = mono_to_color(mel_spect)
    # new_img =  cv2.merge([image[:, :128], image[:, 128:256], image[:,256:384]])
    new_img =  cv2.merge([image, image, image])
    new_img = cv2.resize(new_img, (256, 128))
    return new_img

def create_spectrogram(clip, sample_rate, audio_transfroms=None):
    # clip = remove_silent(clip, sample_rate, 0.0025)
    if audio_transfroms is not None:
        clip, sample_rate = audio_transfroms(clip, sample_rate)
    plt.interactive(False)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = Path("name" + '.jpg')
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2RGB)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del clip,sample_rate,fig,ax,S
    img = cv2.resize(img, (256, 128))
    return img