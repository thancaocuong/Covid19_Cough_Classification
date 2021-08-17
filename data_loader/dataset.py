import torch
import os
import glob
import pandas as pd
import soundfile as sf
from PIL import Image
import numpy as np
import cv2
import librosa
import random
from .audio_preprocessing import mfcc_feature, extract_mfcc_feature
from .audio_preprocessing import trim_and_pad, padding_repeat, random_crop

from transformers import Wav2Vec2Processor, Wav2Vec2Model

import copy
from multiprocessing import Pool
from functools import partial

def load_audio_(i, df=None, target_sr=22050):
    item = df.iloc[i]
    uuid = item['uuid']
    audio_path = item['audio_paths']
    audio, sr = sf.read(audio_path, dtype="float32")
    audio = librosa.resample(audio, sr, target_sr)
    return uuid, audio

def load_audio(df, target_sr):
    audios = {}

    pool = Pool()
    tmp = pool.map(
        partial(
            load_audio_, df = df,
            target_sr = target_sr
        ),
        range(len(df))
    )
    for key, audio in tmp:
        audios[key] = audio
    return audios

class CovidDataset(torch.utils.data.Dataset):
    def __init__(self, df, audio_folder, config, test = False, audio_transforms=None, image_transform=None):
        super().__init__()
        self.df = df
        self.audio_folder = audio_folder
        self.image_transform = image_transform
        self.audio_transforms = audio_transforms
        self.test = test

        self.audio_config = config['dataset'].get('audio_config', None)
        if self.audio_config is None:
            self.audio_config = {}
        
        self.target_sr = self.audio_config.get('target_sr', 48000)
        max_duration = self.audio_config.get('max_duration', 15)
        self.max_samples = int(max_duration * self.target_sr)

        cache = config['dataset'].get('cache', False)
        if cache:
            self.audios = load_audio(self.df, self.target_sr)
        else:
            self.audios = None

        # self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        # processor = Wav2Vec2Processor.from_pretrained(model_name_or_path,)
        # target_sampling_rate = processor.feature_extractor.sampling_rate

    def get_label(self, idx):
        return self.df.iloc[idx]["assessment_result"].astype("float32")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        label = item["assessment_result"].astype("float32")
        # one hot
        # label_encoded = np.zeros((label.size, 2))
        # label_encoded[np.arange(label.size),int(label)] = 1

        label_encoded = torch.tensor(label, dtype=torch.long)
        
        if self.audios is not None:
            uuid = item['uuid']
            audio = copy.deepcopy(self.audios[uuid])
            sr = self.target_sr
        else:
            try:
                audio_path = item['audio_paths']
                audio, sr = sf.read(audio_path, dtype="float32")
                # audio, sr = librosa.load(audio_path, sr = self.target_sr)
            except:
                audio_path = os.path.join(self.audio_folder, "%s.wav"%item['uuid'])
                audio, sr = sf.read(audio_path, dtype="float32")
            audio = librosa.resample(audio, sr, self.target_sr)
        
        # max_duration = random.choice(range(7, 16))
        # max_samples =  max_duration * self.target_sr
        if self.test:
            audio = padding_repeat(audio, self.max_samples)
            # audio = padding_repeat(audio, max_samples)
        else:
            audio = random_crop(audio, self.max_samples)
            # audio = random_crop(audio, max_samples)
            # audio = trim_and_pad(audio, self.max_samples)

        if self.audio_transforms is not None:
            audio_path = item['audio_paths']
            uuid = audio_path.split('/')[-1].split('.')[0]
            if uuid[-2] != '_':
                audio = self.audio_transforms(samples=audio, sample_rate=sr)
            else:
                # print('oversampling no need to aug')
                pass

        #mfcc
        # image = extract_mfcc_feature(audio, self.target_sr, self.mfcc_config)
        # image = extract_mfcc_feature(audio, sr, self.mfcc_config)
        
        # melspec
        # image = audio2melspec(audio, self.target_sr, self.melspec_config)

        # if self.image_transform is not None:
        #     image = self.image_transform(image = image)['image']

        # return image, label_encoded

        #* train 44 - 52
        return torch.from_numpy(audio).float(), label_encoded

        #* wav2vec2
        # audio_ = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values[0]  # Batch size 1
        # return audio_.float(), label_encoded
        
        