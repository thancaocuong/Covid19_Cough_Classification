import csv
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
from pprint import pprint
import os
import pandas as pd

class AstDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_params, transform=None, fold_idx=None):
        super().__init__()
        self.df = None
        pprint(dataset_params)
        self.dataset_params = dataset_params
        self.audio_folder = dataset_params.get("audio_folder", "")
        self.period = dataset_params.get("period", 7)
        self.for_test = dataset_params.get("for_test", False)
        self.melbins = dataset_params.get("melbins", 128)
        self.freqm = dataset_params.get("freqm", 48)
        self.noise = dataset_params.get("noise", False)
        self.timem = dataset_params.get("timem", 192)
        self.target_length = dataset_params.get("target_length", 512)
        self.norm_mean = dataset_params.get("norm_mean", -6.0542064)
        self.norm_std = dataset_params.get("std", 6.219794)
        self.skip_norm = dataset_params.get("skip_norm", True)
        self.transform = transform
        self.fold_idx = fold_idx
        self._load_data()

    def get_label(self, idx):
        return self.df.iloc[idx]["assessment_result"].astype("float32")

    def _load_data(self):
        df_path = os.path.join(self.dataset_params.get("csv_path", ""))
        self.df = pd.read_csv(df_path)
        if self.for_test:
            self.df = self.df[self.df["fold"] == self.fold_idx]
            try:
                self.df = self.df[self.df["is_augmented"] == 0]
            except:
                pass
        else:
            self.df = self.df[self.df["fold"] != self.fold_idx]
    def __len__(self):
        return self.df.shape[0]
    def _wav2fbank(self, filename, label=-1):
        # mixup
        waveform, sr = torchaudio.load(filename)
        waveform = waveform.cpu()
        if self.transform is not None and label == 1:
            waveform = waveform.unsqueeze(0)
            waveform = self.transform(waveform, sample_rate=sr)
            waveform = waveform.squeeze(0)
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.target_length
        n_frames = fbank.shape[0]
        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        uuid = item['uuid']
        label_encoded = item['assessment_result']
        #print(label)
        # label_encoded = np.zeros((label.size, 2))
        # label_encoded[np.arange(label.size),int(label)] = 1

        audio_path = os.path.join(self.audio_folder, "%s.wav"%item['uuid'])
#         audio, fs = librosa.load(audio_path)
#         max_samples = fs * self.period
        if self.for_test:
            fbank = self._wav2fbank(audio_path, -1)
        else:
            fbank = self._wav2fbank(audio_path, label_encoded)
        # if self.for_test:
        #     audio = padding_repeat(audio, max_samples)
        # else:
        #     audio = random_crop(audio, max_samples)
#         audio = padding_repeat(audio, max_samples)
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)


        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank, torch.tensor(label_encoded).float()

class AstTestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_params):
        super().__init__()
        self.df = None
        pprint(dataset_params)
        self.dataset_params = dataset_params
        self.audio_folder = dataset_params.get("audio_folder", "")
        self.period = dataset_params.get("period", 7)
        self.for_test = dataset_params.get("for_test", False)
        self.melbins = dataset_params.get("melbins", 128)
        self.freqm = dataset_params.get("freqm", 0)
        self.noise = dataset_params.get("noise", False)
        self.timem = dataset_params.get("timem", 192)
        self.target_length = dataset_params.get("target_length", 512)
        self.norm_mean = dataset_params.get("norm_mean", -6.0542064)
        self.norm_std = dataset_params.get("std", 6.219794)
        self.skip_norm = dataset_params.get("skip_norm", False)
        self._load_data()

    def get_label(self, idx):
        return self.df.iloc[idx]["assessment_result"].astype("float32")

    def _load_data(self):
        df_path = os.path.join(self.dataset_params.get("csv_path", ""))
        self.df = pd.read_csv(df_path)

    def __len__(self):
        return self.df.shape[0]

    def _wav2fbank(self, filename, label=-1):
        # mixup
        waveform, sr = torchaudio.load(filename)
        waveform = waveform.cpu()

        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.target_length
        n_frames = fbank.shape[0]
        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        uuid = item['uuid']
        #print(label)
        # label_encoded = np.zeros((label.size, 2))
        # label_encoded[np.arange(label.size),int(label)] = 1

        audio_path = os.path.join(self.audio_folder, "%s.wav"%item['uuid'])
#         audio, fs = librosa.load(audio_path)
#         max_samples = fs * self.period
        fbank = self._wav2fbank(audio_path, -1)
        # if self.for_test:
        #     audio = padding_repeat(audio, max_samples)
        # else:
        #     audio = random_crop(audio, max_samples)
#         audio = padding_repeat(audio, max_samples)
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)


        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank, uuid