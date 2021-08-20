import torch
import os
import glob
import pandas as pd
import soundfile as sf
from PIL import Image
import numpy as np
import cv2
import librosa
from .audio_preprocessing import mfcc_feature, extract_mfcc_feature, padding_repeat, random_crop
from .mel_spec import audio2image, create_spectrogram

class CovidDataset(torch.utils.data.Dataset):
    def __init__(self, df, audio_folder, mfcc_config, for_test=False,  audio_transforms=None, image_transform=None):
        super().__init__()
        self.df = df
        self.audio_folder = audio_folder
        self.image_transform = image_transform
        self.audio_transforms = audio_transforms
        self.mfcc_config = mfcc_config
        if mfcc_config is None:
            self.mfcc_config = {}
        self.for_test = for_test
    def get_label(self, idx):
        return self.df.iloc[idx]["assessment_result"].astype("float32")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        label = item["assessment_result"].astype("float32")
        #print(label)
        # label_encoded = np.zeros((label.size, 2))
        # label_encoded[np.arange(label.size),int(label)] = 1

        label_encoded = torch.tensor(label, dtype=torch.float)

        audio_path = os.path.join(self.audio_folder, "%s.wav"%item['uuid'])
        audio, fs = librosa.load(audio_path)
        # audio, fs = sf.read(audio_path, dtype="float32")
        # # image = audio2image(audio, fs, self.audio_transforms)
        # image = mfcc_feature(audio, fs, self.audio_transforms)
        image = extract_mfcc_feature(audio, fs, self.mfcc_config, self.audio_transforms, for_test=self.for_test)
        # image = create_spectrogram(audio, fs, self.audio_transforms)
        if self.image_transform is not None:
            image = self.image_transform(image)

        return image, label_encoded

class CNN14_Dataset(torch.utils.data.Dataset):
    def __init__(self, fold_idx, dataset_params, transform=None):
        super().__init__()
        self.df = None
        self.fold_idx = fold_idx
        self.dataset_params = dataset_params
        self.audio_folder = dataset_params.get("audio_folder", "")
        self.period = dataset_params.get("period", 7)
        self.for_test = dataset_params.get("for_test", False)
        self.transform = transform
        self._load_data()

    def get_label(self, idx):
        return self.df.iloc[idx]["assessment_result"].astype("float32")

    def _load_data(self):
        df_path = os.path.join(self.dataset_params.get("csv_path", ""))
        df = pd.read_csv(df_path)
        if self.for_test:
            self.df = df[df["fold"] == self.fold_idx]
            try:
                self.df = self.df[self.df["is_augmented"] == 0]
            except:
                pass
        else:
            self.df = df[df["fold"] != self.fold_idx]
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        label = item["assessment_result"].astype("float32")
        uuid = item['uuid']
        #print(label)
        # label_encoded = np.zeros((label.size, 2))
        # label_encoded[np.arange(label.size),int(label)] = 1

        label_encoded = torch.tensor(label, dtype=torch.float)

        audio_path = os.path.join(self.audio_folder, "%s.wav"%item['uuid'])
        audio, fs = librosa.load(audio_path)
        
        # if self.transform is not None and label_encoded == 1:
        if self.transform is not None:
            try:
                audio, fs = self.transform(audio, fs)
            except:
                audio = self.transform(samples=audio, sample_rate=fs)

        max_samples = fs * self.period
        if self.for_test:
            audio = padding_repeat(audio, max_samples)
        else:
            audio = random_crop(audio, max_samples)
        return torch.from_numpy(audio).float(), label_encoded

class TestCNN14Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_params):
        super().__init__()
        self.df = None
        self.dataset_params = dataset_params
        self.audio_folder = dataset_params.get("audio_folder", "")
        self.period = dataset_params.get("period", 7)
        self.for_test = dataset_params.get("for_test", False)
        self._load_data()

    def get_label(self, idx):
        return self.df.iloc[idx]["assessment_result"].astype("float32")

    def _load_data(self):
        df_path = os.path.join(self.dataset_params.get("csv_path", ""))
        self.df = pd.read_csv(df_path)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        uuid = item['uuid']
        #print(label)
        # label_encoded = np.zeros((label.size, 2))
        # label_encoded[np.arange(label.size),int(label)] = 1

        audio_path = os.path.join(self.audio_folder, "%s.wav"%item['uuid'])
        audio, fs = librosa.load(audio_path)
        max_samples = fs * self.period
        # if self.for_test:
        #     audio = padding_repeat(audio, max_samples)
        # else:
        #     audio = random_crop(audio, max_samples)
        audio = padding_repeat(audio, max_samples)
        return torch.from_numpy(audio).float(), uuid

class TestDataset:
    def __init__(self, csv_path, audio_folder, mfcc_config=None, image_transform=None):
        self.audio_paths= glob.glob(os.path.join(audio_folder, "*.wav"))
        self.audio_folder = audio_folder
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path)
        self.image_transform = image_transform
        self.mfcc_config = mfcc_config
        if mfcc_config is None:
            self.mfcc_config = {}
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        uuid = item["uuid"]
        # audio_path = item["file_path"]
        audio_path = "%s.wav"%uuid
        audio_path = os.path.join(self.audio_folder, audio_path)
        # audio, fs = sf.read(audio_path, dtype="float32")
        audio, fs = librosa.load(audio_path)
        image = extract_mfcc_feature(audio, fs, self.mfcc_config, for_test=True)
        if self.image_transform is not None:
            image = self.image_transform(image=image)["image"]

        return torch.from_numpy(image), uuid

class Covid19StudyDataset(torch.utils.data.Dataset):
    def __init__(self, df, images_dir, transforms=None):
        self.df = df
        self.images_dir = images_dir
        self.transforms = transforms

    def __getitem__(self, idx):
        items = self.df.iloc[idx]
        image_id = str(items["id"]).split("_")[0] # remove image prefix
        label = items["label"]
        labels = torch.tensor(items[5:9].to_list())
        image_path = os.path.join(self.images_dir, "{}.jpg".format(image_id))
        image = cv2.imread(image_path)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, labels

    def __len__(self):
        return len(self.df)
