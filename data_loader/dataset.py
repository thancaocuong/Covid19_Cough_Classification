import torch
import os
import glob
import pandas as pd
import soundfile as sf
from PIL import Image
import numpy as np
import cv2
from .audio_preprocessing import mfcc_feature, extract_mfcc_feature
from .mel_spec import audio2image, create_spectrogram

class CovidDataset(torch.utils.data.Dataset):
    def __init__(self, df, audio_folder, mfcc_config, audio_transforms=None, image_transform=None):
        super().__init__()
        self.df = df
        self.audio_folder = audio_folder
        self.image_transform = image_transform
        self.audio_transforms = audio_transforms
        self.mfcc_config = mfcc_config
        if mfcc_config is None:
            self.mfcc_config = {}

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
        audio, fs = sf.read(audio_path, dtype="float32")
        # # image = audio2image(audio, fs, self.audio_transforms)
        # image = mfcc_feature(audio, fs, self.audio_transforms)
        image = extract_mfcc_feature(audio, fs, self.mfcc_config, self.audio_transforms, for_test=False)
        # image = create_spectrogram(audio, fs, self.audio_transforms)
        if self.image_transform is not None:
            image = self.image_transform(image)

        return image, label_encoded

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
        audio_path = item["file_path"]
        audio_path = os.path.join(self.audio_folder, audio_path)
        audio, fs = sf.read(audio_path, dtype="float32")
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
