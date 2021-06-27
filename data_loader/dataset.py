import torch
import os
import glob
import pandas as pd
import soundfile as sf
from .audio_preprocessing import mfcc_feature

class CovidDataset(torch.utils.data.Dataset):
    def __init__(self, df, audio_folder, audio_transforms=None, image_transform=None):
        super().__init__()
        self.df = df
        self.audio_folder = audio_folder
        self.image_transform = image_transform
        self.audio_transforms = audio_transforms
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        label = item["assessment_result"].astype("float32")
        #print(label)
        # label_encoded = np.zeros((label.size, 2))
        # label_encoded[np.arange(label.size),int(label)] = 1

        label_encoded = torch.tensor(label, dtype=torch.float)

        audio_path = os.path.join(self.audio_folder, item['file_path'])
        audio, fs = sf.read(audio_path, dtype="float32")

        image = mfcc_feature(audio, fs, self.audio_transforms)
        if self.image_transform is not None:
            image = self.image_transform(image=image)["image"]

        return torch.from_numpy(image), label_encoded

class TestDataset:
    def __init__(self, csv_path, audio_folder, image_transform=None):
        self.audio_paths= glob.glob(os.path.join(audio_folder, "*.wav"))
        self.audio_folder = audio_folder
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path)
        self.image_transform = image_transform
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        uuid = item["uuid"]
        audio_path = item["file_path"]
        audio_path = os.path.join(self.audio_folder, audio_path)
        audio, fs = sf.read(audio_path, dtype="float32")
        image = mfcc_feature(audio, fs)
        if self.image_transform is not None:
            image = self.image_transform(image=image)["image"]

        return torch.from_numpy(image), uuid