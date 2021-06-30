import torch
import os
import glob
import pandas as pd
import soundfile as sf
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
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        label = item["assessment_result"].astype("float32")
        #print(label)
        # label_encoded = np.zeros((label.size, 2))
        # label_encoded[np.arange(label.size),int(label)] = 1

        label_encoded = torch.tensor(label, dtype=torch.float)

        audio_path = os.path.join(self.audio_folder, item['file_path'])
        audio, fs = sf.read(audio_path, dtype="float32")
        # # image = audio2image(audio, fs, self.audio_transforms)
        # image = mfcc_feature(audio, fs, self.audio_transforms)
        image = extract_mfcc_feature(audio, fs, self.mfcc_config, self.audio_transforms)
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
        image = mfcc_feature(audio, fs, self.mfcc_config)
        if self.image_transform is not None:
            image = self.image_transform(image=image)["image"]

        return torch.from_numpy(image), uuid