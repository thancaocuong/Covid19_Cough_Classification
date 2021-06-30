import argparse
import collections
import os
import gc
import torch
import numpy as np
import pandas as pd
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from data_loader import CovidDataset, TestDataset
from data_loader import AudioCompose, WhiteNoise, TimeShift, ChangePitch, ChangeSpeed
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain, PolarityInversion, AddGaussianSNR
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import torchvision
from audiomentations.core.composition import BaseCompose

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
import random
random.seed(SEED)
class OneOf(BaseCompose):
    # TODO: Name can change to WaveformCompose
    def __init__(self, transforms, p=1.0, shuffle=False):
        super(OneOf, self).__init__(transforms, p, shuffle)
    def __call__(self, samples, sample_rate):
        transforms = self.transforms.copy()
        if random.random() < self.p:
            random.shuffle(transforms)
            for transform in transforms:
                samples = transform(samples, sample_rate)
                break

        return samples

    def randomize_parameters(self, samples, sample_rate):
        for transform in self.transforms:
            transform.randomize_parameters(samples, sample_rate)

def init_dataset(csv_path, fold_idx=1, audio_folder=""):
    print("*"*10, " fold {}".format(fold_idx), "*"*10)
    """StratifiedKFold"""
    df_path = os.path.join(csv_path)
    df = pd.read_csv(df_path)
    eval_df = df[df["fold"] == fold_idx]
    train_df = df[df["fold"] != fold_idx]
    # train_audio_transform = AudioCompose([WhiteNoise(0.005),
    #                                       TimeShift(),
    #                                       ChangeSpeed()])
    train_audio_transform = OneOf([
                    # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                    AddGaussianSNR(min_SNR=0.1, max_SNR=0.56, p=0.5), # new
                    TimeStretch(min_rate=0.9, max_rate=1.2, p=0.5),
                    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
                    PolarityInversion(p=0.5),
                    Gain()
                ], p=1.0)
    # train_audio_transform = None
    train_dataset = CovidDataset(
            df=train_df,
            audio_folder=audio_folder,
            audio_transforms=train_audio_transform,
            image_transform=None,
        )

    validation_dataset = CovidDataset(
                                df=eval_df,
                                audio_folder=audio_folder,
                                audio_transforms=None,
                                image_transform=None,
                            )
    return train_dataset, validation_dataset

def init_unlabeled_dataset(csv_path, audio_folder=""):
    return  TestDataset(csv_path, audio_folder)


def main(config, fold_idx):
    logger = config.get_logger('train')
    train_dataset, val_dataset = init_dataset(config["dataset"]["csv_path"],
                                             fold_idx,
                                             config["dataset"]["audio_folder"])
    # setup data_loader instances
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["dataset"]['training_batch_size'], 
        num_workers=config["dataset"]['num_workers'],
        shuffle=True,
        drop_last = True
    )
    eval_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["dataset"]['validate_batch_size'], 
        num_workers=config["dataset"]['num_workers']
    )
    unlabeled_loader = None
    if config["do_pseudo"]:
        unlabeled_dataset = init_unlabeled_dataset(config["unlabeled_dataset"]["csv_path"],
                                                config["unlabeled_dataset"]["audio_folder"]
                                                )
        unlabeled_loader = torch.utils.data.DataLoader(
                    unlabeled_dataset,
                    batch_size=config["unlabeled_dataset"]['training_batch_size'], 
                    num_workers=config["unlabeled_dataset"]['num_workers'],
                    shuffle=True,
                    drop_last = True
                    )
    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=train_loader,
                      valid_data_loader=eval_loader,
                      unlabeled_loader=unlabeled_loader,
                      lr_scheduler=lr_scheduler,
                      fold_idx=fold_idx
                      )

    trainer.train()
    model = model.to("cpu")
    del model, optimizer, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    for fold_idx in range(1, 6):
        main(config, fold_idx)
