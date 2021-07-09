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
from data_loader import CovidDataset, TestDataset, Covid19StudyDataset
from data_loader import AudioCompose, WhiteNoise, TimeShift, ChangePitch, ChangeSpeed
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import torchvision
from sklearn.utils import shuffle

#aug
import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2
# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
import random
random.seed(SEED)


def init_dataset(csv_path, fold_idx=1, images_dir="", input_size=512):
    print("*"*10, " fold {}".format(fold_idx), "*"*10)
    """StratifiedKFold"""
    df_path = os.path.join(csv_path)
    df = pd.read_csv(df_path)
    eval_df = df[df["kfold"] == fold_idx]
    train_df = df[df["kfold"] != fold_idx]
    train_df = shuffle(train_df)
    eval_df = shuffle(eval_df)

    # train_transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
    #                                                 torchvision.transforms.Resize(input_size),
    #                                                 torchvision.transforms.RandomHorizontalFlip(p=0.5),
    #                                                 torchvision.transforms.RandomRotation(45),
    #                                                 torchvision.transforms.RandomCrop(input_size),
    #                                                 torchvision.transforms.ToTensor(),
    #                                                 torchvision.transforms.Normalize((0.485, 0.456, 0.406),
    #                                                                                  (0.229, 0.224, 0.225))])

    # eval_transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
    #                                                 torchvision.transforms.Resize((input_size, input_size)),
    #                                                 torchvision.transforms.ToTensor(),
    #                                                 torchvision.transforms.Normalize((0.485, 0.456, 0.406),
    #                                                                                  (0.229, 0.224, 0.225))])


    train_transforms = A.Compose([

        A.Resize(input_size, input_size, cv2.INTER_AREA),
        A.OneOf([
            A.GaussNoise(var_limit=(150.0, 200.0), mean=0, p=0.5),
        ], p = 0.5),

        A.RandomGamma(gamma_limit=(120, 120), p=0.5),
        A.RandomBrightnessContrast(contrast_limit=0, brightness_limit=0.2, brightness_by_max=True, p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
        ], p = 0.5),

        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)

    eval_transforms = A.Compose([
        A.Resize(input_size, input_size, cv2.INTER_AREA),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)


    # train_audio_transform = None
    train_dataset = Covid19StudyDataset(
            df=train_df,
            images_dir=images_dir,
            transforms=train_transforms,
        )

    validation_dataset = Covid19StudyDataset(
            df=eval_df,
            images_dir=images_dir,
            transforms=eval_transforms,
        )
    return train_dataset, validation_dataset

def init_unlabeled_dataset(csv_path, audio_folder="", mfcc_config=None):
    return  TestDataset(csv_path, audio_folder, mfcc_config)


def main(config, fold_idx):
    logger = config.get_logger('train')
    train_dataset, val_dataset = init_dataset(config["dataset"]["csv_path"],
                                              fold_idx,
                                              config["dataset"]["images_dir"],
                                              config["dataset"]["input_size"])
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
                      unlabeled_loader=None,
                      lr_scheduler=lr_scheduler,
                      fold_idx=fold_idx,
                      warmup=config["trainer"]["warmup"]
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
    for fold_idx in range(1, 5):
        main(config, fold_idx)
