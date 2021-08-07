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
from data_loader import CovidDataset, TestDataset, ImbalancedDatasetSampler
from data_loader import AudioCompose, WhiteNoise, TimeShift, ChangePitch, ChangeSpeed
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain, PolarityInversion, AddGaussianSNR
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import torchvision
from audiomentations.core.composition import BaseCompose
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as albu

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

def init_dataset(csv_path, config=None, fold_idx=1, audio_folder=""):
    print("*"*10, " fold {}".format(fold_idx), "*"*10)
    """StratifiedKFold"""
    df_path = os.path.join(csv_path)
    df = pd.read_csv(df_path)
    df_final = pd.read_csv('/home/hana/sonnh/data/AICovidVN/aicv115m_public_train_full/aicv115m_final_public_train/kfold_multi_feature_oversampling_new_format.csv')
    # df_external = pd.read_csv('/home/hana/sonnh/data/AICovidVN/external.csv')
    # df_external = df_external[df_external['assessment_result']==1]

    df_coswara = pd.read_csv('/home/hana/sonnh/data/AICovidVN/coswara/info_kfold_new_format.csv')
    df_vifury = pd.read_csv('/home/hana/sonnh/data/AICovidVN/virufy-cdf-coughvid/fillter_rm_slient_kfold_new_format.csv')
    df_vifury = df_vifury[df_vifury['assessment_result']==1]

    df_external = pd.concat([df_coswara, df_vifury])

    eval_df = df_external[df_external["fold"] == fold_idx]
    train_df_ = df_external[df_external["fold"] != fold_idx]

    train_df = pd.concat((df_final, train_df_))
    # train_audio_transform = AudioCompose([WhiteNoise(0.005),
    #                                       TimeShift(),
    #                                       ChangeSpeed()])
    # train_audio_transform = OneOf([
    #                 AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.05, p=1),
    #                 AddGaussianSNR(min_SNR=0.1, max_SNR=0.56, p=1), # new
    #                 TimeStretch(min_rate=0.9, max_rate=1.2, p=1),
    #                 PitchShift(min_semitones=-4, max_semitones=4, p=1),
    #                 Shift(min_fraction=-0.5, max_fraction=0.5, p=1),
    #                 PolarityInversion(p = 1),
    #                 Gain(min_gain_in_db=-15,max_gain_in_db=15,p=1)
    #             ], p=1.0)

    # train_audio_transform = Compose([
    #     # AddBackgroundNoise( # mixup audio level
    #     #     sounds_path = '/media/sonnh/AICOVIDVN/aicv115m_public_train/mixup',  # background audio folder không có tiếng ho
    #     #     min_snr_in_db=5,  #Minimum signal-to-noise ratio in dB, tỉ lệ db giữa tiếng gốc và tiếng mix
    #     #     max_snr_in_db=30,
    #     #     p=0.5
    #     # ),
        
    #     OneOf([
    #         AddGaussianNoise(
    #             min_amplitude=0.01, 
    #             max_amplitude=0.05, 
    #             p=0.9),
    #         AddGaussianSNR(
    #             min_snr_in_db=10, #Minimum signal-to-noise ratio in db. A lower number means more noise.
    #             max_snr_in_db=20, 
    #             p=0.9),
    #     ], p=1.0),
        
    #     OneOf([
    #         TimeStretch(
    #             min_rate=0.8, 
    #             max_rate=1.2, 
    #             p=0.9), #nén hoặc kéo gián audio
    #         Shift(
    #             min_fraction=0.3, 
    #             max_fraction=0.7, 
    #             p=0.9), # dich chuyển audio , đầu đuôi nối nhau 
    #         PitchShift(min_semitones=-4, max_semitones=4, p=0.9),
    #         Gain(min_gain_in_db=-15,max_gain_in_db=15,p=0.9)
    #     ], p=1.0),
        
        # Normalize(p = 1)
    # ])

    # train_audio_transform = None
    train_audio_transform = OneOf([
                    # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                    AddGaussianSNR(p=0.5), # new
                    TimeStretch(min_rate=0.9, max_rate=1.2, p=0.5),
                    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
                    PolarityInversion(p=0.5),
                    Gain()
                ], p=1.0)

    # image_transform = albu.Compose([
    #         ToTensorV2(p=1.0),
    #     ], p=1.)
    train_dataset = CovidDataset(
            df=train_df,
            config=config,
            audio_folder=audio_folder,
            audio_transforms=train_audio_transform,
            image_transform=None,
        )
    # eval_df = pd.read_csv('/home/hana/sonnh/data/AICovidVN/coswara/info.csv')
    # audio_folder = '/home/hana/sonnh/data/AICovidVN/coswara/audio/'
    validation_dataset = CovidDataset(
                                df=eval_df,
                                config=config,
                                audio_folder=audio_folder,
                                test = True,
                                audio_transforms=None,
                                image_transform=None,
                            )

    eval_df = pd.read_csv('/home/hana/sonnh/data/AICovidVN/aicv115m_public_train/metadata_train_challenge.csv')
    audio_folder = '/home/hana/sonnh/data/AICovidVN/aicv115m_public_train/train_audio_files_8k/'
    validation_dataset_warmup = CovidDataset(
                                df=eval_df,
                                config=config,
                                audio_folder=audio_folder,
                                test = True,
                                audio_transforms=None,
                                image_transform=None,
                            )


    # train_dataset = VitDataset(
    #     df=train_df[:200],
    #     audio_folder=audio_folder,
    #     train_aug=train_audio_transform,
    # )

    # validation_dataset = VitDataset(

    #     df=eval_df[:200],
    #     audio_folder=audio_folder,
    #     train_aug=None,
    # )
    return train_dataset, validation_dataset, validation_dataset_warmup


def init_unlabeled_dataset(csv_path, audio_folder="", mfcc_config=None):
    return  TestDataset(csv_path, audio_folder, mfcc_config)


def main(config, fold_idx):
    logger = config.get_logger('train')
    train_dataset, val_dataset, validation_dataset_warmup = init_dataset(config["dataset"]["csv_path"],
                                              config,  
                                             fold_idx,
                                             config["dataset"]["audio_folder"])
    # setup data_loader instances
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle = True,
        batch_size=config["dataset"]['training_batch_size'],
        num_workers=config["dataset"]['num_workers'],
        # sampler=ImbalancedDatasetSampler(train_dataset),
        drop_last = True
    )
    eval_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["dataset"]['validate_batch_size'], 
        num_workers=config["dataset"]['num_workers']
    )
    eval_loader_warmup = torch.utils.data.DataLoader(  
        validation_dataset_warmup,
        batch_size=config["dataset"]['validate_batch_size'], 
        num_workers=config["dataset"]['num_workers']
    )

    unlabeled_loader = None
    # if config["do_pseudo"]:
    #     unlabeled_dataset = init_unlabeled_dataset(config["unlabeled_dataset"]["csv_path"],
    #                                             config["unlabeled_dataset"]["audio_folder"],
    #                                             config["dataset"]["mfcc_config"],
    #                                             )
    #     unlabeled_loader = torch.utils.data.DataLoader(
    #                 unlabeled_dataset,
    #                 batch_size=config["unlabeled_dataset"]['training_batch_size'], 
    #                 num_workers=config["unlabeled_dataset"]['num_workers'],
    #                 shuffle=True,
    #                 drop_last = True
    #                 )
    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    logger.info(config)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    train_criterion = getattr(module_loss, config['train_loss'])
    val_criterion = getattr(module_loss, config['val_loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, train_criterion, val_criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=train_loader,
                      valid_data_loader=eval_loader,
                      valid_data_loader_warmup = eval_loader_warmup,
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
