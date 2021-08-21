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
from data_loader import CovidDataset, CovidStackDataset
from data_loader import AudioCompose, WhiteNoise, TimeShift, ChangePitch, ChangeSpeed
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain, PolarityInversion, AddGaussianSNR, Normalize
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import torchvision
from audiomentations.core.composition import BaseCompose
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as albu
from tqdm import tqdm
import sklearn
from torch.optim.swa_utils import AveragedModel, SWALR
import random

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

def roc_auc(output, target):
    output = torch.sigmoid(output.float())
    output = output.numpy()
    target = target.numpy()
    return sklearn.metrics.roc_auc_score(target, output)

def cut_x3(df_train):

    df = pd.DataFrame({})
    for i in range(len(df_train)):
        item = df_train.iloc[i]
        uuid = item['uuid']
        if '_2' in uuid or '_3' in uuid:
        # if uuid[-2] == '_':
            continue
        df = df.append(item, ignore_index=True)
    return df


def cut_x5(df_train):

    df = pd.DataFrame({})
    for i in range(len(df_train)):
        item = df_train.iloc[i]
        uuid = item['uuid']
        if '_2' in uuid or '_3' in uuid or '_1' in uuid or '_0' in uuid:
            continue
        df = df.append(item, ignore_index=True)
    return df

def init_dataset(csv_path, config=None, fold_idx=1):
    print("*"*10, " fold {}".format(fold_idx), "*"*10)
    """StratifiedKFold"""
    # fold_idx = 1
    # df_final = pd.read_csv(csv_path)
    # eval_df = cut_x5(df_final)
    # eval_df = eval_df[eval_df['fold']==1]
    # config['dataset']['data_path'] = '/home/hana/sonnh/covidaivn/Covid19_Cough_Classification/saved/models/57-Covid19-Eff_b7/final_fold1.pickle'

    # * public
    df_test = pd.read_csv('/home/hana/sonnh/data/AICovidVN/aicv115m_public_train_full/aicv115m_final_public_test/public_test_sample_submission.csv')
    audio_dir = '/home/hana/sonnh/data/AICovidVN/aicv115m_public_train_full/aicv115m_final_public_test/public_test_audio_files/'
    df_test['audio_paths'] = ['{}/{}.wav'.format(audio_dir, uuid) for uuid in list(df_test['uuid'])]
    eval_df = df_test
    config['dataset']['data_path'] = '/home/hana/sonnh/covidaivn/Covid19_Cough_Classification/saved/models/57-Covid19-Eff_b7/public_test_fold1.pickle'
    
    validation_dataset = CovidStackDataset(
                                df=eval_df,
                                config=config,
                            )

    return validation_dataset

def main(config, fold_idx):
    val_dataset = init_dataset(config["dataset"]["csv_path"],
                                              config,  
                                             fold_idx)

    eval_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["dataset"]['validate_batch_size'], 
        # batch_size=1, 
        num_workers=config["dataset"]['num_workers'], 
        shuffle = False
    )
    
    checkpoint_paths = [
        'saved/models/57_stack/0819_194719/model_best_fold1',
        'saved/models/57_stack/0819_194719/model_best_fold2',
        'saved/models/57_stack/0819_194719/model_best_fold3',
        'saved/models/57_stack/0819_194719/model_best_fold4',
        'saved/models/57_stack/0819_194719/model_best_fold5'
        ]
    final_outputs = []
    for checkpoint_path in checkpoint_paths:
        model = config.init_obj('arch', module_arch)
        check_point = torch.load(checkpoint_path)
        
        #* normal
        model.load_state_dict(check_point['state_dict'])

        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(config['n_gpu'])
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        targets = []
        outputs = []
        model.eval()
        with torch.no_grad():
            # for batch_idx, (data, target) in enumerate(tqdm(eval_loader)):
            for batch_idx, info in enumerate(tqdm(eval_loader)):
                data = info['data']
                target = info['target']
                uuid  = info['uuid']

                data, target = data.to(device), target.to(device)
                target = target.to(dtype=torch.long)
                output = model(data)
                
                targets.append(target.detach().cpu())
                outputs.append(output.detach().cpu())
        targets = torch.cat(targets)
        outputs = torch.cat(outputs)

        print(targets.size())
        print(outputs.size())
        targets = targets.detach().cpu().numpy()
        outputs = torch.sigmoid(outputs)[:, 0].tolist()


        # print('auc: {}'.format(sklearn.metrics.roc_auc_score(targets, outputs)))

        model = model.to("cpu")
        del model, 

        gc.collect()
        torch.cuda.empty_cache()

        final_outputs.append(outputs)
    final_outputs = np.sum(final_outputs, axis = 0)

    df_test = pd.read_csv('/home/hana/sonnh/data/AICovidVN/aicv115m_public_train_full/aicv115m_final_public_test/public_test_sample_submission.csv')
    df_test['assessment_result'] = final_outputs
    df_test.to_csv('/home/hana/sonnh/covidaivn/covid19_res/57_ens_stack_top10auc_.csv', index = False)


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
    fold_idx = 1
    main(config, fold_idx)
