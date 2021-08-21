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
from data_loader import CovidDataset, CovidStackTestDataset, CovidStackDataset
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
import pickle
import copy

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

def init_dataset(csv_path, config=None, fold_idx=1, audio_folder=""):
    print("*"*10, " fold {}".format(fold_idx), "*"*10)

    df_final = pd.read_csv(csv_path)
    eval_df = df_final[df_final['fold']==fold_idx]

    validation_dataset = CovidStackDataset(
            df=eval_df,
            config=config,
        )
    return validation_dataset

def init_test_dataset(csv_path, config=None, fold_idx=1, audio_folder=""):
    print("*"*10, " fold {}".format(fold_idx), "*"*10)

    # * public
    df_test = pd.read_csv('/home/hana/sonnh/data/AICovidVN/aicv115m_public_train_full/aicv115m_final_public_test/public_test_sample_submission.csv')
    audio_dir = '/home/hana/sonnh/data/AICovidVN/aicv115m_public_train_full/aicv115m_final_public_test/public_test_audio_files/'
    df_test['audio_paths'] = ['{}/{}.wav'.format(audio_dir, uuid) for uuid in list(df_test['uuid'])]
    eval_df = df_test
    config_ = copy.deepcopy(config)
    config_['dataset']['data_path'] = '/home/hana/sonnh/covidaivn/stack_feature/data_all_public_test.pkl'


    validation_dataset = CovidStackTestDataset(
            df=eval_df,
            config=config_,
            fold = fold_idx
        )

    return validation_dataset

def main(config, fold_idx):
    val_dataset = init_dataset(config["dataset"]["csv_path"],
                                              config,  
                                             fold_idx)

    test_dataset = init_test_dataset(config["dataset"]["csv_path"],
                                              config,  
                                             fold_idx)                               

    eval_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["dataset"]['validate_batch_size'], 
        # batch_size=1, 
        num_workers=config["dataset"]['num_workers'], 
        shuffle = False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["dataset"]['validate_batch_size'], 
        # batch_size=1, 
        num_workers=config["dataset"]['num_workers'], 
        shuffle = False
    )

    checkpoint_paths = []
    # checkpoint_path = '/media/sonnh/AICOVIDVN/saved1/models/64-Covid19-Eff_b5/check_point/checkpoint_{}_fold{}.pth'
    # checkpoint_path = '/home/hana/sonnh/covidaivn/Covid19_Cough_Classification/saved/models/59-Covid19-CNN14/checkpoint/checkpoint_{}_fold{}.pth'
    checkpoint_path = 'saved/models/all_stack_3/0821_180901//checkpoint_{}_fold{}.pth'
    top_n_model = 10
    df = pd.read_csv('saved/models/all_stack_3/0821_180901//result_fold{}.csv'.format(fold_idx))
    top_auc = df.sort_values(by='val_roc_auc', ascending=False)
    top10_auc = list(top_auc['epoch'][:top_n_model])
    for epoch in top10_auc:
        checkpoint_path_temp = checkpoint_path.format(epoch, fold_idx)
        checkpoint_paths.append(checkpoint_path_temp)


    for epoch in df['epoch']:
        filename = checkpoint_path.format(epoch, fold_idx)
        if os.path.isfile(filename):
            if epoch not in top10_auc:
                os.remove(filename)

    features = {}
    test_features = {}
    for checkpoint_path in checkpoint_paths:
        model = config.init_obj('arch', module_arch)
        check_point = torch.load(checkpoint_path)

        #* swa
        # model = AveragedModel(model)
        # model.load_state_dict(check_point['state_dict'])
        
        #* normal
        model.load_state_dict(check_point['state_dict'])

        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(config['n_gpu'])
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        # * swa
        # torch.optim.swa_utils.update_bn(train_loader, model, device=torch.device('cuda')) 

        targets = []
        outputs = []
        uuids = []
        model.eval()
        with torch.no_grad():
            # for batch_idx, (data, target) in enumerate(tqdm(eval_loader)):
            for batch_idx, info in enumerate(tqdm(eval_loader)):
                data = info['data']
                target = info['target']
                uuid  = info['uuid']
                uuids = uuids + uuid
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
        print('auc: {}'.format(sklearn.metrics.roc_auc_score(targets, outputs)))

        for i in range(len(uuids)):
            uuid_ = uuids[i]
            prob = float(outputs[i])
            label = float(targets[i])

            # print(prob, label)

            if uuid_ not in features:
                features[uuid_] = {
                    'label': label,
                    'feature': []
                }
            features[uuid_]['feature'].append(prob)

        targets = []
        outputs = []
        uuids = [] 
        with torch.no_grad():
            for batch_idx, info in enumerate(tqdm(test_loader)):
                data = info['data']
                target = info['target']
                uuid  = info['uuid']
                uuids = uuids + uuid
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
        for i in range(len(uuids)):
            uuid_ = uuids[i]
            prob = float(outputs[i])
            label = float(targets[i])

            if uuid_ not in test_features:
                test_features[uuid_] = {
                    'label': label,
                    'feature': []
                }
            test_features[uuid_]['feature'].append(prob)

        model = model.to("cpu")
        del model, 

        gc.collect()
        torch.cuda.empty_cache()

    
    with open('/home/hana/sonnh/covidaivn/stack_res/3/final_fold{}.pickle'.format(fold_idx), 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('/home/hana/sonnh/covidaivn/stack_res/3/public_test_fold{}.pickle'.format(fold_idx), 'wb') as handle:
        pickle.dump(test_features, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
    for i in range(1, 6):
        main(config, i)
