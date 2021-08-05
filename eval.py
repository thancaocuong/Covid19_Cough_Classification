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
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain, PolarityInversion
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import torchvision
from tqdm import tqdm

def post_process_function(output_tensor):
    outputs = []
    for output in output_tensor:
        outputs.append(float(output))
    max_prob = max(outputs)
    min_prob = min(outputs)
    sum_prob = sum(outputs)
    fussion_prob = (1/len(outputs))*((sum_prob - len(outputs)*min_prob)/ (max_prob - min_prob))
    return fussion_prob

class EnsembleModel(torch.nn.Module):
    def __init__(self, models, post_process_func=None, weights=[0.2, 0.2, 0.2, 0.2, 0.2]):
        super().__init__()
        self.models = models
        self.post_process_func = post_process_func
        self.weights = weights
    def forward(self, input, post_process_func=None):
        outputs = []
        for i in range(len(self.models)):
            output = self.models[i](input)
            outputs.append(torch.sigmoid(output).detach().cpu().item())
        return outputs
        # if self.post_process_func is not None:
        #     return self.post_process_func(outputs)
        # return torch.sum(torch.stack(outputs))

def init_unlabeled_dataset(csv_path, audio_folder="", mfcc_config=None):
    return  TestDataset(csv_path, audio_folder, mfcc_config=mfcc_config)

def main(config, args):
    unlabeled_dataset = init_unlabeled_dataset(config["unlabeled_dataset"]["csv_path"],
                                                config["unlabeled_dataset"]["audio_folder"],
                                                mfcc_config=config["dataset"]["mfcc_config"]
                                                )
    device, device_ids = prepare_device(config['n_gpu'])
    model_dir = args.model_dir

    CHECKPOINT = [os.path.join(model_dir, "model_best_fold1"),
                  os.path.join(model_dir, "model_best_fold2"),
                  os.path.join(model_dir, "model_best_fold3"),
                  os.path.join(model_dir, "model_best_fold4"),
                  os.path.join(model_dir, "model_best_fold5"),
                 ]
    models = []
    for i in range(len(CHECKPOINT)):
        state = torch.load(CHECKPOINT[i])["state_dict"]
        model = config.init_obj('arch', module_arch)
        model.load_state_dict(state)
        model = model.to(device)
        models.append(model.eval())
    ensemble_model = EnsembleModel(models, None, [1])
    ensemble_model.cuda().eval()
    results = [pd.DataFrame(columns=['uuid', 'assessment_result']) for i in range(len(models))]
    progress_bar = tqdm(total=len(unlabeled_dataset))
    for i in range(len(unlabeled_dataset)):
        feature, id = unlabeled_dataset[i]
        with torch.no_grad():
            feature = feature[None, ...].cuda()
            scores = ensemble_model(feature)
            for idx, score_fold in enumerate(scores):
                results[idx].loc[i] = [id, score_fold]

        progress_bar.update(1)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    for idx, pd_results in enumerate(results):
        fold_dir = os.path.join(args.save_dir, "fold{}".format(idx+1))
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        result_path = os.path.join(fold_dir, "results.csv")
        pd_results.to_csv(result_path, index=False)
    del models, ensemble_model
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
    args.add_argument("--model_dir", default="saved/models/Covid19-PlainSmallCNN-RandomCrop/0803_134129/",
                      type=str, help="model directory")
    args.add_argument("--save_dir", default="OOF_results",
                      type=str, help="saved fold results")

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    parsered_args = args.parse_args()
    config = ConfigParser.from_args(args, options)
    main(config, parsered_args)