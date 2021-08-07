import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from tqdm import tqdm
from .ema import  ModelEMA

from torch.cuda import amp
# from trainer.mix import mixup

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, train_criterion, val_criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, valid_data_loader_warmup = None, unlabeled_loader=None,
                 lr_scheduler=None, len_epoch=None, fold_idx=0, warmup=0):
        super().__init__(model, train_criterion, val_criterion, metric_ftns, optimizer, config, fold_idx, warmup)
        self.config = config
        self.device = device

        self.fp16 = self.config['fp16']
        self.scaler = amp.GradScaler(enabled=self.fp16)

        if self.config['ema']:
            self.ema =  ModelEMA(self.model)
        else:
            self.ema = None

        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.valid_data_loader_warmup = valid_data_loader_warmup
        self.unlabeled_loader = unlabeled_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.log_step = int(len(data_loader) // 5)
        self.semi_epochs = 20
        self.do_pseudo = self.unlabeled_loader is not None
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
    def _semi_train_epoch(self):
        if self.do_pseudo:
            for epoch_idx in range(self.semi_epochs):
                print("Semi train epoch", epoch_idx)
                print("total batch :", len(self.unlabeled_loader))
                for batch_idx, (data, _) in enumerate(self.unlabeled_loader):
                    self.optimizer.zero_grad()
                    data = data.to(self.device)
                    self.model.eval()
                    with torch.no_grad():
                        output_unlabeled = self.model(data)
                        output_unlabeled[output_unlabeled>=0.5] = 1.0
                        output_unlabeled[output_unlabeled<0.5] = 0.0
                    self.model.train()
                    output = self.model(data)
                    loss = self.train_criterion(output, output_unlabeled)
                    loss.backward()
                    self.optimizer.step()
                    if batch_idx % self.log_step == 0:
                        self.logger.debug('Semi Train Epoch: {} Loss: {:.6f}'.format(
                            epoch_idx,
                            loss.item()))
                    if (batch_idx % 5) == 0:
                        self._train_epoch(self.epochs+1+epoch_idx)
            log = self.train_metrics.result()
            if self.do_validation:
                val_log = self._valid_epoch(self.epochs+1+epoch_idx)
                log.update(**{'val_'+k : v for k, v in val_log.items()})
            return log
        else:
            return False
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        targets = []
        outputs = []
        self.optimizer.zero_grad()
        for batch_idx, (data, target) in enumerate(tqdm(self.data_loader)):
            data, target = data.to(self.device), target.to(self.device)
            with amp.autocast(enabled=self.fp16):

                output = self.model(data, fp16 = self.fp16)
                loss = self.train_criterion(output, target)
                # loss.backward()
                # self.optimizer.step()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad()

            targets.append(target.detach().cpu())
            outputs.append(output.detach().cpu())
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
            if batch_idx == self.len_epoch:
                break


            if self.ema:
                self.ema.update(self.model)

        if self.ema:
            self.ema.update_attr(self.model, include=[])

        targets = torch.cat(targets)
        outputs = torch.cat(outputs)
        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(outputs, targets))
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch, self.valid_data_loader)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

            val_log = self._valid_epoch(epoch, self.valid_data_loader_warmup)
            log.update(**{'val_warmup_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch, valid_data_loader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        if self.ema is not None:
            self.model_eval = self.ema.ema
        else:
            self.model_eval  = self.model

        
        # state_dict_path = '/home/hana/sonnh/Covid19_Cough_Classification/saved/models/13-Covid19-PlainCNNSmall/0803_233415/checkpoint_{}_fold{}.pth'.format(epoch, self.fold_idx)
        # state_dict = torch.load(state_dict_path)['state_dict']
        # self.model_eval.load_state_dict(state_dict)

        self.model_eval.eval()
        self.valid_metrics.reset()
        targets = []
        outputs = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(valid_data_loader)):
                data, target = data.to(self.device), target.to(self.device)
                target = target.to(dtype=torch.long)
                output = self.model_eval(data)
                loss = self.val_criterion(output, target)
                targets.append(target.detach().cpu())
                outputs.append(output.detach().cpu())
                self.writer.set_step((epoch - 1) * len(valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
            targets = torch.cat(targets)
            outputs = torch.cat(outputs)
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outputs, targets))
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
