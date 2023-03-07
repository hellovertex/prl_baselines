# traineval consider making class
import logging

import torch

from prl.baselines.supervised_learning.v2.action_prediction.nn_training.training_config import \
    TrainingParams, get_model
from prl.baselines.supervised_learning.v2.datasets.dataset_config import DatasetConfig
from prl.baselines.supervised_learning.v2.datasets.training_data import get_datasets

import logging
import multiprocessing
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
import pprint

target_names = ['Fold',
                'Check Call',
                'Raise Third Pot',
                'Raise Two Thirds Pot',
                'Raise Pot',
                'Raise 2x Pot',
                'Raise 3x Pot',
                'Raise All in']


def init_state(ckpt_dir, model, optim, resume: bool = True):
    start_n_iter = 0
    start_epoch = 0
    best_accuracy = -np.inf
    if resume:
        try:
            ckpt = torch.load(ckpt_dir + '/ckpt.pt')
            model.load_state_dict(ckpt['net'])
            start_epoch = ckpt['epoch']
            start_n_iter = ckpt['n_iter']
            best_accuracy = ckpt['best_accuracy']
            optim.load_state_dict(ckpt['optim'])
            print("last checkpoint restored")
        except Exception as e:
            # fail silently and start from scratch
            logging.info(f"Loading checkpoints failed with exception: {e}")
            logging.info(f"Continue Training from scratch")
    return {"start_n_iter": start_n_iter,
            "start_epoch": start_epoch,
            "best_accuracy": best_accuracy}


def _mp(dataset_config, params):
    TrainEval(dataset_config).run(params)


def mp(dataset_config, params_list):
    run_fn = partial(_mp,
                     dataset_config=dataset_config)
    # call pool.imap_unordered(run_fn, params_list)
    pass


class TrainEval:
    def __init__(self,
                 dataset_config: DatasetConfig):
        self.dataset_config = dataset_config

        # will be initialized lazily when `self.run` is called using `TrainingParams`
        self.device = None
        self.use_cuda = None
        self.logdir = None
        self.ckptdir = None
        self.writer = None
        self.model = None
        self.optim = None

    def initialize_training(self, params, hdims, lr):
        self.logdir = os.path.join(
            params.results_dir(self.dataset_config, hdims, lr),
            'logdir'
        )
        self.ckptdir = os.path.join(
            params.results_dir(self.dataset_config, hdims, lr),
            'ckptdir')
        self.model = get_model(input_dim=params.input_dim,
                               output_dim=params.output_dim,
                               hdims=hdims,
                               device=self.device)
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=lr)
        self.writer = SummaryWriter(log_dir=self.logdir)

    def train_step(self, x, y, label_weights):
        if self.use_cuda:  # keep
            x = x.cuda()
            y = y.cuda()
        # forward and backward pass
        self.optim.zero_grad()
        output = self.model(x)
        pred = torch.argmax(output,
                            dim=1)  # get the index of the max log-probability
        if label_weights is not None:
            loss = F.cross_entropy(output, y, weight=label_weights.to(self.device))
        else:
            loss = F.cross_entropy(output, y)
        loss.backward()
        self.optim.step()
        return pred, loss

    def run(self, params: TrainingParams):
        # 1. Make dataset, from scratch if necessary
        # (downloading, extracting, encoding, vectorizing, preprocessing, train/test
        # splitting)
        logging.info(f"Starting training with params {params}")
        self.device = params.device
        self.use_cuda = True if 'cuda' in self.device else False
        traindataset, testdataset, label_weights = get_datasets(self.dataset_config)
        train_dataloader = DataLoader(traindataset,
                                      batch_size=params.batch_size,
                                      shuffle=True)
        test_dataloader = DataLoader(testdataset,
                                     batch_size=params.batch_size,
                                     shuffle=True)
        len_data = round(len(train_dataloader))
        for hdims in params.hdims:
            for lr in params.lrs:
                self.initialize_training(params, hdims, lr)
                state_dict = init_state(self.ckptdir, self.model, self.optim)
                it_train_global = state_dict["start_n_iter"]
                it_train_curr = 0
                start_epoch = state_dict["start_epoch"]
                best_accuracy = state_dict["best_accuracy"]
                print('debugme')
                a = 1

                for epoch in range(start_epoch, params.max_epochs):
                    if (epoch * len(traindataset)) > params.max_env_steps:
                        break
                    pbar = tqdm(enumerate(BackgroundGenerator(train_dataloader)),
                                total=len_data)
                    pbar.set_description(
                        f'Training epoch {epoch}/{params.max_epochs} on {len(traindataset)} '
                        f'examples using batches of size {params.batch_size}...')
                    correct = 0
                    total_loss = 0
                    for i, (x, y) in pbar:
                        pred, loss = self.train_step(x, y, label_weights)
                        total_loss += loss.data.item()
                        correct += pred.eq(y.data).cpu().sum().item()
                        # ------------------------
                        # ------- LOGGING --------
                        # ------------------------
                        if it_train_curr % params.log_interval == 0:
                            f1 = f1_score(y.data.cpu(),
                                          pred.cpu(),
                                          average='weighted')
                            n_samples = it_train_curr * params.batch_size
                            # across all batches seen so far
                            self.writer.add_scalar(tag='Training Loss',
                                                   scalar_value=total_loss / n_samples,
                                                   global_step=it_train_global)
                            self.writer.add_scalar(tag='Training F1 score',
                                                   scalar_value=f1,
                                                   global_step=it_train_global)
                            self.writer.add_scalar(tag='Training Accuracy',
                                                   scalar_value=100.0 * correct / n_samples,
                                                   global_step=it_train_global)
                            print(f"\nTrain set: "
                                  f"Average loss: {round(total_loss / n_samples, 4)}, "
                                  f"Accuracy: {correct}/{it_train_global} "
                                  f"({round(100.0 * correct / n_samples, 2)}%)\n")
                            i_train = 0
                            correct = 0
                            total_loss = 0
                        # ------------------------
                        # ---- CHECKPOINTING -----
                        # ------------------------
                        # evaluate once (i==0) every epoch (j % eval_interval)
                        if it_train_curr % params.eval_interval == 0:
                            self.model.eval()
                            test_loss = 0
                            test_correct = 0
                            with torch.no_grad():
                                for x, y in BackgroundGenerator(test_dataloader):
                                    if self.use_cuda:
                                        x = x.cuda()
                                        y = y.cuda()
                                    output = self.model(x)
                                    # sum up batch loss
                                    test_loss += F.cross_entropy(
                                        output, y,
                                        reduction="sum").data.item()
                                    pred = torch.argmax(output, dim=1)
                                    f1 = f1_score(y.data.cpu(), pred.cpu(),
                                                  average='weighted')
                                    f1_0 = f1_score(y.data.cpu(), pred.cpu(),
                                                    labels=[0],
                                                    average='macro')
                                    f1_1 = f1_score(y.data.cpu(), pred.cpu(),
                                                    labels=[1],
                                                    average='macro')
                                    f1_2 = f1_score(y.data.cpu(), pred.cpu(),
                                                    labels=[2],
                                                    average='macro')
                                    f1_3 = f1_score(y.data.cpu(), pred.cpu(),
                                                    labels=[3],
                                                    average='macro')
                                    f1_4 = f1_score(y.data.cpu(), pred.cpu(),
                                                    labels=[4],
                                                    average='macro')
                                    f1_5 = f1_score(y.data.cpu(), pred.cpu(),
                                                    labels=[5],
                                                    average='macro')
                                    f1_6 = f1_score(y.data.cpu(), pred.cpu(),
                                                    labels=[6],
                                                    average='macro')
                                    f1_7 = f1_score(y.data.cpu(), pred.cpu(),
                                                    labels=[7],
                                                    average='macro')
                                    test_correct += pred.eq(y.data).cpu().sum().item()

                            test_loss /= len(testdataset)
                            test_accuracy = 100 * test_correct / len(testdataset)
                            print(
                                "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                                    test_loss, round(test_correct), len(testdataset),
                                    test_accuracy
                                )
                            )
                            if not os.path.exists(self.ckptdir):
                                os.makedirs(self.ckptdir)
                            if best_accuracy < test_accuracy:
                                best_accuracy = test_accuracy
                                torch.save({'epoch': epoch,
                                            'net': self.model.state_dict(),
                                            'n_iter': it_train_global,
                                            # j * batch_size * len(dataloader) == env_steps
                                            'optim': self.optim.state_dict(),
                                            'loss': loss,
                                            'best_accuracy': best_accuracy},
                                           self.ckptdir)  # net
                                # save model for inference
                                torch.save(self.model, self.ckptdir + '/model.pt')
                            else:
                                torch.save({'epoch': epoch,
                                            'net': self.model.state_dict(),
                                            'n_iter': it_train_global,
                                            # j * batch_size * len(dataloader) == env_steps
                                            'optim': self.optim.state_dict(),
                                            'loss': loss,
                                            'best_accuracy': best_accuracy},
                                           self.ckptdir + '/ckpt_tmp.pt')  # net
                                # save model for inference
                                torch.save(self.model, self.ckptdir + '/model_tmp.pt')
                            # return model to training mode
                            self.model.train()

                            # write metrics to tensorboard
                            self.writer.add_scalar(tag='Test Loss',
                                                   scalar_value=test_loss,
                                                   global_step=it_train_global)
                            self.writer.add_scalar(tag='Test F1/average', scalar_value=f1,
                                                   global_step=it_train_global)
                            self.writer.add_scalar(tag='Test F1 score/FOLD',
                                                   scalar_value=f1_0,
                                                   global_step=it_train_global)
                            self.writer.add_scalar(tag='Test F1 score/CHECK/CALL',
                                                   scalar_value=f1_1,
                                                   global_step=it_train_global)
                            self.writer.add_scalar(tag='Test F1 score/Raise Third Pot',
                                                   scalar_value=f1_2,
                                                   global_step=it_train_global)
                            self.writer.add_scalar(
                                tag='Test F1 score/Raise Two Thirds Pot',
                                scalar_value=f1_3,
                                global_step=it_train_global)
                            self.writer.add_scalar(tag='Test F1 score/Raise Pot',
                                                   scalar_value=f1_4,
                                                   global_step=it_train_global)
                            self.writer.add_scalar(tag='Test F1 score/Raise 2x Pot',
                                                   scalar_value=f1_5,
                                                   global_step=it_train_global)
                            self.writer.add_scalar(tag='Test F1 score/Raise 3x Pot',
                                                   scalar_value=f1_6,
                                                   global_step=it_train_global)
                            self.writer.add_scalar(tag='Test F1 score/Raise ALL IN',
                                                   scalar_value=f1_7,
                                                   global_step=it_train_global)
                            self.writer.add_scalar(tag='Test Accuracy',
                                                   scalar_value=test_accuracy,
                                                   global_step=it_train_global)

                            report = classification_report(y.cpu().numpy(),
                                                           pred.cpu().numpy(),
                                                           labels=[0, 1, 2, 3, 4, 5,
                                                                   6,
                                                                   7],
                                                           target_names=target_names,
                                                           output_dict=True)

                            for name, values in report.items():
                                if name in target_names:
                                    self.writer.add_scalar(f'precision/{name}',
                                                           values['precision'],
                                                           global_step=it_train_global)
                            for name, values in report.items():
                                if name in target_names:
                                    self.writer.add_scalar(f'recall/{name}',
                                                           values['recall'],
                                                           global_step=it_train_global)
                            pprint.pprint(report)
                            # write layer histograms to tensorboard
                            k = 1
                            for layer in self.model.children():
                                if isinstance(layer, nn.Linear):
                                    self.writer.add_histogram(f"layer{k}.weights",
                                                              layer.state_dict()[
                                                                  'weight'],
                                                              global_step=it_train_global)
                                    self.writer.add_histogram(f"layer{k}.bias",
                                                              layer.state_dict()['bias'],
                                                              global_step=it_train_global)
                                    k += 1

                        it_train_global += 1
                        it_train_curr += 1
