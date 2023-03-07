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


def make_logs(params,
              writer,
              i_train,
              total_loss,
              correct,
              global_step,
              scalar_value):
    n_batch = i_train * params.batch_size  # how many samples across all batches seen so
    # far
    writer.add_scalar(tag='Training Loss',
                      scalar_value=total_loss / i_train,
                      global_step=global_step)
    writer.add_scalar(tag='Training F1 score', scalar_value=scalar_value,
                      global_step=global_step)
    writer.add_scalar(tag='Training Accuracy',
                      scalar_value=100.0 * correct / n_batch,
                      global_step=global_step)
    print(f"\nTrain set: "
          f"Average loss: {round(total_loss / i_train, 4)}, "
          f"Accuracy: {correct}/{n_batch} ({round(100.0 * correct / n_batch, 2)}%)\n")
    i_train = 0
    correct = 0
    total_loss = 0


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
            loss = F.cross_entropy(output, y, weight=label_weights.to(device))
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

        for hdims in params.hdims:
            for lr in params.lrs:
                self.initialize_training(params, hdims, lr)
                state_dict = init_state(self.ckptdir, self.model, self.optim)
                it = state_dict["start_n_iter"]
                start_epoch = state_dict["start_epoch"]
                best_accuracy = state_dict["best_accuracy"]

                len_data = round(len(train_dataloader))
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
                    i_train = 0
                    for i, (x, y) in pbar:
                        pred, loss = self.train_step(x, y, label_weights)
                        i_train += 1
                        total_loss += loss.data.item()  # add batch loss
                        # udpate tensorboardX
                        correct += pred.eq(y.data).cpu().sum().item()
                        if i % params.log_interval == 0:
                            make_logs()
