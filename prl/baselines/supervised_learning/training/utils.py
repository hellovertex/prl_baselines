import logging

import numpy as np
import torch
from prl.environment.Wrappers.base import ActionSpace
from tianshou.utils.net.common import MLP
from torch import nn
from torch.utils.data import random_split

from prl.baselines.supervised_learning.training.dataset import InMemoryDataset


def init_state(ckpt_dir, model, optim, resume: bool = True):
    # # load checkpoint if needed/ wanted

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
    # return start_n_iter, start_epoch, best_accuracy


def get_model(traindata, hidden_dims, device):
    # network
    classes = [ActionSpace.FOLD,
               ActionSpace.CHECK_CALL,  # CHECK IS INCLUDED
               ActionSpace.RAISE_MIN_OR_3BB,
               ActionSpace.RAISE_HALF_POT,
               ActionSpace.RAISE_POT,
               ActionSpace.ALL_IN]

    output_dim = len(classes)
    input_dim = None
    # waste the first batch to dynamically get the input dimension
    for x, y in traindata:
        input_dim = x.shape[1]
        break

    net = MLP(input_dim=input_dim,
              output_dim=output_dim,
              hidden_sizes=hidden_dims,
              norm_layer=None,
              activation=nn.ReLU,
              device=device,
              linear_layer=nn.Linear,
              flatten_input=False)
    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
    return net


def get_in_mem_datasets(input_dir, seed=1):
    dataset = InMemoryDataset(input_dir)
    total_len = len(dataset)
    train_len = int(total_len * 0.89)
    test_len = int(total_len * 0.1)
    val_len = int(total_len * 0.01)
    # add residuals to val_len to add up to total_len
    val_len += total_len - (int(train_len) + int(test_len) + int(val_len))
    # set seed
    gen = torch.Generator().manual_seed(seed)
    train, test, val = random_split(dataset, [train_len, test_len, val_len], generator=gen)

    return train, test
