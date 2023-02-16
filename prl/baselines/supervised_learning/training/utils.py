import glob
import logging
import math

import numpy as np
import pandas as pd
import torch
from prl.environment.Wrappers.base import ActionSpace
from tianshou.utils.net.common import MLP
from torch import nn
from torch.utils.data import random_split

from prl.baselines.supervised_learning.training.dataset import InMemoryDataset, OutOfMemoryDatasetV2


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


def get_model(traindata, hidden_dims, device, merge_labels567=False):
    # network
    classes = [ActionSpace.FOLD,
               ActionSpace.CHECK_CALL,  # CHECK IS INCLUDED in CHECK_CALL
               ActionSpace.RAISE_MIN_OR_3BB,
               ActionSpace.RAISE_6_BB,
               ActionSpace.RAISE_10_BB,
               ActionSpace.RAISE_20_BB,
               ActionSpace.RAISE_50_BB,
               ActionSpace.RAISE_ALL_IN]
    if merge_labels567:
        classes = [ActionSpace.FOLD,
                   ActionSpace.CHECK_CALL,  # CHECK IS INCLUDED in CHECK_CALL
                   ActionSpace.RAISE_MIN_OR_3BB,
                   ActionSpace.RAISE_6_BB,
                   ActionSpace.RAISE_10_BB,
                   ActionSpace.RAISE_20_BB]
                   # ActionSpace.RAISE_50_BB,
                   # ActionSpace.RAISE_ALL_IN]
    output_dim = len(classes)
    input_dim = None
    # waste the first batch to dynamically get the input dimension
    # for x, y in traindata:
    #     input_dim = x.shape[1]
    #     break

    # net = MLP(input_dim=564,
    net = MLP(input_dim=569,
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


def get_label_counts(input_dir):
    files = glob.glob(input_dir + "/**/*.csv.bz2", recursive=True)
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    n_files = len(files)
    for i, file in enumerate(files):
        print(f'Loading file {i}/{n_files} once to compute total number of labels for weights...')
        tmp = pd.read_csv(file,
                          sep=',',
                          dtype='float32',
                          encoding='cp1252', compression='bz2')
        tmp = tmp.apply(pd.to_numeric, downcast='integer', errors='coerce').dropna()
        for label, count in tmp['label'].value_counts().to_dict().items():
            label_counts[label] += count
    print(f'Starting training with dataset label quantities: {label_counts}')
    return list(label_counts.values())


def get_datasets(input_dir, seed=1):
    # dataset = OutOfMemoryDatasetV2(input_dir, chunk_size=1)
    dataset = InMemoryDataset(input_dir, merge_labels_567=True)
    total_len = len(dataset)
    train_len = math.ceil(len(dataset) * 0.8)
    test_len = total_len - train_len
    # val_len = int(total_len * 0.01)
    # add residuals to val_len to add up to total_len
    # val_len += total_len - (int(train_len) + int(test_len) + int(val_len))
    # set seed
    gen = torch.Generator().manual_seed(seed)
    train, test = random_split(dataset, [train_len, test_len], generator=gen)

    return train, test, dataset.label_counts  # get_label_counts(input_dir)  # dataset.label_counts  #


if __name__ == "__main__":
    """
    import torch
    import numpy as np
    freqs = [1099793,423499,202316,67931,50268,35760,14835,7885]
    
    weights = np.array(freqs) / sum(freqs)
    weights = 1 / weights
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / max(weights)
    print(weights)
    """
    data_dir = "/home/sascha/Documents/github.com/prl_baselines/data/dataset"
    train, test = get_in_mem_datasets(data_dir)
