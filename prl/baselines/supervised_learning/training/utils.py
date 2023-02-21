import glob
import logging
import math
from typing import (
    Optional,
    Sequence,
    Type,
    Union,
)

import numpy as np
import pandas as pd
import torch
from prl.environment.Wrappers.base import ActionSpace
from tianshou.utils.net.common import MLP, ModuleType
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


def get_model_predict_fold_binary(traindata, hidden_dims, device, merge_labels567=False):
    # network

    class MDL(MLP):
        def __init__(self, input_dim: int,
                     output_dim: int = 0,
                     hidden_sizes: Sequence[int] = (),
                     norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
                     activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
                     device: Optional[Union[str, int, torch.device]] = None,
                     linear_layer: Type[nn.Linear] = nn.Linear,
                     flatten_input: bool = True, ):
            super().__init__(input_dim=input_dim,
                             output_dim=output_dim,
                             hidden_sizes=hidden_sizes,
                             norm_layer=norm_layer,
                             activation=activation,
                             device=device,
                             linear_layer=linear_layer,
                             flatten_input=flatten_input)
            self.sigmoid = nn.Sigmoid()

        def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
            if self.device is not None:
                obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
            if self.flatten_input:
                obs = obs.flatten(1)
            x = self.model(obs)
            return self.sigmoid(x)

    net = MDL(input_dim=569,
              output_dim=1,
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


def get_model(traindata, output_dim, hidden_dims, device, merge_labels567=False):
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
