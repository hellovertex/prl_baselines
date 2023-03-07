from dataclasses import dataclass
from typing import List, Tuple

import torch
from tianshou.utils.net.common import MLP
from torch import nn

from prl.baselines.supervised_learning.v2.datasets.dataset_config import DatasetConfig


@dataclass
class TrainingParams:
    # early stopping - whatever is triggered first
    max_epochs: int = 100_000_000
    max_env_steps: int = 1_000_000
    # model params
    device: str = 'cpu'  # 'cuda' or 'cpu'
    input_dim: int = 569
    # FOLD, CHECK_CALL, RAISE
    output_dim: int = 3  # 1<= output_dim <= len(ActionSpace)
    # nn params
    lrs: Tuple[float] = (1e-6,)
    hdims: Tuple[int] = (512,)
    batch_size: int = 512
    # progress
    log_interval: int = 5
    eval_interval: int = 5
    # misc
    debug: bool = False


def get_model(params: TrainingParams):
    # todo check if model is dichotomizer, that for e.g.
    #  label ActionSpaceMinimal.Raise, the loss is computed correctly
    #  because there will only be one output neuron and the label is 2
    net = MLP(input_dim=params.input_dim,
              output_dim=params.output_dim,
              hidden_sizes=params.hdims,
              norm_layer=None,
              activation=nn.ReLU,
              device=params.device,
              linear_layer=nn.Linear,
              flatten_input=False)
    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
    return net
