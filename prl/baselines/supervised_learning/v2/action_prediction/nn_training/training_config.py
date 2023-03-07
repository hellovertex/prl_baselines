import os
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from tianshou.utils.net.common import MLP
from torch import nn

from prl.baselines import DATA_DIR as DEFAULT_DATA_DIR
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
    lrs: Tuple[Tuple[float]] = (1e-6,)
    hdims: Tuple[Tuple[int]] = (512,)
    batch_size: int = 512
    # progress
    log_interval: int = 5
    eval_interval: int = 5
    # misc
    debug: bool = False
    DATA_DIR: Optional[str] = None

    def results_dir(self,
                    conf: DatasetConfig,
                    hdims: Tuple[int],
                    lr: float):
        # require dataset config here for consistency
        assert conf.make_dataset_for_each_individual is not None
        assert conf.action_generation_option is not None
        DATA_DIR = DEFAULT_DATA_DIR if not conf.DATA_DIR else conf.DATA_DIR
        train_result_dir = os.path.join(DATA_DIR, '05_train_results')
        subdir_01_player_or_pool = 'per_selected_player' if \
            conf.make_dataset_for_each_individual else 'player_pool'
        subdir_00_nl = conf.nl
        subdir_02_fold_or_no_fold = conf.action_generation_option.name.replace(
            'make_',
            '')
        # when `make_dataset_for_each_individual` is set, the individual folders
        # must be created during encoding, since we dont know the ranks a priori here
        subdir_03_top_n_players = f'Top{conf.num_top_players}Players_' \
                                  f'n_showdowns={conf.min_showdowns}' if not \
            conf.make_dataset_for_each_individual else ''
        subdir_04_rounds = conf.target_rounds_to_str()  # move target_rounds_to_str
        subdir_05_actions = conf.actions_to_str()  # move  actions_to_str
        return os.path.join(*[
            train_result_dir,
            subdir_00_nl,
            subdir_01_player_or_pool,
            subdir_02_fold_or_no_fold,
            subdir_03_top_n_players,
            subdir_04_rounds,
            subdir_05_actions,
            f'{hdims}_{lr}'
        ])


def get_model(input_dim, output_dim, hdims, device):
    # todo check if model is dichotomizer, that for e.g.
    #  label ActionSpaceMinimal.Raise, the loss is computed correctly
    #  because there will only be one output neuron and the label is 2
    if isinstance(hdims, int):
        hdims = [hdims]
    net = MLP(input_dim=input_dim,
              output_dim=output_dim,
              hidden_sizes=hdims,
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
