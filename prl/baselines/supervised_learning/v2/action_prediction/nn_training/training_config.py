from dataclasses import dataclass
from typing import List, Tuple

from prl.baselines.supervised_learning.v2.datasets.dataset_config import DatasetConfig


@dataclass
class TrainingParams:
    # early stopping - whatever is triggered first
    max_epochs: int = 100_000_000
    max_env_steps: int = 1_000_000
    # nn params
    lrs: Tuple[float] = (1e-6,)
    hdmis: Tuple[Tuple[int]] = ((512,),)
    batch_size: int = 512
    # progress
    log_interval: int = 5
    eval_interval: int = 5


@dataclass
class TrainingConfig:
    dataset_config: DatasetConfig
    training_params: TrainingParams
    debug: bool = False
    # todo: dispatch InMemDS creation and model creation according to
    #  `dataset_config` and `training_params`
