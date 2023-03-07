from __future__ import annotations

import logging
import math

import click
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split

from prl.baselines.supervised_learning.v2.datasets.dataset_config import (
    ActionGenOption,
    Stage,
    DatasetConfig,
    DataImbalanceCorrectionTechnique,
    arg_num_top_players,
    arg_nl,
    arg_from_gdrive_id,
    arg_make_dataset_for_each_individual,
    arg_action_generation_option,
    arg_use_multiprocessing,
    arg_min_showdowns,
    arg_target_rounds,
    arg_action_space,
    arg_sub_sampling_technique,
    arg_seed_dataset
)
from prl.baselines.supervised_learning.v2.datasets.utils import \
    parse_cmd_action_to_action_cls


class InMemoryDataset(Dataset):
    def __init__(self,
                 dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        # todo: read csv files and

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_label_weights(dataset, dataset_config: DatasetConfig):
    weights = None
    use_weights = DataImbalanceCorrectionTechnique. \
        dont_resample_but_use_label_weights_in_optimizer
    if dataset_config.sub_sampling_technique == use_weights:
        # label weights to account for dataset imbalance
        weights = np.array(dataset.label_counts) / sum(dataset.label_counts)
        weights = 1 / weights
        weights[weights == np.inf] = 0
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / max(weights)
    return weights


def get_datasets(dataset_config: DatasetConfig):
    # dataset = OutOfMemoryDatasetV2(input_dir, chunk_size=1)
    dataset = InMemoryDataset(dataset_config)
    total_len = len(dataset)
    train_len = math.ceil(len(dataset) * 0.8)
    test_len = total_len - train_len
    # val_len = int(total_len * 0.01)
    # add residuals to val_len to add up to total_len
    # val_len += total_len - (int(train_len) + int(test_len) + int(val_len))
    # set seed
    gen = torch.Generator().manual_seed(dataset_config.seed)
    train, test = random_split(dataset, [train_len, test_len], generator=gen)
    label_weights = get_label_weights(dataset, dataset_config)
    return train, test, label_weights


@click.command()
@arg_num_top_players
@arg_nl
@arg_from_gdrive_id
@arg_make_dataset_for_each_individual
@arg_action_generation_option
@arg_use_multiprocessing
@arg_min_showdowns
@arg_target_rounds
@arg_action_space
@arg_sub_sampling_technique
@arg_seed_dataset
def main(num_top_players,
         nl,
         make_dataset_for_each_individual,
         action_generation_option,
         min_showdowns,
         target_rounds,
         action_space,
         sub_sampling_technique,
         seed):
    # Assumes raw_data.py has been ran to download and extract hand histories.
    opt = DatasetConfig(
        num_top_players=num_top_players,
        nl=nl,
        make_dataset_for_each_individual=make_dataset_for_each_individual,
        action_generation_option=ActionGenOption(action_generation_option),
        min_showdowns=min_showdowns,
        target_rounds=[Stage(x) for x in target_rounds],
        action_space=[parse_cmd_action_to_action_cls(action_space)],
        sub_sampling_technique=sub_sampling_technique,
        seed=seed
    )


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
