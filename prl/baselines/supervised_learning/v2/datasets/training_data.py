from __future__ import annotations

import glob
import logging
import math
from typing import List

import click
import numpy as np
import pandas as pd
import torch
from prl.environment.Wrappers.base import ActionSpace, ActionSpaceMinimal
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
from prl.baselines.supervised_learning.v2.datasets.preprocessed_data import \
    make_preprocessed_data_if_not_exists_already
from prl.baselines.supervised_learning.v2.datasets.utils import \
    parse_cmd_action_to_action_cls


class InMemoryDataset(Dataset):
    def __init__(self,
                 dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        csv_files = glob.glob(dataset_config.dir_preprocessed_data + '/**/*.csv.bz2',
                              recursive=True)
        df_total = pd.DataFrame()
        for file in csv_files:
            df = pd.read_csv(file,
                             sep=',',
                             dtype='float32',
                             # dtype='float16',
                             encoding='cp1252',
                             compression='bz2')
            df = df.apply(pd.to_numeric, downcast='integer', errors='coerce').dropna()
            df = df.sample(frac=1)
            df_total = pd.concat([df_total, df])
        self.label_counts = self.get_label_counts(df_total)
        self.y = torch.tensor(df_total['label'].values, dtype=torch.int64)
        df_total.drop(['label'], axis=1, inplace=True)
        self.x = torch.tensor(df_total.values, dtype=torch.float32)

        # preprocessor tests should have covered this, but just to be sure we check if
        # only requested actions are present in dataset
        self.assert_correct_labels()
        self.assert_correct_rounds(df_total)

    def assert_correct_labels(self):
        num_unique_labels = len(self.label_counts)
        act = self.dataset_config.action_space[0]
        if act is ActionSpace:
            assert num_unique_labels == len(ActionSpace)
        elif act is ActionSpaceMinimal:
            assert num_unique_labels == len(ActionSpace)
        elif isinstance(act, ActionSpaceMinimal):
            assert num_unique_labels == 1

    def assert_correct_rounds(self, df):
        for stage in self.dataset_config.target_rounds:
            name = 'round_' + stage.name.casefold()
            assert (df[name] == 1).any()
        for stage in list(Stage):
            if stage not in self.dataset_config.target_rounds:
                name = 'round_' + stage.name.casefold()
                assert not (df[name] == 1).any()

    @staticmethod
    def get_label_counts(df) -> List[int]:
        label_dict = df['label'].value_counts().to_dict()
        label_counts = []
        for i in [0, 1, 2, 3, 4, 5, 6, 7]:
            if i in label_dict:
                label_counts.append(label_dict[i])
            else:
                label_counts.append(0)
        # self.label_counts = df['label'].value_counts().to_list()
        return label_counts

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_label_weights(dataset: InMemoryDataset,
                      dataset_config: DatasetConfig):
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
    make_preprocessed_data_if_not_exists_already(dataset_config,
                                                 use_multiprocessing=True)
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
    dataset_config = DatasetConfig(
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
    train, test, label_weights = get_datasets(dataset_config)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
