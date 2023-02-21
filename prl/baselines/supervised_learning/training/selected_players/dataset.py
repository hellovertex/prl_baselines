from __future__ import annotations

import glob
import math
from functools import partial
from typing import Union

import pandas as pd
import torch
from prl.environment.Wrappers.base import ActionSpace
from sklearn.utils import resample
from torch.utils.data import Dataset
from torch.utils.data import random_split

from prl.baselines.supervised_learning.config import DATA_DIR

ROUNDS = ['round_preflop', 'round_flop', 'round_turn', 'round_river']


class InMemoryDataset(Dataset):
    def __init__(self,
                 path_to_csv_files=None,
                 rounds=None,
                 dichotomize=None,
                 blind_sizes="0.25-0.50",
                 merge_labels_567=True):
        files = glob.glob(path_to_csv_files + "/**/*.csv.bz2", recursive=True)
        assert len(files) == 1
        df = pd.read_csv(files[0],
                         # df = pd.read_csv(path_to_csv_files,
                         sep=',',
                         dtype='float32',
                         # dtype='float16',
                         encoding='cp1252',
                         compression='bz2')
        df = df.apply(pd.to_numeric, downcast='integer', errors='coerce').dropna()
        df = df.sample(frac=1)
        # todo: if rounds != 'all', sample for round only
        if rounds != 'all':
            if rounds == 'preflop':
                df = df[df['round_preflop'] == 1]
            elif rounds == 'flop':
                df = df[df['round_flop'] == 1]
            elif rounds == 'turn':
                df = df[df['round_turn'] == 1]
            elif rounds == 'river':
                df = df[df['round_river'] == 1]
            else:
                raise ValueError(f'Round was {rounds}, '
                                 f'but only valid values are "preflop, flop, turn, river, all"')
        label_key = 'label' if 'label' in df.columns else 'label.1'
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)

        if dichotomize is not None:
            # replace non-target labels with 0 and target label with 1
            assert dichotomize in list(range(len(ActionSpace)))
            # replace all negatives with 99
            for i in range(len(ActionSpace)):
                if i != dichotomize:
                    df[label_key].replace(i, 99, inplace=True)
            # replace all positives with 1
            df[label_key].replace(int(dichotomize), 1, inplace=True)
            # replace all negatives with 0
            df[label_key].replace(99, 0, inplace=True)
            label_counts = df[label_key].value_counts().to_dict()
            if label_counts[0] > label_counts[1]:
                df = self.upsample(df, label=1, n_samples=label_counts[0])
            else:
                df = self.upsample(df, label=0, n_samples=label_counts[1])
            self.label_counts = df[label_key].value_counts().to_list()
            self.y = torch.tensor(df[label_key].values, dtype=torch.int64)
        else:
            if merge_labels_567:
                df[label_key].replace(6, 5, inplace=True)
                df[label_key].replace(7, 5, inplace=True)

            label_dict = df[label_key].value_counts().to_dict()
            label_counts = []
            for i in [0, 1, 2, 3, 4, 5, 6, 7]:
                if i in label_dict:
                    label_counts.append(label_dict[i])
                else:
                    label_counts.append(0)
            # self.label_counts = df['label'].value_counts().to_list()
            self.label_counts = label_counts
            self.y = torch.tensor(df[label_key].values, dtype=torch.int64)
        # x = df.drop(['label'], axis=1)
        df.drop([label_key], axis=1, inplace=True)

        print(f'Dataframe size: {df.memory_usage(index=True, deep=True).sum()} bytes.')
        print(f'Starting training with dataset label quantities: {self.label_counts}')
        self.x = torch.tensor(df.values, dtype=torch.float32)
        df = None
        a = 1

    def extract_subset(self, df: pd.DataFrame,
                       label: Union[ActionSpace, int],
                       n_samples: int,
                       n_available: int) -> pd.DataFrame:
        return resample(df[df['label'] == label],
                        replace=False,
                        n_samples=min(n_available, n_samples),
                        random_state=1)

    def upsample(self, df: pd.DataFrame,
                 label: Union[ActionSpace, int],
                 n_samples: int) -> pd.DataFrame:
        df_base = df[df['label'] == label]
        samples = resample(df[df['label'] == label],
                           replace=True,
                           n_samples=max(n_samples - len(df_base), 0),
                           random_state=1)

        return pd.concat([df, samples])

    def downsample_fold_and_upsample_raises(self, df):
        n_fold = len(df[df['label'] == ActionSpace.FOLD])
        n_check_call = len(df[df['label'] == ActionSpace.CHECK_CALL])
        n_min_raise = len(df[df['label'] == ActionSpace.RAISE_MIN_OR_THIRD_OF_POT])
        n_raise_6bb = len(df[df['label'] == ActionSpace.RAISE_TWO_THIRDS_OF_POT])
        n_raise_10bb = len(df[df['label'] == ActionSpace.RAISE_POT])
        n_raise_20bb = len(df[df['label'] == ActionSpace.RAISE_2x_POT])
        n_raise_50bb = len(df[df['label'] == ActionSpace.RAISE_3x_POT])
        n_allin = len(df[df['label'] == ActionSpace.RAISE_ALL_IN])
        n_upsamples = max([n_min_raise,
                           n_raise_6bb,
                           n_raise_10bb,
                           n_raise_20bb,
                           n_raise_50bb,
                           n_allin])
        n_downsamples = n_check_call
        downsample_fn = partial(self.extract_subset, n_samples=n_downsamples)
        # downsample_fn = partial(self.extract_subset, n_samples=n_samples)
        # n_upsample = round(n_raises / 6)  # so we have balanced FOLD, CHECK, RAISE where raises are 1/6 each
        df_fold_downsampled = downsample_fn(df, label=ActionSpace.FOLD, n_available=n_fold)
        df_checkcall_downsampled = downsample_fn(df, label=ActionSpace.CHECK_CALL, n_available=n_check_call)
        df_raise_min_downsampled = self.upsample(df, label=ActionSpace.RAISE_MIN_OR_THIRD_OF_POT,
                                                 n_samples=n_upsamples)
        df_raise_6bb_downsampled = self.upsample(df, label=ActionSpace.RAISE_TWO_THIRDS_OF_POT,
                                                 n_samples=n_upsamples)
        df_raise_10bb_downsampled = self.upsample(df, label=ActionSpace.RAISE_POT,
                                                  n_samples=n_upsamples)
        df_raise_20bb_downsampled = self.upsample(df, label=ActionSpace.RAISE_2x_POT,
                                                  n_samples=n_upsamples)
        df_raise_50bb_downsampled = self.upsample(df, label=ActionSpace.RAISE_3x_POT,
                                                  n_samples=n_upsamples)
        df_allin_downsampled = self.upsample(df, label=ActionSpace.RAISE_ALL_IN, n_samples=n_upsamples)

        return pd.concat([df_fold_downsampled,
                          df_checkcall_downsampled,
                          df_raise_min_downsampled,
                          df_raise_6bb_downsampled,
                          df_raise_10bb_downsampled,
                          df_raise_20bb_downsampled,
                          df_raise_50bb_downsampled,
                          df_allin_downsampled]).sample(frac=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_datasets(input_dir, rounds, seed=1, dichotomize=None, merge_labels_567=False):
    # dataset = OutOfMemoryDatasetV2(input_dir, chunk_size=1)
    dataset = InMemoryDataset(input_dir,
                              rounds,
                              dichotomize=dichotomize,
                              merge_labels_567=merge_labels_567)
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
