from __future__ import annotations

import glob
import math
from functools import partial

import pandas as pd
import torch
from prl.environment.Wrappers.base import ActionSpace
from sklearn.utils import resample
from torch.utils.data import Dataset
from torch.utils.data import random_split

from prl.baselines.supervised_learning.config import DATA_DIR


class InMemoryDataset(Dataset):
    def __init__(self, path_to_csv_files=None, blind_sizes="0.25-0.50", merge_labels_567=True):
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
        if merge_labels_567:
            df['label.1'].replace(6, 5, inplace=True)
            df['label.1'].replace(7, 5, inplace=True)
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)
        try:
            self.label_counts = df['label'].value_counts().to_list()
            self.y = torch.tensor(df['label'].values, dtype=torch.int64)
            # x = df.drop(['label'], axis=1)
            df.drop(['label'], axis=1, inplace=True)
        except KeyError as e:
            print(e)
            self.label_counts = df['label.1'].value_counts().to_list()
            self.y = torch.tensor(df['label.1'].values, dtype=torch.int64)
            # x = df.drop(['label'], axis=1)
            df.drop(['label.1'], axis=1, inplace=True)

        print(f'Dataframe size: {df.memory_usage(index=True, deep=True).sum()} bytes.')
        print(f'Starting training with dataset label quantities: {self.label_counts}')
        self.x = torch.tensor(df.values, dtype=torch.float32)
        df = None
        a = 1

    def extract_subset(self, df: pd.DataFrame,
                       label: ActionSpace,
                       n_samples: int,
                       n_available: int) -> pd.DataFrame:
        return resample(df[df['label'] == label],
                        replace=False,
                        n_samples=min(n_available, n_samples),
                        random_state=1)

    def upsample(self, df: pd.DataFrame,
                 label: ActionSpace,
                 n_samples: int) -> pd.DataFrame:
        df_base = df[df['label'] == label]
        samples = resample(df[df['label'] == label],
                           replace=True,
                           n_samples=max(n_samples - len(df_base), 0),
                           random_state=1)

        return pd.concat([df_base, samples])

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


def get_datasets(input_dir, seed=1, merge_labels_567=False):
    # dataset = OutOfMemoryDatasetV2(input_dir, chunk_size=1)
    dataset = InMemoryDataset(input_dir, merge_labels_567=merge_labels_567)
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
