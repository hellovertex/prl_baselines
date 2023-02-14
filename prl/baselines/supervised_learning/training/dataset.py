from __future__ import annotations

import glob
from functools import partial

import torch
from itertools import chain
from random import shuffle

import pandas as pd
from prl.environment.Wrappers.base import ActionSpace
from sklearn.utils import resample
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

from prl.baselines.supervised_learning.config import DATA_DIR


def row_count(input):
    return sum(1 for line in open(input, encoding='cp1252'))


class InMemoryDataset(Dataset):
    def __init__(self, path_to_csv_files=None, blind_sizes="0.25-0.50"):
        if not path_to_csv_files:
            path_to_csv_files = str(DATA_DIR) + '/03_preprocessed' + f'/{blind_sizes}'

        files = glob.glob(path_to_csv_files + "/**/*.csv.bz2", recursive=True)
        df = pd.read_csv(files[0],
                         sep=',',
                         #dtype='float32',
                         dtype='float16',
                         encoding='cp1252', compression='bz2')
        df = df.apply(pd.to_numeric, downcast='integer', errors='coerce').dropna()
        n_files = len(files[1:])
        for i, file in enumerate(files[1:]):
            tmp = pd.read_csv(file,
                              sep=',',
                              dtype='float16',
                              encoding='cp1252', compression='bz2')
            tmp = tmp.apply(pd.to_numeric, downcast='integer', errors='coerce').dropna()
            df = pd.concat([df, tmp], ignore_index=True)
            print(f'Loaded file {i}/{n_files}...')

        self.label_counts = df['label'].value_counts().to_list()
        self.y = torch.tensor(df['label'].values, dtype=torch.int64)
        # x = df.drop(['label'], axis=1)
        df.drop(['label'], axis=1, inplace=True)

        print(f'Dataframe size: {df.memory_usage(index=True, deep=True).sum()} bytes.')
        print(f'Starting training with dataset label quantities: {self.label_counts}')
        self.x = torch.tensor(df.values, dtype=torch.float16)
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
        n_min_raise = len(df[df['label'] == ActionSpace.RAISE_MIN_OR_3BB])
        n_raise_6bb = len(df[df['label'] == ActionSpace.RAISE_6_BB])
        n_raise_10bb = len(df[df['label'] == ActionSpace.RAISE_10_BB])
        n_raise_20bb = len(df[df['label'] == ActionSpace.RAISE_20_BB])
        n_raise_50bb = len(df[df['label'] == ActionSpace.RAISE_50_BB])
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
        df_raise_min_downsampled = self.upsample(df, label=ActionSpace.RAISE_MIN_OR_3BB,
                                                 n_samples=n_upsamples)
        df_raise_6bb_downsampled = self.upsample(df, label=ActionSpace.RAISE_6_BB,
                                                 n_samples=n_upsamples)
        df_raise_10bb_downsampled = self.upsample(df, label=ActionSpace.RAISE_10_BB,
                                                  n_samples=n_upsamples)
        df_raise_20bb_downsampled = self.upsample(df, label=ActionSpace.RAISE_20_BB,
                                                  n_samples=n_upsamples)
        df_raise_50bb_downsampled = self.upsample(df, label=ActionSpace.RAISE_50_BB,
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


class OutOfMemoryDataset(IterableDataset):
    def __init__(self, path_to_csv_files, batch_size):
        self.path_to_csv_files = path_to_csv_files
        self.batch_size = batch_size
        # important: keep recursive=False parameter because path_to_csv_files includes folder test/
        self.filenames = glob.glob(path_to_csv_files.__str__() + '/*.csv', recursive=False)

        # Compute length of dataset by adding all rows across files
        # happens out of memory
        self._len = 0

        pbar = tqdm(enumerate(self.filenames), total=len(self.filenames))
        descr = f"Computing total length of dataset. This may take a while."
        pbar.set_description(descr)
        for i, f in pbar:
            nrows = row_count(f)
            pbar.set_description(f"Computing total length of dataset. This may take a while. "
                                 f"File {i + 1}/{len(self.filenames)} has {nrows} rows")
            # print(f'File {f} has {nrows} rows.')
            self._len += nrows
        self._len -= len(self.filenames)  # subtract headers

    def _iterators(self):
        # 1. pd.read_csv(..., iterator=True) returns an Iterator for single .csv file
        # 2. chain(*[Iterator]) chains multiple iterators to a single iterator
        # 3. when iterating, one can call next(...).sample(frac=1) for additional shuffling of each dataframe
        iterators = []
        for i, f in enumerate(self.filenames):
            iterators.append(pd.read_csv(f, sep=',', dtype='float32',
                                         iterator=True,
                                         chunksize=self.batch_size,
                                         encoding='cp1252'))
        return chain(*iterators)

    def __iter__(self):
        try:
            res = next(self._iterators())
            # Tip: use next(...).sample(frac=1) for additional shuffling of each dataframe
            # return chain([res], self._iterators())
            return self._iterators()
        except StopIteration:
            # before raising StopIteration, we shuffle the list of .csv filenames
            # so that the next time the __iter__ is called,
            # the files will be loaded in a different order
            shuffle(self.filenames)
            return self._iterators()  # raise StopIteration

    def __len__(self):
        return self._len

import bz2
import csv
import random


class OutOfMemoryDatasetV2(Dataset):
    def __init__(self, path_to_csv_files, chunk_size=0.25):

        self.file_paths = glob.glob(path_to_csv_files + "/**/*.csv.bz2", recursive=True)
        self.chunk_size = chunk_size
        self.data = []

    def __len__(self):
        return int(len(self.file_paths) * self.chunk_size)

    def __getitem__(self, index):
        if self.data:
            return self.data[index]

        # Read a specified number of files (chunk_size) into memory
        file_indices = range(len(self.file_paths))
        chunk = random.sample(file_indices, int(len(file_indices) * self.chunk_size))
        self.data = []
        for file_index in chunk:
            file_path = self.file_paths[file_index]
            with bz2.open(file_path, 'rt') as f:
                reader = csv.reader(f)
                self.data.extend(list(reader))

        return self.data[index]