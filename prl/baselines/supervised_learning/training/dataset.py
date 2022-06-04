from __future__ import annotations

import csv
import itertools
import linecache
import subprocess
from functools import partial
from os import listdir
from os.path import isfile, join, abspath

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.utils import resample
from torch.utils.data import ConcatDataset, SubsetRandomSampler, BatchSampler, RandomSampler

BATCH_SIZE = 512

# ------------ FRESH START -------------
from torch.utils.data import Dataset, IterableDataset, DataLoader
from itertools import chain
from random import shuffle


class OutOfMemoryDataset(IterableDataset):
    def __init__(self, path_to_csv_files):
        self._path_to_csv_files = path_to_csv_files
        self.filenames = ["C:\\Users\\hellovertex\\Documents\\github.com\\dev.azure.com\\prl\\prl_baselines\\data\\dummies\\dummy1.csv",
                          "C:\\Users\\hellovertex\\Documents\\github.com\\dev.azure.com\\prl\\prl_baselines\\data\\dummies\\dummy2.csv",]

    def _iterators(self):
        # read_csv(..., iterator=True) returns an Iterator for single .csv file
        # chain(*[Iterator]) chains multiple iterators to a single iterator
        iterators = []
        for file in self.filenames:
            iterators.append(pd.read_csv(file, sep=',', iterator=True, chunksize=10000))
        return chain(*iterators)
        # when iterating, one can call .sample for additional shuffling of each file
    # def _shuffled_iter(self):
    #     while True:
    #         try:
    #             # .sample returns a sampled(shuffled) fraction of the dataframe
    #             # by setting frac=1 we basically shuffle the loaded chunk before returning it
    #             yield next(self._iterators).sample(frac=1)
    #         except StopIteration:
    #             # before raising StopIteration, we shuffle the list of .csv filenames
    #             # so that the next time the __iter__ is called,
    #             # the files will be loaded in a different order
    #             shuffle(self.filenames)
    #             raise StopIteration

    def __iter__(self):
        try:
            res = next(self._iterators())
            # next(self._iterators()).sample returns a sampled(shuffled) fraction of the dataframe
            # by setting frac=1 we basically shuffle the loaded chunk before returning it
            return chain([res], self._iterators())
        except StopIteration:
            # before raising StopIteration, we shuffle the list of .csv filenames
            # so that the next time the __iter__ is called,
            # the files will be loaded in a different order
            shuffle(self.filenames)
            raise StopIteration

it = OutOfMemoryDataset('')
print(type(it))
for i in it:
    print(i)
    break
print(type(iter(it)))

class LazyTextDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self._filename = filename
        self._total_data = int(subprocess.check_output("wc -l " + filename, shell=True).split()[0]) - 1

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        csv_line = csv.reader([line])
        # todo convert to binary e.g. numpy
        # delete df
        return next(csv_line)

    def __len__(self):
        return self._total_data


# path = "/ where_csv_files_are_dumped /"
# files = list(map(lambda x: path + x, (filter(lambda x: x.endswith("csv"), os.listdir(path)))))
# datasets = list(map(lambda x: LazyTextDataset(x), files))
# dataset = ConcatDataset(datasets)


class MultipleParquetFilesDatasetDownsampled(torch.utils.data.Dataset):
    def __init__(self, file_paths: list):
        self._data = None
        self._labels = None
        # loads everything into memory, which we can do only because our
        # azure compute has 64GB ram
        self._len = None
        self._init_load_data(file_paths)

    def _downsample(self):
        # resample to remove data imbalance
        # caution: happens in memory

        FOLD = 0
        CHECK_CALL = 1
        RAISE_MIN_OR_3BB = 3
        RAISE_HALF_POT = 4
        RAISE_POT = 5
        ALL_IN = 6
        print(f'value_counts = {self._data["label"].value_counts()}')
        n_samples = self._data['label'].value_counts()[ALL_IN]  # ALL_IN is rarest class
        df_fold = self._data[self._data['label'] == FOLD]
        df_checkcall = self._data[self._data['label'] == CHECK_CALL]
        df_raise_min = self._data[self._data['label'] == RAISE_MIN_OR_3BB]
        df_raise_half = self._data[self._data['label'] == RAISE_HALF_POT]
        df_raise_pot = self._data[self._data['label'] == RAISE_POT]
        df_allin = self._data[self._data['label'] == ALL_IN]

        df_fold_downsampled = resample(df_fold, replace=True, n_samples=n_samples, random_state=1)
        df_checkcall_downsampled = resample(df_checkcall, replace=True, n_samples=n_samples, random_state=1)
        df_raise_min_downsampled = resample(df_raise_min, replace=True, n_samples=n_samples, random_state=1)
        df_raise_half_downsampled = resample(df_raise_half, replace=True, n_samples=n_samples, random_state=1)
        df_raise_pot_downsampled = resample(df_raise_pot, replace=True, n_samples=n_samples, random_state=1)

        return pd.concat([df_fold_downsampled,
                          df_checkcall_downsampled,
                          df_raise_min_downsampled,
                          df_raise_half_downsampled,
                          df_raise_pot_downsampled,
                          df_allin])

    def _init_load_data(self, file_paths):
        """ Read from 1 up to n .parquet files into pandas
        dataframe (possibly concatenated) and downsample each class until all classes
        have equally many datapoint examples."""
        for i, file_path in enumerate(file_paths):
            print(f'Loading File {i}/{len(file_paths)} into memory...')
            df = pd.read_parquet(file_path).astype(np.float32)
            print(f'loaded data = {df}')
            # print(f'1 = df.head()')
            # preprocessing
            fn_to_numeric = partial(pd.to_numeric, errors="coerce")
            df = df.apply(fn_to_numeric).dropna()
            # print(f'2 = df.head()')
            if self._data is None:
                self._data = df
            else:
                self._data = pd.concat([self._data, df])
        print(f'self._data before downsampling {self._data.head()}')
        self._data = self._downsample()
        print(f'self._data after downsampling {self._data.head()}')
        self._len = len(self._data)
        print(f'self_len = {self._len}')
        self._labels = self._data.pop('label')
        print(f'self._data after popping label {self._data.head()}')
        self._data = torch.tensor(self._data.values, dtype=torch.float32)
        self._labels = torch.tensor(self._labels.values, dtype=torch.long)
        print(f'self._data after torch.tensor() = {self._data}')

    def __getitem__(self, idx):
        return self._data[idx], self._labels[idx]

    def __len__(self):
        return self._len


def get_dataloaders(train_dir):
    """Makes torch dataloaders by reading training directory files.
    1: Load training data files
    2: Create train, val, test splits
    3: Make datasets for each split
    4: Return dataloaders for each dataset
    """
    # get list of .txt-files inside train_dir
    train_dir_files = [join(train_dir, f) for f in listdir(train_dir) if isfile(join(train_dir, f))]

    # by convention, any .txt files inside this folder
    # that do not have .meta in their name, contain training data
    train_dir_files = [abspath(f) for f in train_dir_files if ".parquet" in f and ".aml" not in f]

    print(train_dir_files)
    print(f'{len(train_dir_files)} train files loaded')

    # splits
    # total_count = len(train_dir_files)
    # train_count = int(0.7 * total_count)
    # valid_count = int(0.2 * total_count)
    # test_count = total_count - train_count - valid_count

    # splits filepaths
    # train_files = train_dir_files[:train_count]
    # valid_files = train_dir_files[train_count:train_count + valid_count]
    # test_files = train_dir_files[-test_count:]

    # splits datasets
    train_dataset = MultipleParquetFilesDatasetDownsampled(train_dir_files)
    valid_dataset = MultipleParquetFilesDatasetDownsampled(train_dir_files)  # downsampling induces enough randomness
    test_dataset = MultipleParquetFilesDatasetDownsampled(train_dir_files)  # to generate three sets from one file
    # train_dataset = ConcatDataset(
    #     [SingleTxtFileDataset(train_file) for train_file in train_files])
    # valid_dataset = ConcatDataset(
    #     [SingleTxtFileDataset(train_file) for train_file in valid_files])
    # test_dataset = ConcatDataset(
    #     [SingleTxtFileDataset(train_file) for train_file in test_files])

    # print(f'Number of files loaded = {len(train_files) + len(test_files) + len(valid_files)}')
    print(f'For a total number of {len(test_dataset) + len(train_dataset) + len(valid_dataset)} examples')

    train_dataset_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=psutil.cpu_count(logical=False)
    )
    # train_dataset_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     # batch_size=BATCH_SIZE,
    #     sampler=BatchSampler(RandomSampler(train_dataset, True), BATCH_SIZE, False),
    #     num_workers=psutil.cpu_count(logical=False)
    # )
    valid_dataset_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=psutil.cpu_count(logical=False)
    )
    test_dataset_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=psutil.cpu_count(logical=False)
    )
    dataloaders = {
        "train": train_dataset_loader,
        "val": valid_dataset_loader,
        "test": test_dataset_loader,
    }

    return dataloaders
