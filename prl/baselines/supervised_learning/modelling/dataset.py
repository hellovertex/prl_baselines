from functools import partial
from os import listdir
from os.path import isfile, join, abspath

import pandas as pd
import psutil
import torch
from torch.utils.data import ConcatDataset, SubsetRandomSampler, BatchSampler, RandomSampler
from sklearn.utils import resample
import numpy as np

BATCH_SIZE = 512


class SingleTxtFileDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str):
        print(f"Initializing file {file_path}")
        self.file_path = file_path
        self._data: torch.Tensor | None = None
        self._labels: torch.Tensor | None = None
        self._len = self._get_len() - 1  # subtract one for column names

    def _get_len(self):
        """Get line count of large files cheaply"""
        with open(self.file_path, 'rb') as f:
            lines = 0
            buf_size = 1024 * 1024
            read_f = f.raw.read if hasattr(f, 'raw') and hasattr(f.raw, 'read') else f.read

            buf = read_f(buf_size)
            while buf:
                lines += buf.count(b'\n')
                buf = read_f(buf_size)

        return lines

    def load_file(self):
        # loading
        df = pd.read_csv(self.file_path, sep=",")

        # preprocessing
        fn_to_numeric = partial(pd.to_numeric, errors="coerce")
        df = df.apply(fn_to_numeric).dropna()
        labels = None
        try:
            # todo remove this when we do not have
            # todo two label columns by accident anymore
            labels = df.pop('label.1')
        except KeyError:
            labels = df.pop('label')
        assert len(df.index) > 0
        self._data = torch.tensor(df.values, dtype=torch.float32)
        self._labels = torch.tensor(labels.values, dtype=torch.long)

    def __getitem__(self, idx):
        if self._data is None:
            self.load_file()

        return self._data[idx], self._labels[idx]

    def __len__(self):
        return self._len


class HDF5Dataset(torch.utils.data.Dataset):
    """Efficiently load training data. Pytorchs Default Dataloaders and Samplers are #$)%#."""


class MultipleTxtFilesDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths: list):
        self._data = []
        self._labels = []
        # loads everything into memory, which we can do only because our
        # azure compute has 64GB ram
        self._init_load_data(file_paths)
        self._len = len(self._data)

    def _init_load_data(self, file_paths):
        for i, file_path in enumerate(file_paths):
            print(f'Loading File {i}/{len(file_paths)} into memory...')
            df = pd.read_csv(file_path, sep=",")
            # preprocessing
            fn_to_numeric = partial(pd.to_numeric, errors="coerce")
            df = df.apply(fn_to_numeric).dropna()
            labels = None
            try:
                # todo remove this when we do not have
                # todo two label columns by accident anymore
                labels = df.pop('label.1')
            except KeyError:
                labels = df.pop('label')
            assert len(df.index) > 0
            data = torch.tensor(df.values, dtype=torch.float32)
            labels = torch.tensor(labels.values, dtype=torch.long)
            for i, _ in enumerate(data):
                self._data.append(data[i])
                self._labels.append(labels[i])

    def __getitem__(self, idx):
        return self._data[idx], self._labels[idx]

    def __len__(self):
        return self._len


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
    test_dataset = MultipleParquetFilesDatasetDownsampled(train_dir_files)    # to generate three sets from one file
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
