from __future__ import annotations

import csv
import glob

import enlighten as enlighten
import pandas as pd
from torch.utils.data import IterableDataset
from itertools import chain
from random import shuffle
import logging

BATCH_SIZE = 512


def row_count(input):
    file = csv.reader(input)
    return sum(1 for row in file)


class OutOfMemoryDataset(IterableDataset):
    def __init__(self, path_to_csv_files, batch_size):
        self.path_to_csv_files = path_to_csv_files
        self.batch_size = batch_size
        self.filenames = glob.glob(path_to_csv_files.__str__() + '/**/*.csv', recursive=True)

        # Setup progress bar
        self.manager = enlighten.get_manager()
        self.pbar_len = self.manager.counter(total=len(self.filenames), desc='Ticks', unit='ticks')
        self.pbar_iter = self.manager.counter(total=len(self.filenames), desc='Ticks', unit='ticks')

        # Compute length of dataset by adding all rows across files
        # happens out of memory
        self._len = 0
        for i, f in enumerate(self.filenames):
            # print("Estimating length of dataset", end='') if i == 0 else print('.', end='')
            logging.info(f"Estimating length of dataset: file {i}/{len(self.filenames)}")
            self.pbar_len.update()
            self._len += row_count(f)
        self._len -= len(self.filenames)  # subtract headers

    def _iterators(self):
        # 1. pd.read_csv(..., iterator=True) returns an Iterator for single .csv file
        # 2. chain(*[Iterator]) chains multiple iterators to a single iterator
        # 3. when iterating, one can call next(...).sample(frac=1) for additional shuffling of each dataframe
        iterators = []
        for i, f in enumerate(self.filenames):
            # print("Computing iterators of dataset", end='') if i == 0 else print('.', end='')
            logging.info(f"Computing iterators of dataset: file {i}/{len(self.filenames)}")
            iterators.append(pd.read_csv(f, sep=',',
                                         iterator=True,
                                         chunksize=self.batch_size,
                                         encoding='cp1252'))
            self.pbar_iter.update()
        return chain(*iterators)

    def __iter__(self):
        try:
            res = next(self._iterators())
            # Tip: use next(...).sample(frac=1) for additional shuffling of each dataframe
            return chain([res], self._iterators())
        except StopIteration:
            # before raising StopIteration, we shuffle the list of .csv filenames
            # so that the next time the __iter__ is called,
            # the files will be loaded in a different order
            shuffle(self.filenames)
            raise StopIteration

    def __len__(self):
        return self._len
