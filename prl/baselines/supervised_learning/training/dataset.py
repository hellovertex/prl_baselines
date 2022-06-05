from __future__ import annotations

import glob

import pandas as pd

BATCH_SIZE = 512

# ------------ FRESH START -------------
from torch.utils.data import IterableDataset
from itertools import chain
from random import shuffle


class OutOfMemoryDataset(IterableDataset):
    def __init__(self, path_to_csv_files, batch_size):
        self.path_to_csv_files = path_to_csv_files
        self.batch_size = batch_size
        self.filenames = glob.glob(path_to_csv_files.__str__() + '/**/*.csv', recursive=True)

    def _iterators(self):
        # 1. pd.read_csv(..., iterator=True) returns an Iterator for single .csv file
        # 2. chain(*[Iterator]) chains multiple iterators to a single iterator
        # 3. when iterating, one can call next(...).sample(frac=1) for additional shuffling of each dataframe
        iterators = []
        for file in self.filenames:
            iterators.append(pd.read_csv(file, sep=',', iterator=True, chunksize=self.batch_size))
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


for i in OutOfMemoryDataset('', 1000):
    print(i)
    break
