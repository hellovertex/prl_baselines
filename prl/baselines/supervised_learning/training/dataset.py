from __future__ import annotations

import glob
from itertools import chain
from random import shuffle

import pandas as pd
from torch.utils.data import IterableDataset
from tqdm import tqdm


def row_count(input):
    return sum(1 for line in open(input, encoding='cp1252'))


class OutOfMemoryDataset(IterableDataset):
    def __init__(self, path_to_csv_files, batch_size):
        self.path_to_csv_files = path_to_csv_files
        self.batch_size = batch_size
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
            return chain([res], self._iterators())
        except StopIteration:
            # before raising StopIteration, we shuffle the list of .csv filenames
            # so that the next time the __iter__ is called,
            # the files will be loaded in a different order
            shuffle(self.filenames)
            raise StopIteration

    def __len__(self):
        return self._len
