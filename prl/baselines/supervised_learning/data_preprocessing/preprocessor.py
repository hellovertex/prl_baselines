import glob
import os
from functools import partial
from pathlib import Path

import pandas as pd
from prl.environment.Wrappers.augment import ActionSpace
from sklearn.utils import resample

from prl.baselines.supervised_learning.config import DATA_DIR


class Preprocessor:
    def __init__(self, path_to_csv_files, recursive=False):
        self._path_to_csv_files = path_to_csv_files
        self._csv_files = glob.glob(path_to_csv_files.__str__() + '**/*.csv', recursive=recursive)
        if not self._csv_files:
            self._csv_files = glob.glob(path_to_csv_files.__str__() + '/**/*.csv', recursive=recursive)

    def run(self, use_downsampling=True, callbacks=None):
        """
        For each csv file, create a numerical dataframe and remove erroneous lines
        Additional callbacks can be provided, e.g. to write the file back as .csv file """

        cbs = [] if not callbacks else callbacks
        df_total = pd.DataFrame()
        for file in self._csv_files:
            df = pd.read_csv(file, sep=',',
                             dtype='float32',
                             encoding='cp1252')
            fn_to_numeric = partial(pd.to_numeric, errors="coerce")
            df = df.apply(fn_to_numeric).dropna()
            df_total = pd.concat([df_total, df])

        print(df_total.head())
        # todo: do downsampling on total df not individual dfs
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

    def downsample(self, df, n_samples):
        """ Each label in df will be downsampled to the number of all ins, "
            so that training data is equally distributed. """
        n_fold = len(df[df['label'] == ActionSpace.FOLD])
        n_check_call = len(df[df['label'] == ActionSpace.CHECK_CALL])
        n_min_raise = len(df[df['label'] == ActionSpace.RAISE_MIN_OR_3BB])
        n_raise_6bb = len(df[df['label'] == ActionSpace.RAISE_6_BB])
        n_raise_10bb = len(df[df['label'] == ActionSpace.RAISE_10_BB])
        n_raise_20bb = len(df[df['label'] == ActionSpace.RAISE_20_BB])
        n_raise_50bb = len(df[df['label'] == ActionSpace.RAISE_50_BB])
        n_allin = len(df[df['label'] == ActionSpace.RAISE_ALL_IN])
        # n_upsamples = n_downsamples = max([n_min_raise,
        #                               n_raise_6bb,
        #                                    n_raise_10bb,
        #                                    n_raise_20bb,
        #                                    n_raise_50bb,
        #                                    n_allin])
        n_upsamples = 12000
        downsample_fn = partial(self.extract_subset, n_samples=90000)
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
