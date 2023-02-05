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

        for file in self._csv_files:
            df = pd.read_csv(file, sep=',',
                             dtype='float32',
                             encoding='cp1252')
            fn_to_numeric = partial(pd.to_numeric, errors="coerce")
            df = df.apply(fn_to_numeric).dropna()

            df_fold = df[df['label'] == ActionSpace.FOLD]
            df_checkcall = df[df['label'] == ActionSpace.CHECK_CALL]
            df_raise_min = df[df['label'] == ActionSpace.RAISE_MIN_OR_3BB]
            df_raise_6bb = df[df['label'] == ActionSpace.RAISE_6_BB]
            df_raise_10bb = df[df['label'] == ActionSpace.RAISE_10_BB]
            df_raise_20bb = df[df['label'] == ActionSpace.RAISE_20_BB]
            df_raise_50bb = df[df['label'] == ActionSpace.RAISE_50_BB]
            df_allin = df[df['label'] == ActionSpace.RAISE_ALL_IN]

            df = pd.concat([df_fold,
                            df_checkcall,
                            df_raise_min,
                            df_raise_6bb,
                            df_raise_10bb,
                            df_raise_20bb,
                            df_raise_50bb,
                            df_allin])
            label_freqs = df['label'].value_counts()
            n_samples = sum([
                label_freqs[ActionSpace.RAISE_MIN_OR_3BB],
                label_freqs[ActionSpace.RAISE_6_BB],
                label_freqs[ActionSpace.RAISE_10_BB],
                label_freqs[ActionSpace.RAISE_20_BB],
                label_freqs[ActionSpace.RAISE_50_BB],
                label_freqs[ActionSpace.RAISE_ALL_IN]]
            )
            if use_downsampling:
                df = self.downsample(df, n_samples)
            print(f'Label_frequencies for player {Path(file).parent.name}: {label_freqs}')
            # shuffle
            df = df.sample(frac=1)
            [c(df, file) for c in cbs]

    def extract_subset(self, df: pd.DataFrame,
                       label: ActionSpace,
                       n_samples: int,
                       n_available: int) -> pd.DataFrame:
        return resample(df[df['label'] == label],
                        replace=True,
                        n_samples=min(n_available, n_samples),
                        random_state=1)
    def upsample(self, df: pd.DataFrame,
                       label: ActionSpace,
                       n_samples: int) -> pd.DataFrame:
        return resample(df[df['label'] == label],
                        replace=True,
                        n_samples=n_samples,
                        random_state=1)
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
        downsample_fn = partial(self.extract_subset, n_samples=n_samples)
        n_raises = n_samples
        n_upsample = round(n_raises / 6)  # so we have balanced FOLD, CHECK, RAISE where raises are 1/6 each
        df_fold_downsampled = downsample_fn(df, label=ActionSpace.FOLD, n_available=n_fold)
        df_checkcall_downsampled = downsample_fn(df, label=ActionSpace.CHECK_CALL, n_available=n_check_call)
        df_raise_min_downsampled = self.upsample(df, label=ActionSpace.RAISE_MIN_OR_3BB, n_samples=n_upsample)
        df_raise_6bb_downsampled = self.upsample(df, label=ActionSpace.RAISE_6_BB, n_samples=n_upsample)
        df_raise_10bb_downsampled = self.upsample(df, label=ActionSpace.RAISE_10_BB, n_samples=n_upsample)
        df_raise_20bb_downsampled = self.upsample(df, label=ActionSpace.RAISE_20_BB, n_samples=n_upsample)
        df_raise_50bb_downsampled = self.upsample(df, label=ActionSpace.RAISE_50_BB, n_samples=n_upsample)
        df_allin_downsampled = self.upsample(df, label=ActionSpace.RAISE_ALL_IN, n_samples=n_upsample)

        return pd.concat([df_fold_downsampled,
                          df_checkcall_downsampled,
                          df_raise_min_downsampled,
                          df_raise_6bb_downsampled,
                          df_raise_10bb_downsampled,
                          df_raise_20bb_downsampled,
                          df_raise_50bb_downsampled,
                          df_allin_downsampled]).sample(frac=1)
