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
            df_raise_half = df[df['label'] == ActionSpace.RAISE_HALF_POT]
            df_raise_pot = df[df['label'] == ActionSpace.RAISE_POT]
            df_allin = df[df['label'] == ActionSpace.ALL_IN]

            df = pd.concat([df_fold, df_checkcall, df_raise_min, df_raise_half, df_raise_pot, df_allin])
            label_freqs = df['label'].value_counts()
            n_samples = sum([
                label_freqs[ActionSpace.RAISE_MIN_OR_3BB],
                label_freqs[ActionSpace.RAISE_HALF_POT],
                label_freqs[ActionSpace.RAISE_POT],
                label_freqs[ActionSpace.ALL_IN]]
            )
            if use_downsampling:
                df = self.downsample(df, n_samples)
            print(f'Label_frequencies for player {Path(file).parent.name}: {label_freqs}')
            # shuffle
            df = df.sample(frac=1)
            [c(df, file) for c in cbs]

    def downsample(self, df, n_samples):
        """ Each label in df will be downsampled to the number of all ins, "
            so that training data is equally distributed. """
        n_fold = len(df[df['label'] == ActionSpace.FOLD])
        n_check_call = len(df[df['label'] == ActionSpace.CHECK_CALL])
        n_min_raise = len(df[df['label'] == ActionSpace.RAISE_MIN_OR_3BB])
        n_raise_half_pot = len(df[df['label'] == ActionSpace.RAISE_HALF_POT])
        n_raise_pot = len(df[df['label'] == ActionSpace.RAISE_POT])
        n_allin = len(df[df['label'] == ActionSpace.ALL_IN])
        df_fold_downsampled = resample(df[df['label'] == ActionSpace.FOLD],
                                       replace=True,
                                       n_samples=min(n_fold, n_samples),
                                       random_state=1)
        df_checkcall_downsampled = resample(df[df['label'] == ActionSpace.CHECK_CALL],
                                            replace=True,
                                            n_samples=min(n_check_call, n_samples),
                                            random_state=1)
        df_raise_min_downsampled = resample(df[df['label'] == ActionSpace.RAISE_MIN_OR_3BB],
                                            replace=True,
                                            n_samples=min(n_min_raise, n_samples),
                                            random_state=1)
        df_raise_half_downsampled = resample(df[df['label'] == ActionSpace.RAISE_HALF_POT],
                                             replace=True,
                                             n_samples=min(n_raise_half_pot, n_samples),
                                             random_state=1)
        df_raise_pot_downsampled = resample(df[df['label'] == ActionSpace.RAISE_POT],
                                            replace=True,
                                            n_samples=min(n_raise_pot, n_samples),
                                            random_state=1)
        df_allin_downsampled = resample(df[df['label'] == ActionSpace.ALL_IN],
                                        replace=True,
                                        n_samples=min(n_allin, n_samples),
                                        random_state=1)

        return pd.concat([df_fold_downsampled,
                          df_checkcall_downsampled,
                          df_raise_min_downsampled,
                          df_raise_half_downsampled,
                          df_raise_pot_downsampled,
                          df_allin_downsampled]).sample(frac=1)
