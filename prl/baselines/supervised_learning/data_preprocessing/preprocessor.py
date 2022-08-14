import glob
import os
from functools import partial

import pandas as pd
from sklearn.utils import resample

from prl.baselines.supervised_learning.config import DATA_DIR

FOLD = 0
CHECK_CALL = 1
RAISE_MIN_OR_3BB = 3
RAISE_HALF_POT = 4
RAISE_POT = 5
ALL_IN = 6


class Preprocessor:
    """Out of memory preprocessor. Means the
    - csv files get loaded one by one
    - preprocessing is applied
    - csv file is written back to `output_dir`, which is data/03_preprocessed by default
    - additional callbacks can be provided, e.g. to write the file back as .parquet file """

    def __init__(self, path_to_csv_files, callbacks=None):
        self._path_to_csv_files = path_to_csv_files
        self._csv_files = glob.glob(path_to_csv_files.__str__() + '/*.csv', recursive=False)
        self._callbacks = [] if not callbacks else callbacks

    def run(self, use_downsampling=True):
        for file in self._csv_files:
            df = pd.read_csv(file, sep=';',
                             dtype='float32',
                             encoding='cp1252')
            fn_to_numeric = partial(pd.to_numeric, errors="coerce")
            df = df.apply(fn_to_numeric).dropna()

            df_fold = df[df['label'] == FOLD]
            df_checkcall = df[df['label'] == CHECK_CALL]
            df_raise_min = df[df['label'] == RAISE_MIN_OR_3BB]
            df_raise_half = df[df['label'] == RAISE_HALF_POT]
            df_raise_pot = df[df['label'] == RAISE_POT]
            df_allin = df[df['label'] == ALL_IN]

            # overwrite labels such that 0,1,3,4,5,6 become 0,1,2,3,4,5 because 2 is never used
            df_allin['label'] = 5
            df_raise_pot['label'] = 4
            df_raise_half['label'] = 3
            df_raise_min['label'] = 2

            df = pd.concat([df_fold, df_checkcall, df_raise_min, df_raise_half, df_raise_pot, df_allin])

            if use_downsampling:
                df = self.downsample(df)
            [c(df, file) for c in self._callbacks]

    def downsample(self, df):
        n_samples = df['label'].value_counts()[ALL_IN]  # ALL_IN is rarest class

        df_fold_downsampled = resample(df[df['label'] == FOLD], replace=True, n_samples=n_samples, random_state=1)
        df_checkcall_downsampled = resample(df[df['label'] == CHECK_CALL], replace=True, n_samples=n_samples,
                                            random_state=1)
        df_raise_min_downsampled = resample(df[df['label'] == RAISE_MIN_OR_3BB], replace=True, n_samples=n_samples,
                                            random_state=1)
        df_raise_half_downsampled = resample(df[df['label'] == RAISE_HALF_POT], replace=True, n_samples=n_samples,
                                             random_state=1)
        df_raise_pot_downsampled = resample(df[df['label'] == RAISE_POT], replace=True, n_samples=n_samples,
                                            random_state=1)

        return pd.concat([df_fold_downsampled,
                          df_checkcall_downsampled,
                          df_raise_min_downsampled,
                          df_raise_half_downsampled,
                          df_raise_pot_downsampled,
                          df[df['label'] == ALL_IN]]).sample(frac=1)
