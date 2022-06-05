import glob
from functools import partial

import pandas as pd
from sklearn.utils import resample

FOLD = 0
CHECK_CALL = 1
RAISE_MIN_OR_3BB = 3
RAISE_HALF_POT = 4
RAISE_POT = 5
ALL_IN = 6


def to_csv(df: pd.DataFrame,
           filename,
           output_dir='../../../../data/03_preprocessed/'):
    df.to_csv(path=output_dir + filename)


def to_parquet(df: pd.DataFrame,
               filename,
               output_dir='../../../../data/03_preprocessed/'):
    df.to_parquet(path=output_dir + filename)


class Preprocessor:
    """Out of memory preprocessor. Means the
    - csv files get loaded one by one
    - preprocessing is applied
    - csv file is written back to `output_dir`, which is data/03_preprocessed by default
    - additional callbacks can be provided, e.g. to write the file back as .parquet file """

    def __init__(self, path_to_csv_files, callbacks=[to_csv]):
        self._path_to_csv_files = path_to_csv_files
        self._csv_files = glob.glob(path_to_csv_files.__str__() + '/**/*.csv', recursive=True)
        self._callbacks = callbacks

    def run(self):
        for file in self._csv_files:
            df = pd.read_csv(file)
            fn_to_numeric = partial(pd.to_numeric, errors="coerce")
            df = df.apply(fn_to_numeric).dropna()
            df = self.downsample(df)
            [c(df, file) for c in self._callbacks]

    def downsample(self, df):
        n_samples = df['label'].value_counts()[ALL_IN]  # ALL_IN is rarest class
        df_fold = df[df['label'] == FOLD]
        df_checkcall = df[df['label'] == CHECK_CALL]
        df_raise_min = df[df['label'] == RAISE_MIN_OR_3BB]
        df_raise_half = df[df['label'] == RAISE_HALF_POT]
        df_raise_pot = df[df['label'] == RAISE_POT]
        df_allin = df[df['label'] == ALL_IN]

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

# preprocess("C:\\Users\\hellovertex\\Documents\\github.com\\dev.azure.com\\prl\\prl_baselines\data\\02_vectorized\\0.25-0.50\\6MAX_0.25-0.50.txt_2")
