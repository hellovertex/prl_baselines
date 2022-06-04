from functools import partial

import pandas as pd
from sklearn.utils import resample

FOLD = 0
CHECK_CALL = 1
RAISE_MIN_OR_3BB = 3
RAISE_HALF_POT = 4
RAISE_POT = 5
ALL_IN = 6


def downsample(df):
    print(f'value_counts = {df["label"].value_counts()}')
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


def to_parquet(path_to_csv_file,
               use_downsampling=True,
               output_dir='../../../../data/03_preprocessed/'):
    # read csv, sample properly, and save to parquet file
    df = pd.read_csv(path_to_csv_file)
    fn_to_numeric = partial(pd.to_numeric, errors="coerce")
    df = df.apply(fn_to_numeric).dropna()
    if use_downsampling:
        df = downsample(df)
    df.to_parquet(path=output_dir)
    return df



# preprocess("C:\\Users\\hellovertex\\Documents\\github.com\\dev.azure.com\\prl\\prl_baselines\data\\02_vectorized\\0.25-0.50\\6MAX_0.25-0.50.txt_2")