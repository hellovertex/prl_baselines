import os
from pathlib import Path

import pandas as pd
import glob
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as cols


def main():
    path_to_csv_files = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/v2/top_100_only_wins_no_folds"
    files = glob.glob(path_to_csv_files + "/**/*.csv.bz2", recursive=True)

    df = pd.read_csv(files[0],
                     # df = pd.read_csv(path_to_csv_files,
                     sep=',',
                     dtype='float32',
                     # dtype='float16',
                     encoding='cp1252',
                     compression='bz2')
    df = df.apply(pd.to_numeric, downcast='integer', errors='coerce').dropna()
    n_files = len(files[1:])
    for i, file in enumerate(files[1:]):
        tmp = pd.read_csv(file,
                          sep=',',
                          # dtype='float16',
                          dtype='float32',
                          encoding='cp1252', compression='bz2')
        tmp = tmp.apply(pd.to_numeric, downcast='integer', errors='coerce').dropna()
        df = pd.concat([df, tmp], ignore_index=True)
        print(f'Loaded file {i}/{n_files}...')
    for round in [cols.Round_preflop, cols.Round_flop, cols.Round_turn, cols.Round_river]:
        file_path = f'./top_100_only_wins_no_folds_per_round/{round.name}/data.csv.bz2'
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(os.path.realpath(Path(file_path).parent), exist_ok=True)
        tmp = df[df[cols.Round_preflop.name.lower()] == 1]
        tmp.to_csv(file_path,
                   index=True,
                   header=True,
                   index_label='label',
                   mode='a',
                   float_format='%.5f',
                   compression='bz2'
                   )


if __name__ == "__main__":
    main()
