import pandas as pd
import glob
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as cols

def main():
    path_to_csv_files = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/v2/dataset"
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

    df_preflop = df[df[cols.Round_preflop.name] == 1]
    df_flop = df[df[cols.Round_flop.name] == 1]
    df_turn = df[df[cols.Round_turn.name] == 1]
    df_river = df[df[cols.Round_river.name] == 1]
    df = df.sample(frac=1)
    a = 1

if __name__ == "__main__":
    main()