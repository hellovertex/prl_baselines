import glob
from functools import partial

import click

from prl.baselines.supervised_learning.training.preprocessor import Preprocessor, to_csv


@click.command
@click.option('--input_dir',
              type=str,
              help='location of .csv files containing vectorized information')
@click.option('--output_dir',
              type=str,
              help='location of where to write .parquet files [optional]')
@click.option('--to_parquet',
              is_flag=True,
              show_default=True,
              default=False,
              help='Whether the preprocessed dataframes should be written to .parquet files.'
                   'Parquet files consume less memory than csv files.')
@click.option('--to_csv',
              is_flag=True,
              show_default=True,
              default=True,
              help='Whether the preprocessed dataframes should be written to .csv files.'
                   '')
@click.option('--skip_preprocessing',
              is_flag=True,
              show_default=True,
              default=True,
              help='Preprocessing can be skipped, e.g. if preprocessed data is already written to disk.')
@click.option("--blind_sizes",
              default="0.25-0.50",
              type=str,
              help="Possible values are e.g. '0.25-0.50', '0.50-1.00', '1.00-2.00'")
def main(input_dir, to_parquet, blind_sizes, output_dir, skip_preprocessing):

    if not skip_preprocessing:
        # write large preprocessed dataset to disk
        to_csv_fn = partial(to_csv, output_dir=output_dir + f'/{blind_sizes}')
        callbacks = [to_csv_fn]
        if to_parquet:
            to_parquet_fn = partial(to_parquet, output_dir=output_dir + f'/{blind_sizes}')
            callbacks.append(to_parquet_fn)
        preprocessor = Preprocessor(path_to_csv_files=input_dir, callbacks=callbacks)
        preprocessor.run()
    # load training data from output_dir (default: data/03_preprocessed)

    # run training



if __name__ == '__main__':
    main()
