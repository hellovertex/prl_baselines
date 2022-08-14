from functools import partial

import click

from prl.baselines.supervised_learning.config import DATA_DIR
from prl.baselines.supervised_learning.data_preprocessing.callbacks import to_csv
from prl.baselines.supervised_learning.data_preprocessing.preprocessor import Preprocessor


@click.command()
@click.option("--blind_sizes",
              default="0.25-0.50",
              type=str,
              help="Possible values are e.g. '0.25-0.50', '0.50-1.00', '1.00-2.00'")
@click.option("--path_to_csv_files",
              default="",
              type=str,  # absolute path
              help="Passing path_to_csv_files we can bypass the naming convention "
                   "that will look up data/02_vectorized/{blind_sizes} for data to preprocess. ")
@click.option("--output_dir",
              default="",
              type=str,  # absolute path
              help="Optionally pass an output dir to circumvent convetion of writing to ./data/03_preprocessed ")
@click.option("--use_downsampling",
              default=True,
              type=bool,  # absolute path
              help="Optionally pass an output dir to circumvent convetion of writing to ./data/03_preprocessed ")
def main(blind_sizes, path_to_csv_files, output_dir, use_downsampling):
    if not path_to_csv_files:
        path_to_csv_files = str(DATA_DIR) + '/02_vectorized' + f'/{blind_sizes}'
    if not output_dir:
        output_dir = str(DATA_DIR) + '/03_preprocessed' + f'/{blind_sizes}'
    callbacks = [partial(to_csv, output_dir=output_dir)]
    preprocessor = Preprocessor(path_to_csv_files, callbacks)
    preprocessor.run(use_downsampling=use_downsampling)


if __name__ == '__main__':
    main()
