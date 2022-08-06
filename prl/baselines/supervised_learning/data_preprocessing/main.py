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
def main(blind_sizes, path_to_csv_files):
    if not path_to_csv_files:
        path_to_csv_files = str(DATA_DIR) + '/02_vectorized' + f'/{blind_sizes}'
    callbacks = [partial(to_csv, output_dir=str(DATA_DIR) + '/03_preprocessed' + f'/{blind_sizes}')]
    preprocessor = Preprocessor(path_to_csv_files, callbacks)
    preprocessor.run()
