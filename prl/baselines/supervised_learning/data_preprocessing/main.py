import glob
from functools import partial

import click

from prl.baselines.supervised_learning.config import DATA_DIR
from prl.baselines.supervised_learning.data_preprocessing.callbacks import to_csv
from prl.baselines.supervised_learning.data_preprocessing.preprocessor import Preprocessor

DEFAULT_VECTORIZED_DATA_PATH = str(DATA_DIR) + '/02_vectorized'
DEFAULT_PREPROCESSED_DATA_PATH = str(DATA_DIR) + '/03_preprocessed'


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
              help="Optionally pass an output dir to circumvent convetion of writing to "
                   "./data/03_preprocessed/{blind_sizes} ")
@click.option("--use_downsampling",
              default=True,
              type=bool,  # absolute path
              help="If True, each label will be downsampled to the number of all ins, "
                   "so that training data is equally distributed. ")
def main(blind_sizes, path_to_csv_files, output_dir, use_downsampling):
    # player_folders = glob.glob(
    #     "/home/hellovertex/Documents/github.com/prl_baselines/data/02_vectorized/0.25-0.50" "/**/*.csv", recursive=True)
    # for path_to_csv_files in player_folders[1:]:
    # blind_sizes = "2NL"
    use_downsampling = False
    if not path_to_csv_files:
        path_to_csv_files = DEFAULT_VECTORIZED_DATA_PATH + f'/{blind_sizes}'
        path_to_csv_files = "/home/hellovertex/Documents/github.com/prl_baselines/data/02_vectorized/0.25-0.50_no_folds"
    if not output_dir:
        output_dir = DEFAULT_PREPROCESSED_DATA_PATH + f'/{blind_sizes}'
        output_dir = DEFAULT_PREPROCESSED_DATA_PATH + f'/no_folds'
    output_dir =output_dir
    callbacks = [partial(to_csv, output_dir=output_dir)]
    preprocessor = Preprocessor(path_to_csv_files, recursive=True)
    preprocessor.run(use_downsampling=use_downsampling, callbacks=callbacks)


if __name__ == '__main__':
    main()
