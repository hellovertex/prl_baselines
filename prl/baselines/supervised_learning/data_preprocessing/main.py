import glob
from enum import IntEnum
from functools import partial
from pathlib import Path

import click
from prl.environment.Wrappers.base import ActionSpace

from prl.baselines.supervised_learning.config import DATA_DIR
from prl.baselines.supervised_learning.data_preprocessing.callbacks import to_csv
from prl.baselines.supervised_learning.data_preprocessing.preprocessor import Preprocessor
from prl.baselines.supervised_learning.data_preprocessing.preprocessor import DatasetLabelBalanceFactor__PercentageOf

DEFAULT_VECTORIZED_DATA_PATH = str(DATA_DIR) + '/02_vectorized'
DEFAULT_PREPROCESSED_DATA_PATH = str(DATA_DIR) + '/03_preprocessed'


@click.command()
@click.option("--out_dir",
              default="0.25-0.50",
              type=str,
              help="Possible values are e.g. '0.25-0.50', '0.50-1.00', '1.00-2.00'")
@click.option("--path_to_csv_files",
              default="",
              type=str,  # absolute path
              help="Passing path_to_csv_files we can bypass the naming convention "
                   "that will look up data/02_vectorized/{out_dir} for data to preprocess. ")
@click.option("--output_dir",
              default="",
              type=str,  # absolute path
              help="Optionally pass an output dir to circumvent convetion of writing to "
                   "./data/03_preprocessed/{out_dir} ")
@click.option("--use_downsampling",
              default=True,
              type=bool,  # absolute path
              help="If True, each label will be downsampled to the number of all ins, "
                   "so that training data is equally distributed. ")
def main(blind_sizes, path_to_csv_files, output_dir, use_downsampling):
    # player_folders = glob.glob(
    #     "/home/hellovertex/Documents/github.com/prl_baselines/data/02_vectorized/0.25-0.50" "/**/*.csv", recursive=True)
    # for path_to_csv_files in player_folders[1:]:
    # out_dir = "2NL"
    sampling_fractions = [1 for _ in range(len(ActionSpace))]
    percentage_of = DatasetLabelBalanceFactor__PercentageOf.MAX_RAISE_LABELS

    """Want two predefined sets of balanced label frequencies.
    make folder naming accordingly. pipe input dir and append labelling strategy name
    
    """
    if not path_to_csv_files:
        path_to_csv_files = DEFAULT_VECTORIZED_DATA_PATH + f'/{blind_sizes}'
    if not output_dir:
        output_dir = DEFAULT_PREPROCESSED_DATA_PATH + f'/{blind_sizes}'
    subdir = Path(path_to_csv_files).stem
    output_dir += f'/{subdir}'
    callbacks = [partial(to_csv, output_dir=output_dir)]
    preprocessor = Preprocessor(path_to_csv_files,
                                recursive=True)

    use_label_balancing = True
    preprocessor.run(use_label_balancing=use_label_balancing,
                     sampling_fractions=sampling_fractions,
                     percentage_of=percentage_of,
                     callbacks=callbacks)
    # preprocess per player dataset
    # run for and write to individual files
    # preprocess whole dataset


if __name__ == '__main__':
    main()
