import os
import shutil
import glob
from functools import partial

import click

from prl.baselines.supervised_learning.training.preprocessor import Preprocessor, to_csv, to_parquet
from prl.baselines.supervised_learning.training.train_eval import run_train_eval

DATA_DIR = "../../../../data"
EPOCHS = 1000
LR = 1e-6
RESUME = True


@click.command
@click.option('--input_dir',
              type=str,
              help='location of .csv files containing vectorized information')
@click.option('--output_dir_preprocessing',
              type=str,
              help='location of where to write .parquet files [optional]')
@click.option('--write_to_parquet',
              is_flag=True,
              show_default=True,
              default=False,
              help='Whether the preprocessed dataframes should be written to .parquet files.'
                   'Parquet files consume less memory than csv files.')
@click.option('--write_to_csv',
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
def main(input_dir, write_to_parquet, blind_sizes, output_dir_preprocessing, skip_preprocessing):
    """ Call with `python main.py --output_dir_preprocessing <PATH_TO_DATAFOLDER>/03_preprocessed
    """
    if not input_dir:
        input_dir = DATA_DIR + f'/02_vectorized/{blind_sizes}'
    if not output_dir_preprocessing:
        output_dir_preprocessing = DATA_DIR + f'/03_preprocessed/{blind_sizes}'
    run_preprocessing(input_dir=input_dir,
                      output_dir=output_dir_preprocessing,
                      write_to_parquet=write_to_parquet,
                      blind_sizes=blind_sizes,
                      skip_preprocessing=skip_preprocessing)
    make_testfolder(output_dir_preprocessing, blind_sizes)

    run_train_eval(input_dir=output_dir_preprocessing,
                   epochs=EPOCHS,
                   lr=LR,
                   resume=RESUME)


def make_testfolder(output_dir_preprocessing, blind_sizes):
    # get all datasamples, by convention they are stored in <output_dir_preprocessing>/<blind_sizes>
    output_dir_preprocessing = output_dir_preprocessing + f'/{blind_sizes}'
    files = glob.glob(output_dir_preprocessing.__str__() + '/**/*.csv', recursive=True)
    # move a subset to test directory
    testdir = output_dir_preprocessing + '/test'
    if not os.exists(testdir):
        os.makedirs(testdir)
    for testfile in files[-2:]:  # use only two files (2GB)
        shutil.copyfile(testfile, testdir + os.path.basename(testfile))


def run_preprocessing(input_dir, output_dir, skip_preprocessing, write_to_parquet=False, blind_sizes='0.25-0.50'):
    if not skip_preprocessing:
        # write large preprocessed dataset to disk
        to_csv_fn = partial(to_csv, output_dir=output_dir + f'/{blind_sizes}')
        callbacks = [to_csv_fn]
        if write_to_parquet:
            to_parquet_fn = partial(to_parquet, output_dir=output_dir + f'/{blind_sizes}')
            callbacks.append(to_parquet_fn)
        preprocessor = Preprocessor(path_to_csv_files=input_dir, callbacks=callbacks)
        preprocessor.run()


if __name__ == '__main__':
    main()
