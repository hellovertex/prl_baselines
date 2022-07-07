import os
import shutil
import glob
from functools import partial

import click

from prl.baselines.supervised_learning.training.preprocessor import Preprocessor, to_csv, to_parquet
from prl.baselines.supervised_learning.training.train_eval import run_train_eval

DATA_DIR = "../../../../data"
EPOCHS = 8000
LR = 1e-6
RESUME = True
BATCH_SIZE = 50000
TEST_BATCH_SIZE = 10000

@click.command
@click.option('--input_dir',
              type=str,
              help='location of .csv files containing vectorized information '
                   'e.g. C:\\Users\\<...>\\prl_baselines\\data\\02_vectorized\\0.25-0.50\\. '
                   'Note the blind size is part of the input dir because it has been created already."')
@click.option('--output_dir_preprocessing',
              type=str,
              help='location of where to write .parquet files [optional] '
                   'e.g. C:\\Users\\<...>\\prl_baselines\\data\\03_preprocessed\\'
                   'Note the blind size is NOT part of the output_dir because it will be created.')
@click.option('--write_to_parquet',
              is_flag=True,
              show_default=True,
              default=False,
              help='Whether the preprocessed dataframes should be written to .parquet files.'
                   'Parquet files consume less memory than csv files.')
@click.option('--skip_preprocessing',
              is_flag=True,
              show_default=True,
              default=False,
              help='Preprocessing can be skipped, e.g. if preprocessed data is already written to disk.')
@click.option('--skip_testfolder_creation',
              is_flag=True,
              show_default=True,
              default=False)
@click.option("--blind_sizes",
              default="0.25-0.50",
              type=str,
              help="Possible values are e.g. '0.25-0.50', '0.50-1.00', '1.00-2.00'")
def main(input_dir, write_to_parquet, blind_sizes, output_dir_preprocessing, skip_preprocessing,
         skip_testfolder_creation):
    """ Call with `python main.py --output_dir_preprocessing <PATH_TO_DATAFOLDER>/03_preprocessed
    """
    if not input_dir:
        input_dir = DATA_DIR + f'/02_vectorized/{blind_sizes}'
    if not output_dir_preprocessing:
        output_dir_preprocessing = DATA_DIR + f'/03_preprocessed/{blind_sizes}'
    else:
        output_dir_preprocessing += f'/{blind_sizes}'
    run_preprocessing(input_dir=input_dir,
                      output_dir=output_dir_preprocessing,
                      write_to_parquet=write_to_parquet,
                      blind_sizes=blind_sizes,
                      skip_preprocessing=skip_preprocessing)
    if not skip_testfolder_creation:
        make_folder_testdataset(output_dir_preprocessing)

    run_train_eval(input_dir=output_dir_preprocessing,
                   epochs=EPOCHS,
                   batch_size=BATCH_SIZE,
                   test_batch_size=TEST_BATCH_SIZE,
                   lr=LR,
                   resume=RESUME)


def make_folder_testdataset(output_dir_preprocessing):
    # get all datasamples
    files = glob.glob(output_dir_preprocessing.__str__() + '/**/*.csv', recursive=True)
    # move a subset to test directory
    testdir = output_dir_preprocessing + '/test'
    if not os.path.exists(testdir):
        os.makedirs(testdir)
    for testfile in files[-2:]:  # use only two files (2GB)
        try:
            shutil.move(testfile, testdir + '/' + os.path.basename(testfile))
        except shutil.SameFileError:
            # file already exists, thats ok
            pass


def run_preprocessing(input_dir, output_dir, skip_preprocessing, write_to_parquet=False, blind_sizes='0.25-0.50'):
    if not skip_preprocessing:
        # write large preprocessed dataset to disk
        to_csv_fn = partial(to_csv, output_dir=output_dir)
        callbacks = [to_csv_fn]
        if write_to_parquet:
            to_parquet_fn = partial(to_parquet, output_dir=output_dir)
            callbacks.append(to_parquet_fn)
        preprocessor = Preprocessor(path_to_csv_files=input_dir, callbacks=callbacks)
        preprocessor.run()


if __name__ == '__main__':
    main()
