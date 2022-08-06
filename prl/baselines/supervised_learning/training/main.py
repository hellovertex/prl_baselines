import glob
import os
import shutil

import click

from prl.baselines.supervised_learning.config import DATA_DIR
from prl.baselines.supervised_learning.training.train_eval import run_train_eval

EPOCHS = 8000
LR = 1e-6
RESUME = True
BATCH_SIZE = 50000
TEST_BATCH_SIZE = 10000


@click.command
@click.option("--blind_sizes",
              default="0.25-0.50",
              type=str,
              help="Possible values are e.g. '0.25-0.50', '0.50-1.00', '1.00-2.00'")
def main(blind_sizes):
    """ Call with `python main.py --output_dir_preprocessing <PATH_TO_DATAFOLDER>/03_preprocessed
    """
    path_to_test_data = str(DATA_DIR) + '/03_preprocessed' + f'/{blind_sizes}/test/'
    if not path_to_test_data:
        make_folder_testdataset(blind_sizes)

    run_train_eval(input_dir=output_dir_preprocessing,
                   epochs=EPOCHS,
                   batch_size=BATCH_SIZE,
                   test_batch_size=TEST_BATCH_SIZE,
                   lr=LR,
                   resume=RESUME)


def make_folder_testdataset(blind_sizes):
    # get all datasamples
    path_to_datasamples = str(DATA_DIR) + '/03_preprocessed' + f'/{blind_sizes}'
    path_to_testsamples = path_to_datasamples + '/test'
    files = glob.glob(path_to_datasamples + '/**/*.csv', recursive=True)
    # move a subset to test directory
    if not os.path.exists(path_to_testsamples):
        os.makedirs(path_to_testsamples)
    for testfile in files[-2:]:  # use only two files (2GB)
        try:
            shutil.move(testfile, path_to_testsamples + '/' + os.path.basename(testfile))
        except shutil.SameFileError:
            # file already exists, thats ok
            pass


if __name__ == '__main__':
    main()
