import glob
import os
import shutil

import click

from prl.baselines.supervised_learning.config import DATA_DIR
from prl.baselines.supervised_learning.training.from_all_players.train_eval import run_train_eval

EPOCHS = 8000000
LR = 1e-6
RESUME = True
BATCH_SIZE = 3690  # dont make batch size larger than dataset
TEST_BATCH_SIZE = 64


@click.command
@click.option("--out_dir",
              default="0.25-0.50",
              type=str,
              help="Possible values are e.g. '0.25-0.50', '0.50-1.00', '1.00-2.00'")
@click.option("--path_to_datasamples",
              type=str,
              help="Optional if you do not want to use training data from data/03_preprocessed/{out_dir}, "
                   "e.g. if you want to train using data that is not down-sampled.")
@click.option("--path_to_testsamples",
              type=str,
              help="Optional if you do not want to use test data from data/03_preprocessed/{out_dir}/test, "
                   "e.g. if you want to test using data that is not down-sampled.")
def main(blind_sizes, path_to_datasamples, path_to_testsamples):
    """ Call with `python main.py --output_dir_preprocessing <PATH_TO_DATAFOLDER>/03_preprocessed
    """
    if not path_to_datasamples:
        path_to_datasamples = str(DATA_DIR) + '/03_preprocessed' + f'/{blind_sizes}'
    if not path_to_testsamples:
        path_to_testsamples = path_to_datasamples + '/test'
    if not os.path.exists(path_to_testsamples):
        make_folder_testdataset(path_to_datasamples, path_to_testsamples)

    run_train_eval(input_dir=path_to_datasamples,
                   epochs=EPOCHS,
                   batch_size=BATCH_SIZE,
                   test_batch_size=TEST_BATCH_SIZE,
                   lr=LR,
                   resume=RESUME)


def make_folder_testdataset(path_to_datasamples, path_to_testsamples, n_files=1):
    files = glob.glob(path_to_datasamples + '/**/*.csv', recursive=True)
    # move a subset to test directory
    if not os.path.exists(path_to_testsamples):
        os.makedirs(path_to_testsamples)
    for testfile in files[-n_files:]:
        try:
            shutil.move(testfile, path_to_testsamples + '/' + os.path.basename(testfile))
        except shutil.SameFileError:
            # file already exists, thats ok
            pass


if __name__ == '__main__':
    main()
