import click

from prl.baselines.supervised_learning.config import DATA_DIR
from prl.baselines.supervised_learning.training.from_selected_players.train_eval import run_train_eval

EPOCHS = 8000000
LR = 1e-6
RESUME = True
BATCH_SIZE = 3690  # dont make batch size larger than dataset
TEST_BATCH_SIZE = 64


@click.command
@click.option("--blind_sizes",
              default="0.25-0.50",
              type=str,
              help="Possible values are e.g. '0.25-0.50', '0.50-1.00', '1.00-2.00'")
@click.option("--path_to_csv_files",
              type=str,
              help="Optional if you do not want to use training data from data/03_preprocessed/{blind_sizes}, "
                   "e.g. if you want to train using data that is (not) down-sampled.")
def main(blind_sizes, path_to_csv_files):
    if not path_to_csv_files:
        path_to_csv_files = str(DATA_DIR) + '/03_preprocessed' + f'/{blind_sizes}'

    run_train_eval(input_dir=path_to_csv_files,
                   epochs=EPOCHS,
                   batch_size=BATCH_SIZE,
                   test_batch_size=TEST_BATCH_SIZE,
                   lr=LR,
                   resume=RESUME)


if __name__ == '__main__':
    main()
