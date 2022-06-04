import click
from train_eval import train_eval_fn

BLIND_SIZES = "0.25-0.50"


@click.command()
@click.option("--training_dir",
              type=str,
              default=f"../../../../data/02_vectorized/{BLIND_SIZES}/",
              help="Directory that points to the set of .csv-files "
                   "which should be used to read training data from.")
def main(training_dir):
    print('running main')
    train_eval_fn("C:\\Users\\hellovertex\\Documents\\github.com\\dev.azure.com\\prl\\prl_baselines\\data\\02_vectorized\\0.25-0.50")


if __name__ == '__main__':
    main()
