import click
from prl.environment.Wrappers.prl_wrappers import AugmentObservationWrapper

from csv_writer import CSVWriter
from hsmithy_parser import HSmithyParser
from prl.baselines.supervised_learning.config import LOGFILE
from prl.baselines.supervised_learning.data_acquisition.runner import Runner
from rl_state_encoder import RLStateEncoder


@click.command()
@click.option("--blind_sizes",
              default="0.25-0.50",
              type=str,
              help="Possible values are e.g. '0.25-0.50', '0.50-1.00', '1.00-2.00'")
@click.option("--from_gdrive_id",
              default="",
              type=str,
              help="If a string value is passed, it should contain a DL link for "
                   "google drive to a bulkhands.zip file containing poker hands. "
                   "The generator will try to download the data from there.")
@click.option("--unzipped_dir",
              default="",
              type=str,
              help="Passing unzipped_dir we can bypass the unzipping step and assume "
                   "files have alredy been unzipped. "
                   "In this case, the `zip_path` arg will be ignored.")
def main(blind_sizes, from_gdrive_id, unzipped_dir):
    """Extracts .zip files found in prl_baselines/data/01_raw unless `unzipped_dir` is provided.
     Reads the extracted .txt files and 1) parses, 2) encodes, 3) vectorizes poker hands and 4) writes them to disk.
     The .zip file can also be downloaded from gdrive by providing a gdrive-url."""
    # Creates PokerEpisode instances from raw .txt files
    parser = HSmithyParser()

    # Steps Steinberger Poker Environment, augments observations and vectorizes them
    encoder = RLStateEncoder(env_wrapper_cls=AugmentObservationWrapper)

    # writes training data from encoder to disk
    writer = CSVWriter(out_filename_base=f'6MAX_{blind_sizes}')

    # Uses the results of parser and encoder to write training data to disk or cloud
    with Runner(parser=parser,
                encoder=encoder,
                writer=writer,
                write_azure=False,
                logfile=LOGFILE) as runner:
        # parse PokerEpisodes, encode, vectorize, write training data and labels to disk
        runner.run(blind_sizes, unzipped_dir=unzipped_dir, from_gdrive_id=from_gdrive_id)


if __name__ == '__main__':
    main()
