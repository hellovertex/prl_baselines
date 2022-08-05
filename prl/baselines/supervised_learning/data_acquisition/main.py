import click
from prl.environment.Wrappers.prl_wrappers import AugmentObservationWrapper

from csv_writer import CSVWriter
from hsmithy_parser import HSmithyParser
from prl.baselines.supervised_learning.config import DATA_DIR
from prl.baselines.supervised_learning.data_acquisition.runner import Runner
from rl_state_encoder import RLStateEncoder

LOGFILE = DATA_DIR + "log.txt"


@click.command()
# @click.option("--path-to-bulkhands_zip",
#               default=DATA_DIR + "01_raw/0.25-0.50/BulkHands_example.zip",
#               type=str,
#               help="Path to zip file that was provided by hhsmithy.com "
#                    "and contains poker hands.")
@click.option("--zip_path",
              default="/home/sascha/Documents/github.com/prl_baselines/data/01_raw",
              type=str,
              help="Indicates, which folder is searched "
                   "for .zip files to be extracted.")
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
def main(zip_path, blind_sizes, from_gdrive_id, unzipped_dir):
    # Creates PokerEpisode instances from raw .txt files
    parser = HSmithyParser()

    # Steps Steinberger Poker Environment, augments observations and vectorizes them
    encoder = RLStateEncoder(env_wrapper_cls=AugmentObservationWrapper)

    # writes training data from encoder to disk
    writer = CSVWriter(out_filename_base=f'6MAX_{blind_sizes}.txt')

    # Uses the results of parser and encoder to write training data to disk or cloud
    with Runner(parser=parser,
                encoder=encoder,
                writer=writer,
                write_azure=False,
                logfile=LOGFILE) as runner:
        # Looks for .zip files inside folder derived from "which_data_files"
        # or downloads from gdrive. Extracts found .zip files
        # reads the extracted .txt files for poker hands
        # parses, encodes, vectorizes, and writes them to disk.
        runner.run(blind_sizes,
                   unzipped_dir=unzipped_dir,
                   from_gdrive_id=from_gdrive_id)
        # # run example using google drive file id, which also works fine
        # generator.run_data_generation(which_data_files, from_gdrive_id="18GE6Xw4K1XE2PNiXSyh762mJ5ZCRl2SO")


if __name__ == '__main__':
    main()
