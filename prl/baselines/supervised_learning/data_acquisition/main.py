import ast
import glob
import multiprocessing
import time

import click
from prl.environment.Wrappers.augment import AugmentObservationWrapper

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
              # for small example, use 18GE6Xw4K1XE2PNiXSyh762mJ5ZCRl2SO
              # for complete database (VERY LARGE), use 18kkgEM2CYF_Tl4Dn8oro6tUgqDfr9IAN
              default="18kkgEM2CYF_Tl4Dn8oro6tUgqDfr9IAN",
              type=str,
              help="Google drive id of a .zip file containing poker hands. "
                   "For small example, use 18GE6Xw4K1XE2PNiXSyh762mJ5ZCRl2SO"
                   "For complete database (VERY LARGE), use 18kkgEM2CYF_Tl4Dn8oro6tUgqDfr9IAN"
                   "The id can be obtained from the google drive download-link url."
                   "The runner will try to download the data from gdrive and proceed with unzipping."
                   "If unzipped_dir is passed as an argument, this parameter will be ignored.")
@click.option("--unzipped_dir",
              default="/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data",
              type=str,  # absolute path
              help="Absolute Path. Passing unzipped_dir we can bypass the unzipping step and assume "
                   "files have alredy been unzipped. ")
@click.option("--version_two",
              is_flag=True,
              default=False,
              help="See runner.run docstring for an explanation of what changed with version two.")
@click.option("--use_player_names_as_outdir",
              is_flag=True,
              default=False,
              help="See runner.run docstring for an explanation of what changed with version two.")
def main(blind_sizes, from_gdrive_id, unzipped_dir, version_two, use_player_names_as_outdir):
    """Extracts .zip files found in prl_baselines/data/01_raw -- unless `unzipped_dir` is provided.
     Reads the extracted .txt files and 1) parses, 2) encodes, 3) vectorizes poker hands and 4) writes them to disk.
     The .zip file can also be downloaded from gdrive by providing a gdrive-url."""
    blind_sizes = "2NL"
    use_player_names_as_outdir = False
    version_two = False
    unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/2.5NL/unzipped"
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
        # unzipped_dir = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
        runner.run(blind_sizes,
                   unzipped_dir=unzipped_dir,
                   from_gdrive_id=from_gdrive_id,
                   version_two=version_two,
                   use_outdir_per_player=use_player_names_as_outdir,
                   use_parallelization=False)


def parse_encode_write(filename):
    blind_sizes = "2.5NL"
    parser = HSmithyParser()

    # Steps Steinberger Poker Environment, augments observations and vectorizes them
    encoder = RLStateEncoder(env_wrapper_cls=AugmentObservationWrapper)

    # writes training data from encoder to disk
    writer = CSVWriter(out_filename_base=f'6MAX_{blind_sizes}')
    with open("/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/eda_result_filtered.txt",
              "r") as data:
        player_dict = ast.literal_eval(data.read())
    # Uses the results of parser and encoder to write training data to disk or cloud
    runner = Runner(parser=parser,
                    blind_sizes=blind_sizes,
                    encoder=encoder,
                    writer=writer,
                    write_azure=False,
                    logfile=LOGFILE,
                    use_outdir_per_player=False,
                    only_from_selected_players=False,
                    selected_players=None)
                    #selected_players=list(player_dict.keys()))
    # parse PokerEpisodes, encode, vectorize, write training data and labels to disk
    # unzipped_dir = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    return runner.parse_encode_write(filename)


if __name__ == '__main__':
    main()
    # unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/2.5NL/unzipped"
    # filenames = glob.glob(unzipped_dir.__str__() + '/**/*.txt', recursive=True)
    # parse_encode_write(filename=filenames[0])
    # print(f'Starting job. This may take a while.')
    # start = time.time()
    # p = multiprocessing.Pool()
    # t0 = time.time()
    # for x in p.imap_unordered(parse_encode_write, filenames):
    #     print(x + f'. Took {time.time() - t0} seconds')
    # print(f'Finished job after {time.time() - start} seconds.')
    # p.close()
