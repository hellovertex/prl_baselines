import glob
import os
from typing import Dict

import click
import gdown

from prl.baselines import DATA_DIR
from prl.baselines.supervised_learning.v2.datasets.dataset_options import DatasetOptions
from prl.baselines.supervised_learning.v2.fast_hsmithy_parser import \
    ParseHsmithyTextToPokerEpisode
from prl.baselines.utils.unzip_recursively import extract


def download_data(from_gdrive_id) -> bool:
    path_to_zipfile = os.path.join(DATA_DIR, *['00_tmp', 'bulk_hands.zip'])
    # 1. download .zip file from gdrive to disk
    gdown.download(id=from_gdrive_id,
                   # must end with .zip
                   output=path_to_zipfile,
                   quiet=False)
    # 2. unzip recursively to DATA_DIR
    zipfiles = glob.glob(path_to_zipfile, recursive=False)
    unzipped_dir = os.path.join(DATA_DIR, *['01_raw', 'all_players', 'NL50'])
    [extract(zipfile, out_dir=unzipped_dir) for zipfile in zipfiles]
    return True


def get_top_n_players(nl, num_top_players) -> Dict:
    # parse all files
    parser = ParseHsmithyTextToPokerEpisode(nl=nl)
    for hand_histories in parser.parse_hand_histories():
        for hand in hand_histories:
            if not hand.has_showdown:
                continue
            # for each player update relevant stats
            # top earners
            pass
    return {}


@click.command()
@click.option("--num_top_players", default=10,
              type=int,
              help="How many top players hand histories should be used to generate the "
                   "data.")
@click.option("--nl",
              default='NL50',
              type=str,
              help="Which stakes the hand history belongs to."
                   "Determines the data directory.")
@click.option("--from_gdrive_id",
              default="18GE6Xw4K1XE2PNiXSyh762mJ5ZCRl2SO",
              type=str,
              help="Google drive id of a .zip file containing hand histories. "
                   "For small example, use 18GE6Xw4K1XE2PNiXSyh762mJ5ZCRl2SO"
                   "For complete database (VERY LARGE), use "
                   "18kkgEM2CYF_Tl4Dn8oro6tUgqDfr9IAN"
                   "The id can be obtained from the google drive download-link url."
                   "The runner will try to download the data from gdrive and proceed "
                   "with unzipping.")
def main(num_top_players, nl, from_gdrive_id):
    opt = DatasetOptions(num_top_players, nl)
    if not opt.hand_history_has_been_downloaded_and_unzipped():
        download_data(from_gdrive_id)
    if not opt.exists_raw_data_for_all_selected_players():
        # 1. need to get top n players make separate main at some point
        # 1a) make list of n best players and write it to 00_tmp
        top_players = get_top_n_players(opt.nl, opt.num_top_players)
        # 2 need to run hsmithy extractor (create HandHistoryExtractorV2 that runs on
        # new directory structure
        # todo: implement and test if successfull

if __name__ == '__main__':
    main()
