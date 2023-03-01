import ast
import dataclasses
import glob
import json
import logging
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import click
import gdown
import pandas as pd

from prl.baselines import DATA_DIR
from prl.baselines.supervised_learning.v2.datasets.dataset_options import DatasetOptions
from prl.baselines.supervised_learning.v2.fast_hsmithy_parser import \
    ParseHsmithyTextToPokerEpisode
from prl.baselines.utils.unzip_recursively import extract


@dataclass
class PlayerSelection:
    # don't need these to select top players
    # n_hands_dealt: int
    # n_hands_played: int
    n_showdowns: int
    n_won: int
    total_earnings: float


def download_data(from_gdrive_id, nl: str) -> bool:
    path_to_zipfile = os.path.join(DATA_DIR, *['00_tmp', 'bulk_hands.zip'])
    # 1. download .zip file from gdrive to disk
    gdown.download(id=from_gdrive_id,
                   # must end with .zip
                   output=path_to_zipfile,
                   quiet=False)
    # 2. unzip recursively to DATA_DIR
    zipfiles = glob.glob(path_to_zipfile, recursive=False)
    unzipped_dir = os.path.join(DATA_DIR, *['01_raw', 'all_players', nl])
    [extract(zipfile, out_dir=unzipped_dir) for zipfile in zipfiles]
    logging.info(f'Unzipped hand histories to {unzipped_dir}')
    return True


def get_players_showdown_stats(parser):
    players: Dict[str, PlayerSelection] = {}

    for hand_histories in parser.parse_hand_histories():
        for episode in hand_histories:
            if not episode.has_showdown:
                continue
            # for each player update relevant stats
            for player in episode.showdown_players:
                if not player.name in players:
                    players[player.name] = PlayerSelection(n_showdowns=0,
                                                           n_won=0,
                                                           total_earnings=0)
                else:
                    players[player.name].n_showdowns += 1
                    # can be negative
                    players[player.name].total_earnings += player.money_won_this_round
                    for winner in episode.winners:
                        if winner.name == player.name:
                            players[player.name].n_won += 1
    return players


def get_top_n_players(nl, num_top_players) -> Dict:
    # parse all files
    parser = ParseHsmithyTextToPokerEpisode(nl=nl)
    players = get_players_showdown_stats(parser)
    # sort for winnings per showdown
    serialized = dict([(k, dataclasses.asdict(v)) for k, v in players.items()])
    df = pd.DataFrame.from_dict(serialized, orient='index')
    df = df['total_earnings'] / df['n_showdowns']
    df = df.sort_values(0, ascending=False).dropna()
    if len(df) < num_top_players:
        warnings.warn(f'{num_top_players} top players were '
                      f'requests, but only {len(df)} players '
                      f'are in database. Please decrease '
                      f'`num_top_players` or provide more '
                      f'hand histories.')
    return df[:num_top_players].to_dict()

class WritePlayerSelectionDataSubset:
    def _extract_hands(self, hands_played):
        for current in hands_played:  # c for current_hand
            # Only parse hands that went to Showdown stage, i.e. were shown
            if not '*** SHOW DOWN ***' in current:
                continue

            for target_player in self.target_players:
                if target_player in current:
                    result = "PokerStars Hand #" + current
                    if not os.path.exists(self.file_path_out):
                        os.makedirs(self.file_path_out)
                    with open(f'{self.file_path_out + "/" + target_player}.txt', 'a+',
                              encoding='utf-8') as f:
                        f.write(result)

    def extract_file(self, file_path_in, file_path_out, target_players):
        self._variant = 'NoLimitHoldem'
        self.target_players = target_players
        self.file_path_out = file_path_out

        with open(file_path_in, 'r',
                  encoding='utf-8') as f:  # pylint: disable=invalid-name,unspecified-encoding
            hand_database = f.read()
            hands_played = re.split(r'PokerStars Hand #', hand_database)[1:]
            self._extract_hands(hands_played)

    def extract_files(self,
                      fpaths,
                      file_path_out,
                      target_players):
        n_files = len(fpaths)
        n_files_skipped = 0
        for i, f in enumerate(fpaths[:-1]):
            print(f'Extracting file {i}/{n_files}....')
            try:
                self.extract_file(f, file_path_out, target_players)
            except UnicodeDecodeError:
                n_files_skipped += 1
        return f"Success. Skipped {n_files_skipped} / {n_files}."


class RawData:
    # todo how to move all 01_raw-data-related pieces here
    #  including download_data, get_top_players, extract_top_players
    def __init__(self, dataset_options: DatasetOptions):
        self.opt = dataset_options

    def generate(self,
                 from_gdrive_id: Optional[str] = None):
        if not self.opt.hand_history_has_been_downloaded_and_unzipped():
            download_data(from_gdrive_id, self.opt.nl)
        if not self.opt.exists_raw_data_for_all_selected_players():
            # 1. need to get top n players make separate main at some point
            # 1a) make list of n best players and write it to 00_tmp
            top_players = get_top_n_players(self.opt.nl, self.opt.num_top_players)
            # 2 need to run hsmithy extractor (create HandHistoryExtractorV2 that runs on
            a = 1

            # new directory structure
            # todo: implement and test if successfull


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
    dataset_options = DatasetOptions(num_top_players, nl)
    raw_data = RawData(dataset_options)
    raw_data.generate(from_gdrive_id)


if __name__ == '__main__':
    main()
