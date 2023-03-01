import ast
import dataclasses
import glob
import json
import logging
import os
import re
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


class RawData:
    # todo how to move all 01_raw-data-related pieces here
    #  including download_data, get_top_players, extract_top_players
    def __init__(self,
                 dataset_options: DatasetOptions):
        self.opt = dataset_options
        tmp_data_dir = self.opt.dir_raw_data_all_players
        self.data_files = glob.glob(tmp_data_dir + '**/*.txt')
        self.data_dir = self.opt.dir_raw_data_top_players
        self.n_files = len(self.data_files)

    def _to_disk(self, alias, player_name, hand_histories):
        file_path_out = os.path.join(self.data_dir, alias)
        for current in hand_histories:  # c for current_hand
            if player_name in current:
                result = "PokerStars Hand #" + current
                if not os.path.exists(file_path_out):
                    os.makedirs(file_path_out)
                with open(os.path.join(file_path_out, f'{alias}.txt'),
                          'a+',
                          encoding='utf-8') as f:
                    f.write(result)

    def player_dataset_to_disk(self, target_players):
        for i, file in enumerate(self.data_files):
            if i % 100 == 0: logging.info(
                f'Extracting games for top {self.opt.num_top_players} players'
                f'from file {i}/{self.n_files}')
            with open(file, 'r') as f:
                hand_histories = re.split(r'PokerStars Hand #', f.read())[1:]
                for rank, player_name in enumerate(target_players):
                    alias = f'PlayerRank{str(rank).zfill(3)}'
                    self._to_disk(alias, player_name, hand_histories)

    def generate(self,
                 from_gdrive_id: Optional[str] = None):
        if not self.opt.hand_history_has_been_downloaded_and_unzipped():
            download_data(from_gdrive_id, self.opt.nl)
        if not self.opt.exists_raw_data_for_all_selected_players():
            top_players = get_top_n_players(self.opt.nl, self.opt.num_top_players)
            self.player_dataset_to_disk(list(top_players.keys()))

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
