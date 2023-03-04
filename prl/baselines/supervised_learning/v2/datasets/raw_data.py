import glob
import logging
import os
import re
from typing import Optional

import click
import gdown
from tqdm import tqdm

from prl.baselines import DATA_DIR
from prl.baselines.supervised_learning.v2.datasets.dataset_options import DatasetOptions
from prl.baselines.supervised_learning.v2.datasets.top_player_selector import \
    TopPlayerSelector
from prl.baselines.supervised_learning.v2.fast_hsmithy_parser import \
    ParseHsmithyTextToPokerEpisode
from prl.baselines.utils.unzip_recursively import extract


def download_data(from_gdrive_id, nl: str) -> bool:
    logging.info(f'No hand histories found in data/01_raw/all_players. '
                 f'Downloading using gdrive_id {from_gdrive_id}')

    path_to_zipfile = os.path.join(DATA_DIR, *['00_tmp', 'bulk_hands.zip'])
    # 1. download .zip file from gdrive to disk
    gdown.download(id=from_gdrive_id,
                   # must end with .zip
                   output=path_to_zipfile,
                   quiet=False)
    # 2. unzip recursively to DATA_DIR
    zipfiles = glob.glob(path_to_zipfile, recursive=False)
    unzipped_dir = os.path.join(DATA_DIR, *['01_raw', nl, 'all_players'])
    [extract(zipfile, out_dir=unzipped_dir) for zipfile in zipfiles]

    logging.info(f'Unzipped hand histories to {unzipped_dir}')
    return True


class RawData:
    def __init__(self,
                 dataset_options: DatasetOptions,
                 top_player_selector: TopPlayerSelector):
        self.opt = dataset_options
        self.top_player_selector = top_player_selector
        tmp_data_dir = self.opt.dir_raw_data_all_players
        self.data_files = glob.glob(tmp_data_dir + '**/*.txt')
        assert self.data_files, "Data Files must not be empty "
        self.data_dir = self.opt.dir_raw_data_top_players
        self.n_files = len(self.data_files)

    def _to_disk(self, alias, player_name, hand_histories):
        file_path_out = os.path.join(self.data_dir, alias)
        file = os.path.join(file_path_out, f'{alias}.txt')
        for current in hand_histories:  # c for current_hand
            if player_name in current:
                result = "PokerStars Hand #" + current
                if not os.path.exists(file_path_out):
                    os.makedirs(file_path_out)
                with open(file, 'a+', encoding='utf-8') as f:
                    f.write(result)

    def player_dataset_to_disk(self, target_players):
        for rank, player_name in enumerate(target_players):
            alias = f'PlayerRank{str(rank + 1).zfill(3)}'
            if os.path.exists(os.path.join(self.data_dir, alias)):
                continue

            logging.info(f'Writing hand history for {alias}')
            for file in tqdm(self.data_files):
                with open(file, 'r') as f:
                    try:
                        hand_histories = re.split(r'PokerStars Hand #', f.read())[1:]
                    except UnicodeDecodeError:
                        # very few files have invalid continuation bytes, skip them
                        continue
                    self._to_disk(alias, player_name, hand_histories)

    def generate(self,
                 from_gdrive_id: Optional[str] = None):
        if from_gdrive_id is None:
            from_gdrive_id = self.opt.from_gdrive_id

        # Maybe download + unzip
        if not self.opt.hand_history_has_been_downloaded_and_unzipped():
            assert from_gdrive_id, "Downloading data requires parameter `from_gdrive_id`"
            download_data(from_gdrive_id, self.opt.nl)

        # Maybe extract Top N players hand_histories
        if not self.opt.exists_raw_data_for_all_selected_players():
            top_players = self.top_player_selector.get_top_n_players_min_showdowns(
                self.opt.num_top_players, self.opt.min_showdowns)
            self.player_dataset_to_disk(list(top_players.keys()))


@click.command()
@click.option("--num_top_players", default=20,
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
    parser = ParseHsmithyTextToPokerEpisode(nl=nl)
    dataset_options = DatasetOptions(num_top_players, nl)
    top_player_selector = TopPlayerSelector(parser)
    raw_data = RawData(dataset_options, top_player_selector)
    raw_data.generate(from_gdrive_id)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
