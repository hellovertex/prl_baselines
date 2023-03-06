import glob
import os
from pathlib import Path

import pytest

from prl.baselines.supervised_learning.v2.datasets.dataset_config import ActionGenOption, \
    DatasetConfig
from prl.baselines.supervised_learning.v2.datasets.raw_data import \
    make_raw_data_if_not_exists_already
from prl.baselines.supervised_learning.v2.fast_hsmithy_parser import \
    ParseHsmithyTextToPokerEpisode


@pytest.fixture
def dataset_config():
    num_top_players = 20
    nl = 'NL50'
    from_gdrive_id = '18GE6Xw4K1XE2PNiXSyh762mJ5ZCRl2SO'
    make_dataset_for_each_individual = False
    action_generation_option = ActionGenOption. \
        make_folds_from_top_players_with_randomized_hand
    min_showdowns = 10
    data_dir = os.path.join(Path(__file__).parent, 'data')
    dataset_config = DatasetConfig(num_top_players=num_top_players,
                         make_dataset_for_each_individual=make_dataset_for_each_individual,
                         action_generation_option=action_generation_option,
                         min_showdowns=min_showdowns,
                         nl=nl,
                         from_gdrive_id=from_gdrive_id,
                         DATA_DIR=data_dir)
    return dataset_config

def test_parser(dataset_config):
    make_raw_data_if_not_exists_already(dataset_config)
    parser = ParseHsmithyTextToPokerEpisode(dataset_config)
    episodes = []
    files = glob.glob(f'{dataset_config.dir_raw_data_top_players}/**/*.txt',
                          recursive=True)
    for file in files:
        episodes.append(parser.parse_file(file))
    # run some assertions
    a = 1
# def test_episode_encoding():
#     make_vectorized_data_if_not_exists_already(dataset_config=dataset_config,
#                                                use_multiprocessing=True)
