import os
from pathlib import Path

import pytest

from prl.baselines.supervised_learning.v2.datasets.dataset_config import ActionGenOption, \
    DatasetConfig


@pytest.fixture
def default_conf():
    num_top_players = 20
    nl = 'NL50'
    from_gdrive_id = '18GE6Xw4K1XE2PNiXSyh762mJ5ZCRl2SO'
    make_dataset_for_each_individual = True
    action_generation_option = ActionGenOption. \
        make_folds_from_top_players_with_randomized_hand
    min_showdowns = 10
    data_dir = os.path.join(Path(__file__).parent, 'data')
    return DatasetConfig(num_top_players=num_top_players,
                         make_dataset_for_each_individual=make_dataset_for_each_individual,
                         action_generation_option=action_generation_option,
                         min_showdowns=min_showdowns,
                         nl=nl,
                         from_gdrive_id=from_gdrive_id,
                         DATA_DIR=data_dir)


def test_exists_vectorized_data_for_all_selected_players(default_conf):
    data_dir = default_conf.DATA_DIR
    a = 1
    # dont need to test this atm
