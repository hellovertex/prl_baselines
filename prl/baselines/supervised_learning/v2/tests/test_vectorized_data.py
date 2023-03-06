import os
from pathlib import Path

import pytest

from prl.baselines.supervised_learning.v2.datasets.dataset_config import ActionGenOption, \
    DatasetConfig
import os
from pathlib import Path

import pytest

from prl.baselines.supervised_learning.v2.datasets.dataset_config import ActionGenOption, \
    DatasetConfig


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
