import os
from pathlib import Path
from typing import List

import numpy as np
import pytest
from prl.environment.Wrappers.utils import init_wrapped_env
from prl.environment.Wrappers.vectorizer import AgentObservationType

from prl.baselines.deprecated.fast_hsmithy_parser import ParseHsmithyTextToPokerEpisode
from prl.baselines.evaluation.utils import get_player_cards, get_board_cards
from prl.baselines.supervised_learning.v2.datasets.dataset_config import ActionGenOption, \
    DatasetConfig
from prl.baselines.supervised_learning.v2.datasets.vectorized_data import \
    make_vectorized_data_if_not_exists_already
from prl.baselines.supervised_learning.v2.poker_model import PokerEpisodeV2
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns, \
    AugmentObservationWrapper
from prl.baselines.supervised_learning.v2.datasets.encoder import EncoderV2
import pytest


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
