import os
from pathlib import Path
from typing import List

import numpy as np
import pytest
from prl.environment.Wrappers.utils import init_wrapped_env

from prl.baselines.deprecated.fast_hsmithy_parser import ParseHsmithyTextToPokerEpisode
from prl.baselines.evaluation.utils import get_player_cards
from prl.baselines.supervised_learning.v2.datasets.dataset_config import ActionGenOption, \
    DatasetConfig
from prl.baselines.supervised_learning.v2.poker_model import PokerEpisodeV2
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns, \
    AugmentObservationWrapper
from prl.baselines.supervised_learning.v2.datasets.encoder import EncoderV2
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


@pytest.fixture
def episodes(dataset_config) -> List[PokerEpisodeV2]:
    file = 'hand_histories.txt'
    parser = ParseHsmithyTextToPokerEpisode(dataset_config)
    return parser.parse_file(file)


def test_parser_has_eps(episodes):
    hasep_1 = False
    hasep_2 = False
    for ep in episodes:
        if ep.hand_id == 208958099944:
            hasep_1 = True
        if ep.hand_id == 208958141851:
            hasep_2 = True
    assert hasep_1
    assert hasep_2


def test_vectorized_episode(dataset_config: DatasetConfig,
                            episodes: List[PokerEpisodeV2]):
    dummy_env = init_wrapped_env(AugmentObservationWrapper,
                                 [5000 for _ in range(6)],
                                 blinds=(25, 50),
                                 multiply_by=1)
    enc = EncoderV2(dummy_env)
    opt = dataset_config
    selected_players = ['JuanAlmighty']
    for ep in episodes:
        if ep.hand_id == 208958141851:
            # todo: make sure vectorized form is ok
            obs, act = enc.encode_episode(ep,
                                          a_opt=opt.action_generation_option,
                                          selected_players=selected_players,
                                          limit_num_players=5,
                                          verbose=True)
            # assert cards remain correct after encoding
            cards = get_player_cards(obs[0])[0]
            assert cards == '[As, Ah]'
            assert cards == get_player_cards(obs[1])[0]
            assert cards == get_player_cards(obs[2])[0]
            # assert actions remain correct after encoding
            assert np.array_equal(act, [6,5,1])
            a = 1
            print(obs)
        elif ep.hand_id == 208958099944:
            # todo: make sure vectorized form is ok
            pass
