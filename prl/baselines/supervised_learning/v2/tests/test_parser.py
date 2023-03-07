import os
from pathlib import Path
from typing import List

import pytest

from prl.baselines.supervised_learning.v2.datasets.dataset_config import ActionGenOption, \
    DatasetConfig
from prl.baselines.supervised_learning.v2.fast_hsmithy_parser import \
    ParseHsmithyTextToPokerEpisode
from prl.baselines.supervised_learning.v2.poker_model import PokerEpisodeV2


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


def test_parser_ep_money_won(episodes):
    for ep in episodes:
        if ep.hand_id == 208958099944:
            names = ['SWING BOLOO', 'romixon36', 'supersimple2018', 'Flyyguyy403',
                     'Clamfish0', 'JuanAlmighty']
            for name, player in ep.players.items():
                assert name in names
                if name == 'SWING BOLOO':
                    assert player.money_won_this_round == 2185 + 115 - 750
                elif name == 'supersimple2018':
                    assert player.money_won_this_round == -750
                elif name == 'Flyyguyy403':
                    assert player.money_won_this_round == -50
                elif name == 'romixon36' or name == 'Clamfish0' or name == 'romixon36':
                    assert player.money_won_this_round == 0
        elif ep.hand_id == 208958141851:
            names = ['SWING BOLOO', 'romixon36', 'supersimple2018', 'Flyyguyy403',
                     'Clamfish0', 'JuanAlmighty']
            for name, player in ep.players.items():
                assert name in names
                if name == 'Flyyguyy403':
                    assert player.money_won_this_round == -25
                elif name == 'Clamfish0':
                    assert player.money_won_this_round == -50
                elif name == 'JuanAlmighty':
                    assert player.money_won_this_round == 9875 - (1250 + 3750)
                elif name == 'supersimple2018':
                    assert player.money_won_this_round == -5000
        elif ep.hand_id == 208959234900:
            assert ep.board == '[Jc 6d Jd 6h]'


def test_parser_cards(episodes):
    for ep in episodes:
        if ep.hand_id == 208958141851:
            for pname, player in ep.players.items():
                if pname == 'supersimple2018':
                    assert player.cards == '[Qs Qd]'
                elif pname == 'JuanAlmighty':
                    assert player.cards == '[As Ah]'
                else:
                    assert player.cards is None
