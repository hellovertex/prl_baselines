from typing import Tuple

import numpy as np
from prl.environment.steinberger.PokerRL import NoLimitHoldem, Poker

from prl.baselines.supervised_learning.data_acquisition.core.encoder import PlayerInfo
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode
from prl.baselines.supervised_learning.data_acquisition.environment_utils import build_cards_state_dict


def test_get_cards():
    # Arrange
    # create card deck
    board = '[6h Ts Td 9c Jc]'
    player_hands = ['3h 3c',  'Tc 9s', 'Jd Js', 'Kc Ks', 'Ac Ad']
    card_state_dict = build_cards_state_dict(board, player_hands)
    # reset enviornment

    expected_cards_p0 = None
    expected_cards_p0 = None

    # Act
    # parse initial observation back to cards
    # step one time and check the other players cards

    # Assert
    assert True


def test_get_stub():
    # Arrange
    # Act
    # Assert
    assert True


if __name__ == "__main__":
    test_get_cards()
