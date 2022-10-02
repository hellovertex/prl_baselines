from typing import List

import numpy as np
from prl.environment.Wrappers.prl_wrappers import AugmentObservationWrapper

from prl.baselines.supervised_learning.data_acquisition.environment_utils import build_cards_state_dict, \
    init_wrapped_env


def get_cards(obs: np.array) -> List[str]:
    pass


def test_get_cards():
    # Arrange
    board = '[6h Ts Td 9c Jc]'  # we need specify board to bootstrap env from state dict
    player_hands = ['3h 3c', 'Tc 9s', 'Jd Js', 'Kc Ks', 'Ac Ad', '2h 2c']
    cards_state_dict = build_cards_state_dict(board, player_hands)
    env = init_wrapped_env(env_wrapper_cls=AugmentObservationWrapper,
                           stack_sizes=[100, 110, 120, 130, 140, 150])
    state_dict = {'deck_state_dict': cards_state_dict}

    expected_cards_p0 = None
    expected_cards_p1 = None
    expected_cards_p2 = None
    expected_cards_p3 = None
    expected_cards_p4 = None
    expected_cards_p5 = None

    # Act
    obs, _, done, _ = env.reset(config=state_dict)
    # parse initial observation back to cards

    # step 5 time and check the other players cards

    # Assert
    assert True


def test_get_stub():
    # Arrange
    # Act
    # Assert
    assert True


if __name__ == "__main__":
    test_get_cards()
