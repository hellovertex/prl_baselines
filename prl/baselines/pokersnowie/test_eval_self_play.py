from typing import List

import numpy as np
from prl.environment.Wrappers.prl_wrappers import AugmentObservationWrapper

from prl.baselines.supervised_learning.data_acquisition.environment_utils import build_cards_state_dict, \
    init_wrapped_env, make_player_cards, make_board_cards

import prl.environment.steinberger.PokerRL.game._.look_up_table as lut

def get_ranking():
    """
    Evaluates its own hand score and compares with possible opponent’s hands - scores

    "Defined a criterion to fold
    some hands. The defined criterion was that if the Effective Hand Strength [11] is
    below 50%, then the agent has a probability of folding the hand equal to its tightness
    level. The agent will choose the class that has better acceptance level on the current
    game state. If that class acceptance level is below a defined minimum, the agent folds"

    net -> logits
    [11] < 50% -> fold_prob

    [11] > 50% -> argmax of logits
    [11] < 50% -> max(fold_prob, ,max(logits))

    if result < threshold: fold
    :return:
    """
    # dear future me, if you can find the time, please watch
    # https://www.youtube.com/watch?v=TM_sMACxSzY 
    # and
    # https://www.youtube.com/watch?v=_4T6RBa7P0o
    # and consider implementing the evaluation yourself, instead of using Erics 
    # .so blackbox magic 
    # use HS preflop (apparently no board info directly available so HS implicitly includes board info)
    # use EHS postflop as we now have the additional info of flop


def get_cards(obs: np.array, feature_names=None) -> List[str]:
    pass


def test_get_cards():
    # Arrange
    board = '[6h Ts Td 9c Jc]'  # we need specify board to bootstrap env from state dict
    player_hands = ['3h 3c', 'Tc 9s', 'Jd Js', 'Kc Ks', 'Ac Ad', '2h 2c']
    cards_state_dict = build_cards_state_dict(board, player_hands)
    env: AugmentObservationWrapper = init_wrapped_env(env_wrapper_cls=AugmentObservationWrapper,
                                                      stack_sizes=[100, 110, 120, 130, 140, 150])
    feature_names = list(env.obs_idx_dict.keys()) + ["button_index"]
    state_dict = {'deck_state_dict': cards_state_dict}

    # TMP
    lookup = env.env.get_lut_holder()  # todo: rename env.env to env.base_env
    from prl.environment.steinberger.PokerRL.game._.cpp_wrappers.CppHandeval import CppHandeval
    cpp_poker = CppHandeval()
    player_hands = ['7h 2s', '7d 2s', 'Ac As', 'Kc Ks', '2h 2c']
    player_cards = make_player_cards(player_hands)
    board = '['' '' '' '' '']'
    board_cards = make_board_cards(board)
    # b = np.array([[2, 0], [2, 3], [11, 1], [10, 2], [11, 2]], dtype=np.int8)
    # h = np.array([[11, 3], [5, 1]], dtype=np.int8)
    cpp_poker.get_hand_rank_52_holdem(hand_2d=np.array(player_cards[0]), board_2d=np.array(board_cards))

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


def inspect_lut_and_hs():
    from prl.environment.steinberger.PokerRL.game._.cpp_wrappers.CppHandeval import CppHandeval
    cpp_poker = CppHandeval()
    player_hands = ['3h 3c', 'Tc 9s', 'Jd Js', 'Kc Ks', 'Ac Ad', '2h 2c']
    player_cards = make_player_cards(player_hands)
    # string to array
    b = np.array([[2, 0], [2, 3], [11, 1], [10, 2], [11, 2]], dtype=np.int8)
    h = np.array([[11, 3], [5, 1]], dtype=np.int8)

    # have np.array([ [[0, 1], [2,3]],  # first hand
    #                 [[4, 5], [6,7]] ]  # second hand
    #
    pass

if __name__ == "__main__":
    test_get_cards()
    inspect_lut_and_hs()
