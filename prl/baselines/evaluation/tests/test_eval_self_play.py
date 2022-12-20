from typing import List

import numpy as np
from prl.baselines.supervised_learning.data_acquisition.environment_utils import build_cards_state_dict
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env

IDX_C0_0 = 167  # feature_names.index('0th_player_card_0_rank_0')
IDX_C0_1 = 184  # feature_names.index('0th_player_card_1_rank_0')
IDX_C1_0 = 184  # feature_names.index('0th_player_card_1_rank_0')
IDX_C1_1 = 201  # feature_names.index('1th_player_card_0_rank_0')
IDX_BOARD_START = 82  #
IDX_BOARD_END = 167  #
N_FEATURES = 564
CARD_BITS = np.array(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c'])
BOARD_BITS = np.array(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
                       'h', 'd', 's', 'c', '2', '3', '4', '5', '6', '7', '8', '9', 'T',
                       'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c', '2', '3', '4', '5', '6',
                       '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c', '2',
                       '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h',
                       'd', 's', 'c', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J',
                       'Q', 'K', 'A', 'h', 'd', 's', 'c'])
SUITS_HAND_EVALUATOR = ['s', 'h', 'd', 'c']
RANKS_HAND_EVALUATOR = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
CARDS_HAND_EVALUTOR_1D = ['As',
                          'Ah',
                          'Ad',
                          'Ac',
                          'Ks',
                          'Kh',
                          'Kd',
                          'Kc',
                          'Qs',
                          'Qh',
                          'Qd',
                          'Qc',
                          'Js',
                          'Jh',
                          'Jd',
                          'Jc',
                          'Ts',
                          'Th',
                          'Td',
                          'Tc',
                          '9s',
                          '9h',
                          '9d',
                          '9c',
                          '8s',
                          '8h',
                          '8d',
                          '8c',
                          '7s',
                          '7h',
                          '7d',
                          '7c',
                          '6s',
                          '6h',
                          '6d',
                          '6c',
                          '5s',
                          '5h',
                          '5d',
                          '5c',
                          '4s',
                          '4h',
                          '4d',
                          '4c',
                          '3s',
                          '3h',
                          '3d',
                          '3c',
                          '2s',
                          '2h',
                          '2d',
                          '2c']
DICT_CARDS_HAND_EVALUATOR = {'As': 0,
                             'Ah': 1,
                             'Ad': 2,
                             'Ac': 3,
                             'Ks': 4,
                             'Kh': 5,
                             'Kd': 6,
                             'Kc': 7,
                             'Qs': 8,
                             'Qh': 9,
                             'Qd': 10,
                             'Qc': 11,
                             'Js': 12,
                             'Jh': 13,
                             'Jd': 14,
                             'Jc': 15,
                             'Ts': 16,
                             'Th': 17,
                             'Td': 18,
                             'Tc': 19,
                             '9s': 20,
                             '9h': 21,
                             '9d': 22,
                             '9c': 23,
                             '8s': 24,
                             '8h': 25,
                             '8d': 26,
                             '8c': 27,
                             '7s': 28,
                             '7h': 29,
                             '7d': 30,
                             '7c': 31,
                             '6s': 32,
                             '6h': 33,
                             '6d': 34,
                             '6c': 35,
                             '5s': 36,
                             '5h': 37,
                             '5d': 38,
                             '5c': 39,
                             '4s': 40,
                             '4h': 41,
                             '4d': 42,
                             '4c': 43,
                             '3s': 44,
                             '3h': 45,
                             '3d': 46,
                             '3c': 47,
                             '2s': 48,
                             '2h': 49,
                             '2d': 50,
                             '2c': 51}


def playground():
    pass
    # lh = env.env.get_lut_holder()  # todo: rename env.env to env.base_env
    # from prl.environment.steinberger.PokerRL.game._.cpp_wrappers.CppHandeval import CppHandeval
    # cpp_poker = CppHandeval()
    # player_hands = ['7d 2h', '2h 7d', 'Ac As', 'Kc Ks', '2h 2c']
    # player_cards = make_player_cards(player_hands)
    # board = '['' '' '' '' '']'
    # board_cards = make_board_cards(board)
    # lh.get_1d_card(player_cards[0])
    # # b = np.array([[2, 0], [2, 3], [11, 1], [10, 2], [11, 2]], dtype=np.int8)
    # # h = np.array([[11, 3], [5, 1]], dtype=np.int8)
    # cpp_poker.get_hand_rank_52_holdem(hand_2d=np.array(player_cards[0]), board_2d=np.array(board_cards))
    # rank_all_fn can actually compute an array of ranks with shape N_BOARDS, 1326
    # wowzers
    # rank_all_fn = env.env.get_hand_rank_all_hands_on_given_boards([lh.get_1d_card(card2d) for card2d in make_board_cards('[6h Ts Td 9c Jc]')], lh)

    # since (50 choose 2) up to (50 choose 10) for 2 to 5 players does not scale very well (1k to 10B),
    # if we wanted to include the board that would be (50 choose 15) = 2.25e12
    # were just going to run monte-carlo simulations


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
    # todo: contra monte-carlo additional dependency on reuter repo
    #  better: vectorize hand evaluation to compute ehs for multiple players
    # dear future me, if you can find the time, please watch
    # https://www.youtube.com/watch?v=TM_sMACxSzY 
    # and
    # https://www.youtube.com/watch?v=_4T6RBa7P0o
    # and consider implementing the evaluation yourself, instead of using Erics 
    # .so blackbox magic 
    # use HS preflop (apparently no board info directly available so HS implicitly includes board info)
    # use EHS postflop as we now have the additional info of flop
    # run monte carlo will be least effort
    # equity: 210 total pot with ehs of .3 then you should have less than 63 chips in
    # equity is a percentage 63 / 210 but that is equal to ehs so EHS = Equity
    # EV is .3 * 147  - .7 * 63 an integer


def get_cards(obs: np.array, feature_names=None) -> List[str]:
    pass


def test_get_cards():
    # Arrange
    board = '[Ks 3h 2h Ah  ]'  # we need specify board to bootstrap env from state dict
    player_hands = ['2d 2c', 'Ac As']
    cards_state_dict = build_cards_state_dict(board, player_hands)
    env: AugmentObservationWrapper = init_wrapped_env(env_wrapper_cls=AugmentObservationWrapper,
                                                      stack_sizes=[140, 150])
    feature_names = list(env.obs_idx_dict.keys()) + ["button_index"]
    state_dict = {'deck_state_dict': cards_state_dict}

    obs, _, done, _ = env.reset(config=state_dict)
    obs, _, done, _ = env.step((1, -1))
    obs, _, done, _ = env.step((1, -1))
    obs, _, done, _ = env.step((1, -1))
    obs, _, done, _ = env.step((1, -1))

    # Act
    board_mask = obs[IDX_BOARD_START:IDX_BOARD_END].astype(int)
    board = BOARD_BITS[board_mask.astype(bool)]
    board_cards = []
    for i in range(0, sum(board_mask) - 1, 2):  # sum is 6,8,10 for flop turn river resp.
        board_cards.append(DICT_CARDS_HAND_EVALUATOR[board[i] + board[i + 1]])

    # Assert
    assert board_cards[0] == 4
    assert board_cards[1] == 45
    assert board_cards[2] == 49
    assert board_cards[3] == 1


def test_get_stub():
    # Arrange
    # Act
    # Assert
    assert True


if __name__ == "__main__":
    test_get_cards()
