import time

from hand_evaluator import rank
from typing import List
from numba import jit, njit
import numba as nb

import numpy as np
from prl.environment.Wrappers.prl_wrappers import AugmentObservationWrapper
from prl.baselines.supervised_learning.data_acquisition.environment_utils import build_cards_state_dict, \
    init_wrapped_env, make_player_cards, make_board_cards
import random


def setup(board: str, player_hands: List[str]):
    cards_state_dict = build_cards_state_dict(board, player_hands)
    env: AugmentObservationWrapper = init_wrapped_env(env_wrapper_cls=AugmentObservationWrapper,
                                                      stack_sizes=[100, 110, 120, 130, 140, 150])
    state_dict = {'deck_state_dict': cards_state_dict}
    return env, state_dict


hand_size = 2


def mc(hero_cards_1d, board_cards_1d, n_opponents, n_iter=1000000):
    """
    Returns estimated Effective Hand Strength after running n_iter Monte Carlo rollouts.
    :param hero_cards_1d: n * 4-byte representations of cards where n is the number of cards
    :param board_cards_1d: 5 * 4-byte representations of cards where 5 board cards may be zero-bytes
    :param n_iter: Number of rollouts to run before returning the estimated EHS. Default is 1 Million.
    :param n_opponents: Number of opponents simulated in the MC rollouts.
    :return: The Effective Hand Strength Pr(win), i.e. Pr(win) = HS x (1 - NPot) + (1 - HS) x PPot
    where HS is computed as in [LINK HAND STRENGTH]
    """

    # https: // github.com / kennethshackleton / SKPokerEval / blob / develop / tests / FiveEval.h
    deck = []
    for i in range(52):
        if i not in hero_cards_1d and i not in board_cards_1d:
            deck.append(i)

    n_missing_board_cards = len(deck) - 45
    cards_to_sample = 2 * n_opponents + n_missing_board_cards
    for i in range(n_iter):
        drawn_cards_1d = random.sample(deck, cards_to_sample)
        if n_missing_board_cards == 0:
            board = board_cards_1d
        else:
            board = board_cards_1d[:-n_missing_board_cards] + drawn_cards_1d[-n_missing_board_cards:]
        hero_hand = hero_cards_1d + board
        hero_rank = rank(*hero_hand)
        for opp in range(n_opponents):
            opp_hand = [drawn_cards_1d[hand_size * opp], drawn_cards_1d[hand_size * opp + 1]] + board
            rank(*opp_hand)


if __name__ == "__main__":
    # mc([42, 0], [11, 22, 33, 44], 1000, 2)  # river yet to come)
    # 1, 41, 18, 19, 16, 20, 24 must return 5586
    # 51, 47, 43, 39, 35, 0, 1 must return a lower value than random num
    print(rank(1, 41, 18, 19, 16, 20, 24))
    for i in range(10):
        s0 = time.time()
        mc([1, 41], [18, 19, 16, 20, 24], 2, 100000)
        print(time.time() - s0)