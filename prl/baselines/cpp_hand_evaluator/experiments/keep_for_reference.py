"""WIP - UNDER HEAVY CONSTRUCTION"""
from typing import List
from numba import jit, njit

import numpy as np
import random
from prl.environment.Wrappers.prl_wrappers import AugmentObservationWrapper
from prl.environment.steinberger.PokerRL.game._.cpp_wrappers.CppHandeval import CppHandeval
from prl.baselines.supervised_learning.data_acquisition.environment_utils import build_cards_state_dict, \
    init_wrapped_env, make_player_cards, make_board_cards
from hand_evaluator import rank
# from prl.environment.steinberger.PokerRL.game._.cpp_wrappers.CppLUT import CppLibHoldemLuts
# cpp_poker = CppHandeval()
# luts = CppLibHoldemLuts()


def setup(board: str, player_hands: List[str]):
    cards_state_dict = build_cards_state_dict(board, player_hands)
    env: AugmentObservationWrapper = init_wrapped_env(env_wrapper_cls=AugmentObservationWrapper,
                                                      stack_sizes=[100, 110, 120, 130, 140, 150])
    state_dict = {'deck_state_dict': cards_state_dict}
    return env, state_dict


# board = '['' '' '' '' '']'
# board = "[Jd Qd Kd  ]"
board = "[2h 3h 4h  ]"
# player_hands = ['3h 3c', 'Tc 9s', 'Jd Js', 'Kc Ks', 'Ac Ad', '2h 2c']
player_hands = ['5c 7s', 'Ad Td', '7c 2s']
wrapped_env, state_dict = setup(board, player_hands)
lh = wrapped_env.env.get_lut_holder()
feature_names = list(wrapped_env.obs_idx_dict.keys()) + ["button_index"]

player_cards = make_player_cards(player_hands)
board_cards = make_board_cards(board)
b1 = lh.get_1d_card(board_cards[0])
b2 = lh.get_1d_card(board_cards[1])
b3 = lh.get_1d_card(board_cards[2])
b4 = lh.get_1d_card(board_cards[3])
print(b1)
print(b2)
print(b3)
print(b4)
c1 = lh.get_1d_card(player_cards[0][0])
c2 = lh.get_1d_card(player_cards[0][1])
c3 = lh.get_1d_card(player_cards[2][0])
c4 = lh.get_1d_card(player_cards[2][1])
print(c1)
print(c2)
print(c3)
print(c4)

# toprank = cpp_poker.get_hand_rank_52_holdem(hand_2d=np.array(player_cards[0]), board_2d=np.array(board_cards))
# botrank = cpp_poker.get_hand_rank_52_holdem(hand_2d=np.array(player_cards[2]), board_2d=np.array(board_cards))


# draw three hands wo replacement
# draw flop
# draw turn
# draw river
# rh = rank(h0, board)
# r1 = rank(h1, board)
# r2 = rank(h2, board)
# if rh > r1 and rh > r2:
# won += 1
# else
# lost += 1

# hand: 4,22
# vil_0: 44, 50
# vil_1: 18, 28

# flop comes 0,12, 29

# rank([4,22], [0,12,29])
@njit
def mc(hero_cards_1d, board_cards_1d, n_iter, n_villains):
    """
    Returns estimated Effective Hand Strength after running n_iter Monte Carlo rollouts.
    :param hero_cards_1d: n * 4-byte representations of cards where n is the number of cards
    :param board_cards_1d: 5 * 4-byte representations of cards where 5 board cards may be zero-bytes
    :param n_iter: Number of rollouts to run before returning the estimated EHS.
    :param n_villains: Number of opponents simulated in the MC rollouts.
    :return: The Effective Hand Strength Pr(win), i.e. Pr(win) = HS x (1 - NPot) + (1 - HS) x PPot
    where HS is computed as in [LINK HAND STRENGTH]
    """

    # https: // github.com / kennethshackleton / SKPokerEval / blob / develop / tests / FiveEval.h
    deck = []
    for i in range(51):
        if i not in hero_cards_1d and i not in board_cards_1d:
            deck.append(i)
    deck = np.array(deck)
    won = 0
    lost = 0
    to_deal = 5 - len(board_cards_1d)
    for i in range(n_iter):
        cards = np.random.choice(deck, [2 for _ in range(n_villains)] + [to_deal], replace=False)
        board = cards[-to_deal:]
        hero_rank = rank(hero_cards_1d, board)
        for i in range(0, n_villains+1, 2):
            # if rank([cards[i]+cards[i+1], board) > hero_rank:
            #     {lost+=1} else {won += 1}
            pass

        #     # todo call RANK FROM CLIB using pybind
        #     # todo might have to use CFFI as this is definitely supported as shown in the DOCs
        #     # todo look at how to speed up vectorizer by using jit


if __name__ == "__main__":
    mc([42, 0], [11, 22, 33, 44], 1000, 2)  # river yet to come)
