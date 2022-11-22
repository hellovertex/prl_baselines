import time

import numba
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
numba.vectorize()


class MonteCarlo_HandEvaluator:

    # def mc(self, id_caller_thread, deck, hero_cards_1d, board_cards_1d, n_opponents, n_iter):
    def mc(self, deck, hero_cards_1d, board_cards_1d, n_opponents, n_iter):
        n_missing_board_cards = len(deck) - 45
        cards_to_sample = 2 * n_opponents + n_missing_board_cards

        won = 0
        lost = 0
        tied = 0

        for i in range(n_iter):
            # draw board, if not complete already
            drawn_cards_1d = random.sample(deck, cards_to_sample)
            if n_missing_board_cards == 0:
                board = board_cards_1d
            else:
                board = board_cards_1d[:-n_missing_board_cards] + drawn_cards_1d[-n_missing_board_cards:]

            # rank hero hand
            hero_hand = hero_cards_1d + board
            hero_rank = rank(*hero_hand)

            # compare hero hand to opponent hands
            player_still_winning = True
            ties = 0
            for opp in range(n_opponents):
                opp_hand = [drawn_cards_1d[hand_size * opp], drawn_cards_1d[hand_size * opp + 1]] + board
                opp_rank = rank(*opp_hand)
                if opp_rank > hero_rank:
                    player_still_winning = False
                    break
                elif opp_rank == hero_rank:
                    ties += 1

            # update won/lost/tied stats
            if not player_still_winning:
                lost += 1
            elif player_still_winning and ties < n_opponents:
                won += 1
            elif player_still_winning and ties == n_opponents:
                tied += 1
            else:
                raise ValueError("Hero can tie against at most n_opponents, not more. Aborting MC Simulation...")
        return {'won': won, 'lost': lost, 'tied': tied}

    def run_mc(self, hero_cards_1d, board_cards_1d, n_opponents, n_iter=1000000):
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

        return self.mc(deck, hero_cards_1d, board_cards_1d, n_opponents, n_iter)


if __name__ == "__main__":
    # mc([42, 0], [11, 22, 33, 44], 1000, 2)  # river yet to come)
    # 1, 41, 18, 19, 16, 20, 24 must return 5586
    # 51, 47, 43, 39, 35, 0, 1 must return a lower value than random num
    print(rank(1, 41, 18, 19, 16, 20, 24))
    mc = MonteCarlo_HandEvaluator()
    for i in range(10):
        s0 = time.time()
        result_dict = mc.run_mc([1, 41], [18, 19, 16, 20, 24], 2, 5000)
        # mc.run_mc([1, 41], [-1, -1, -1, -1, -1], 2, 100000)
        print(time.time() - s0)
