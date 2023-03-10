import random

from prl.baselines.cpp_hand_evaluator.cpp.hand_evaluator import rank

LEN_DECK_WITHOUT_HERO_AND_BOARD_CARDS = 45  # 52 - 2 - 5
from typing import Tuple, List, Union

import numpy as np
import torch

from prl.baselines.cpp_hand_evaluator.rank import dict_str_to_sk

IDX_C0_0 = 167  # feature_names.index('0th_player_card_0_rank_0')
IDX_C0_1 = 184  # feature_names.index('0th_player_card_1_rank_0')
IDX_C1_0 = 184  # feature_names.index('0th_player_card_1_rank_0')
IDX_C1_1 = 201  # feature_names.index('1th_player_card_0_rank_0')
IDX_BOARD_START = 82  #
IDX_BOARD_END = 167  #
CARD_BITS_TO_STR = np.array(
    ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c'])
BOARD_BITS_TO_STR = np.array(
    ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
     'h', 'd', 's', 'c', '2', '3', '4', '5', '6', '7', '8', '9', 'T',
     'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c', '2', '3', '4', '5', '6',
     '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c', '2',
     '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h',
     'd', 's', 'c', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J',
     'Q', 'K', 'A', 'h', 'd', 's', 'c'])
RANK = 0
SUITE = 1


def card_bit_mask_to_int(c0: np.array, c1: np.array, board_mask: np.array) -> Tuple[
    List[int], List[int]]:
    c0_1d = dict_str_to_sk[CARD_BITS_TO_STR[c0][RANK] + CARD_BITS_TO_STR[c0][SUITE]]
    c1_1d = dict_str_to_sk[CARD_BITS_TO_STR[c1][RANK] + CARD_BITS_TO_STR[c1][SUITE]]
    board = BOARD_BITS_TO_STR[board_mask]
    # board = array(['A', 'c', '2', 'h', '8', 'd'], dtype='<U1')
    board_cards = []
    for i in range(0, int(sum(board_mask)) - 1,
                   2):  # sum is 6,8,10 for flop turn river resp.
        board_cards.append(dict_str_to_sk[board[i] + board[i + 1]])

    return [c0_1d, c1_1d], board_cards

# def card_bit_mask_to_int_torch(c0: np.array, c1: np.array, board_mask: np.array) ->
# Tuple[
#     List[int], List[int]]:
#     c0 = c0.cpu()
#     c1 = c1.cpu()
#     board_mask = board_mask.cpu()
#     c0_1d = dict_str_to_sk[CARD_BITS_TO_STR[c0][RANK] + CARD_BITS_TO_STR[c0][SUITE]]
#     c1_1d = dict_str_to_sk[CARD_BITS_TO_STR[c1][RANK] + CARD_BITS_TO_STR[c1][SUITE]]
#     board = BOARD_BITS_TO_STR[board_mask.bool()]
#     # board = array(['A', 'c', '2', 'h', '8', 'd'], dtype='<U1')
#     board_cards = []
#     for i in range(0, int(torch.sum(board_mask)) - 1,
#                    2):  # sum is 6,8,10 for flop turn river resp.
#         board_cards.append(dict_str_to_sk[board[i] + board[i + 1]])
#
#     return [c0_1d, c1_1d], board_cards


def look_at_cards(obs: np.array) -> Tuple[List[int], List[int]]:
    c0_bits = obs[IDX_C0_0:IDX_C0_1].astype(bool)
    c1_bits = obs[IDX_C1_0:IDX_C1_1].astype(bool)
    board_bits = obs[IDX_BOARD_START:IDX_BOARD_END].astype(bool)  # bit representation
    return card_bit_mask_to_int(c0_bits, c1_bits, board_bits)

# def look_at_cards_torch(obs: np.array) -> Tuple[List[int], List[int]]:
#     c0_bits = obs[IDX_C0_0:IDX_C0_1].bool()
#     c1_bits = obs[IDX_C1_0:IDX_C1_1].bool()
#     board_bits = obs[IDX_BOARD_START:IDX_BOARD_END]  # bit representation
#     return card_bit_mask_to_int(c0_bits, c1

class HandEvaluator_MonteCarlo:

    # def mc(self, id_caller_thread, deck, hero_cards_1d, board_cards_1d, n_opponents, n_iter):
    def mc(self, deck, hero_cards_1d, board_cards_1d, n_opponents, n_iter):
        n_missing_board_cards = len(deck) - LEN_DECK_WITHOUT_HERO_AND_BOARD_CARDS
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
                board = board_cards_1d + drawn_cards_1d[-n_missing_board_cards:]

            # rank hero hand
            hero_hand = hero_cards_1d + board
            hero_rank = rank(*hero_hand)

            # compare hero hand to opponent hands
            player_still_winning = True
            ties = 0
            for opp in range(n_opponents):
                opp_hand = [drawn_cards_1d[2 * opp], drawn_cards_1d[2 * opp + 1]] + board
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
                raise ValueError(
                    "Hero can tie against at most n_opponents, not more. Aborting MC Simulation...")
        return {'won': won, 'lost': lost, 'tied': tied}

    def _run_mc(self, hero_cards_1d, board_cards_1d, n_opponents, n_iter=1000000) -> dict:
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

    def run_mc(self,
               obs: Union[np.ndarray, list],
               n_opponents: int,
               n_iter=5000):
        hero_cards_1d, board_cards_1d = look_at_cards(obs)
        return self._run_mc(hero_cards_1d,
                            board_cards_1d,
                            n_opponents,
                            n_iter)
