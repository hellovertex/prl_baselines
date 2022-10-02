from typing import List, Type, Tuple

import numpy as np
from prl.environment.Wrappers.prl_wrappers import Wrapper
from prl.environment.steinberger.PokerRL import NoLimitHoldem, Poker

from prl.baselines.supervised_learning.data_acquisition.core.encoder import PlayerInfo
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode

DICT_RANK = {'': -127,
             '2': 0,
             '3': 1,
             '4': 2,
             '5': 3,
             '6': 4,
             '7': 5,
             '8': 6,
             '9': 7,
             'T': 8,
             'J': 9,
             'Q': 10,
             'K': 11,
             'A': 12}

DICT_SUITE = {'': -127,
              'h': 0,
              'd': 1,
              's': 2,
              'c': 3}


def init_wrapped_env(env_wrapper_cls: Type[Wrapper],
                     stack_sizes: List[float],
                     multiply_by=100) -> Wrapper:  # Tuple[Wrapper, List[int]]:
    """
    Wrappes a NoLimitHoldEm instance with a custom wrapper class.
    Returns the initialized (not reset yet!) environment, together with
    a list of integer starting stacks.
    i) Use multiplier of 100 to convert two-decimal floats to integer
    ii) Assumes Btn is at stack_sizes index 0.
    :param env_wrapper_cls: Custom implementation of NoLimitHoldem-Wrapper
    :param stack_sizes: List of starting stack sizes. Starts with Button.

    # keep this to make sure nobody forgets removing the decimals
    :param multiply_by: Default is 100 to convert two-decimal floats to int.
    :return: Returns the initialized (not reset yet!) environment, together with
    a list of starting stacks. Starting stacks begin with the BTN.
    """
    # get starting stacks, starting with button at index 0
    starting_stack_sizes_list = [int(float(stack) * multiply_by) for stack in stack_sizes]

    # make args for env
    args = NoLimitHoldem.ARGS_CLS(n_seats=len(stack_sizes),
                                  starting_stack_sizes_list=starting_stack_sizes_list)
    # return wrapped env instance
    env = NoLimitHoldem(is_evaluating=True,
                        env_args=args,
                        lut_holder=NoLimitHoldem.get_lut_holder())
    wrapped_env = env_wrapper_cls(env)
    return wrapped_env  # todo urgent replace:, starting_stack_sizes_list


def card_tokens(cards: str) -> List[str]:
    """
    Examples:
    In: '[6h Ts Td 9c Jc]'
    Out: ['6h', 'Ts', 'Td', '9c', 'Jc'].

    In: '6h'
    Out: ['6h']

    In: '[6h Ts Td]'
    Out: ['6h', 'Ts', 'Td']

    In: '3h 3c'
    Out: ['3h', '3c']
    """
    # '[6h Ts Td 9c Jc]'
    rm_brackets = cards.replace('[', '').replace(']', '')
    # '6h Ts Td 9c Jc'
    card_list = rm_brackets.split(' ')
    # ['6h', 'Ts', 'Td', '9c', 'Jc']
    return card_list


def card(token: str) -> List[int]:
    rank = DICT_RANK[token[0]]
    suite = DICT_SUITE[token[1]]
    return [rank, suite]


def make_board_cards(board: str) -> List[List[int]]:
    """
    Return representation of board_cards that is understood by Steinberger NoLimitHoldem environment.
    :param board: String representation of (full!) board, e.g. '[6h Ts Td 9c Jc]'
    :return: List of cards, where each card is a list of two integers, one for rank and one for suit.
    """
    board_card_tokens = card_tokens(board)
    assert len(board_card_tokens) == 5
    return [card(token) for token in board_card_tokens]


def build_cards_state_dict(board: str, player_hands: List[str]):
    """
    Creates dictionary with remaining deck, initial board and player hands that is used to reset
    the NoLimitHoldem environment, see prl.environment.steinberger.PokerRL.game.games.
    :param board:  String representation of (full!) board, e.g. '[6h Ts Td 9c Jc]'
    :param player_hands: List of player hands represented as single string,
    e.g. ['3h 3c',  'Tc 9s', 'Jd Js', 'Kc Ks', 'Ac Ad']
    :return: Dictionary with remaining deck, initial board and player hands that is used to reset
    the NoLimitHoldem environment, see prl.environment.steinberger.PokerRL.game.games.
    """
    board_cards = make_board_cards(board)
    # --- set deck ---
    # cards are drawn without ghost cards, so we simply replace the first 5 cards of the deck
    # with the board cards that we have parsed
    deck = np.empty(shape=(13 * 4, 2), dtype=np.int8)
    deck[:len(board_cards)] = board_cards
    # convert ['3h 3c',  'Tc 9s', 'Jd Js', 'Kc Ks', 'Ac Ad'] to List[List[int]]
    hands = [card_tokens(hand) for hand in player_hands]
    player_hands = []
    for hand in hands:
        player_hands.append([card(hand[0]), card(hand[1])])
    initial_board = np.full((5, 2), Poker.CARD_NOT_DEALT_TOKEN_1D, dtype=np.int8)
    return {'deck': {'deck_remaining': deck},  # np.ndarray(shape=(52-n_cards*num_players, 2))
            'board': initial_board,  # np.ndarray(shape=(n_cards, 2))
            'hand': player_hands}  # List[[rank, suit]] with length equal to number of players
