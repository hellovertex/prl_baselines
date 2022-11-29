from typing import List, Type, TypeVar

import numpy as np
from prl.environment.Wrappers.base import EnvWrapperBase
from prl.environment.steinberger.PokerRL import NoLimitHoldem, Poker

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

    In: ''
    Out: ['']
    """
    # '[6h Ts Td 9c Jc]'
    rm_brackets = cards.replace('[', '').replace(']', '')
    # '6h Ts Td 9c Jc'
    card_list = rm_brackets.split(' ')
    # ['6h', 'Ts', 'Td', '9c', 'Jc']
    return card_list


def card(token: str) -> List[int]:
    if token == '':
        return [DICT_RANK[''], DICT_SUITE['']]
    rank = DICT_RANK[token[0]]
    suite = DICT_SUITE[token[1]]
    return [rank, suite]


def make_board_cards(board: str) -> List[List[int]]:
    """
    Return representation of board_cards that is understood by Steinberger NoLimitHoldem environment.
    :param board: String representation of board, e.g. '[6h Ts Td 9c  ]' on Turn
    :return: List of cards, where each card is a list of two integers, one for rank and one for suit.
    """
    board_card_tokens = card_tokens(board)
    # assert len(board_card_tokens) == 5
    return [card(token) for token in board_card_tokens]


def make_player_cards(player_hands: List[str]):
    """
    Converts string representation of cards to integer representation as used by Steinbergers PokerEnv.
    :param player_hands: List of player hands represented as single string,
    e.g. ['3h 3c',  'Tc 9s', 'Jd Js', 'Kc Ks', 'Ac Ad']
    :return: List[List[int]]
    e.g. [  [[2, 0], [2, 3]], [[11, 1], [10, 2]], [[11, 2]], ..], ... ] dtype=np.int8)
    """
    # convert ['3h 3c',  'Tc 9s', 'Jd Js', 'Kc Ks', 'Ac Ad'] to List[List[int]]
    hands = [card_tokens(hand) for hand in player_hands]
    player_hands = []
    for hand in hands:
        player_hands.append([card(hand[0]), card(hand[1])])
    return player_hands


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
    initial_board = np.full((5, 2), Poker.CARD_NOT_DEALT_TOKEN_1D, dtype=np.int8)
    return {'deck': {'deck_remaining': deck},  # np.ndarray(shape=(52-n_cards*num_players, 2))
            'board': initial_board,  # np.ndarray(shape=(n_cards, 2))
            'hand': make_player_cards(player_hands)}  # List[[rank, suit]] with length equal to number of players
