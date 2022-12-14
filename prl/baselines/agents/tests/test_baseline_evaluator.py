# what is required, in order to safely say that the evaluation results are correct?
# agents pick action based on correct observation vector --> assume obs is correct FOR NOW, later we can
# assert obs == obs'
from typing import Dict, Any, List

import numpy as np
from prl.environment.steinberger.PokerRL import Poker

from prl.baselines.supervised_learning.data_acquisition.core.encoder import PlayerInfo, Positions6Max
from prl.baselines.supervised_learning.data_acquisition.environment_utils import card_tokens, card
from prl.baselines.supervised_learning.data_acquisition.rl_state_encoder import RLStateEncoder


def setup_1():
    """Heads up: Player 0 vs Player 1
    Player 0:
    """
    deck = {}
    actions = []
    return deck, actions


def make_state_dict(player_hands: List[str], board_cards: str) -> Dict[str, Any]:
    hands = []
    for hand in player_hands:
        hands.append([card(token) for token in card_tokens(hand)])
    board = [card(token) for token in card_tokens(board_cards)]
    deck = np.empty(shape=(13 * 4, 2), dtype=np.int8)
    initial_board = np.full((5, 2), Poker.CARD_NOT_DEALT_TOKEN_1D, dtype=np.int8)
    deck[:len(board)] = board
    return {'deck': {'deck_remaining': deck},  # np.ndarray(shape=(52-n_cards*num_players, 2))
            'board': initial_board,  # np.ndarray(shape=(n_cards, 2))
            'hand': hands}


def test_episode_matches_environment_states_and_actions():
    # setup
    encoder = RLStateEncoder()  # to parse human-readable cards to 2d arrays
    num_players = 2
    starting_stack_size = 1000
    table = tuple([PlayerInfo(seat_number=i,
                              position_index=i,
                              position=Positions6Max(i).name,
                              player_name=f'Player_{i}',
                              stack_size=starting_stack_size) for i in range(num_players)])
    # 1. need deck_state_dict = hand cards + board
    hand_0 = '[Ah Kh]'
    hand_1 = '[7s 2c]'
    player_hands = [hand_0, hand_1]
    board = 'Qh Jh Th 9h 8h'
    state_dict = make_state_dict(player_hands, board)

    # 2. need action sequence that results in showdown (any will do, e.g. all in and call)
    # 3. assert expected outcome is really PokerEpisode
    pass
