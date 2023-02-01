from typing import List, Dict

import numpy as np
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as cols, AugmentObservationWrapper
from prl.environment.Wrappers.base import ActionSpace
from prl.environment.steinberger.PokerRL import Poker, NoLimitHoldem
from prl.environment.steinberger.PokerRL.game._.rl_env.game_rules import HoldemRules

from prl.baselines.supervised_learning.data_acquisition.environment_utils import make_board_cards, card_tokens, card

N_RANKS = 13
N_SUITS = 4
CI = N_RANKS + N_SUITS


def cards2str(cards_2d, seperator=", "):
    """
    Args:
        cards_2d:           2D representation of any amount of cards
        seperator (str):    token to put between cards when printing

    Returns:
        str
    """
    hand_as_str = "["
    invisible = True
    for c in cards_2d:
        if not np.array_equal(c, Poker.CARD_NOT_DEALT_TOKEN_2D):
            invisible = False
            hand_as_str += HoldemRules.RANK_DICT[c[0]]
            hand_as_str += HoldemRules.SUIT_DICT[c[1]]
            hand_as_str += seperator
    if invisible:
        return "[? ?]"
    return hand_as_str[:-2] + "]"


def print_player_cards(obs):
    cards = get_player_cards(obs)
    print(f'Player 0 cards: {cards[0]}')
    print(f'Player 1 cards: {cards[1]}')
    print(f'Player 2 cards: {cards[2]}')
    print(f'Player 3 cards: {cards[3]}')
    print(f'Player 4 cards: {cards[4]}')
    print(f'Player 5 cards: {cards[5]}')


def get_player_cards(obs):
    """Returns human readable cards of players."""

    def card_bits_to_2d(c0, c1):
        CARD_NOT_VISIBLE_2D = [-127, -127]
        try:
            r0, s0 = np.where(c0 == 1)[0]
            s0 -= N_RANKS

            r1, s1 = np.where(c1 == 1)[0]
            s1 -= N_RANKS
            cards = [[r0, s0], [r1, s1]]
        except ValueError:
            cards = [CARD_NOT_VISIBLE_2D, CARD_NOT_VISIBLE_2D]
        return cards

    # first player
    p0_bits = obs[cols.First_player_card_0_rank_0:cols.Second_player_card_0_rank_0]
    c0 = p0_bits[:CI]
    c1 = p0_bits[CI:]
    cards_p0 = card_bits_to_2d(c0, c1)
    cp0 = cards2str(cards_p0)

    # second player
    p1_bits = obs[cols.Second_player_card_0_rank_0:cols.Third_player_card_0_rank_0]
    c0 = p1_bits[:CI]
    c1 = p1_bits[CI:]
    cards_p1 = card_bits_to_2d(c0, c1)
    cp1 = cards2str(cards_p1)

    # third player
    p2_bits = obs[cols.Third_player_card_0_rank_0:cols.Fourth_player_card_0_rank_0]
    c0 = p2_bits[:CI]
    c1 = p2_bits[CI:]
    cards_p2 = card_bits_to_2d(c0, c1)
    cp2 = cards2str(cards_p2)

    # fourth player
    p3_bits = obs[cols.Fourth_player_card_0_rank_0:cols.Fifth_player_card_0_rank_0]
    c0 = p3_bits[:CI]
    c1 = p3_bits[CI:]
    cards_p3 = card_bits_to_2d(c0, c1)
    cp3 = cards2str(cards_p3)

    # fifth player
    p4_bits = obs[cols.Fifth_player_card_0_rank_0:cols.Sixth_player_card_0_rank_0]
    c0 = p4_bits[:CI]
    c1 = p4_bits[CI:]
    cards_p4 = card_bits_to_2d(c0, c1)
    cp4 = cards2str(cards_p4)

    # sixth player
    p5_bits = obs[cols.Sixth_player_card_0_rank_0:cols.Sixth_player_card_1_suit_3 + 1]
    c0 = p5_bits[:CI]
    c1 = p5_bits[CI:]
    cards_p5 = card_bits_to_2d(c0, c1)
    cp5 = cards2str(cards_p5)

    return {0: cp0, 1: cp1, 2: cp2, 3: cp3, 4: cp4, 5: cp5}


def get_round(obs):
    if obs[cols.Round_preflop]:
        return "Preflop"
    elif obs[cols.Round_flop]:
        return "Flop"
    elif obs[cols.Round_turn]:
        return "Turn"
    elif obs[cols.Round_river]:
        return "River"
    else:
        raise ValueError("Faulty Observation, "
                         "no game-round preflop, flop, etc. specified")


def pretty_print(player_id, obs, action):
    cards = get_player_cards(obs)
    round = get_round(obs)
    acted = ActionSpace(action).name
    result = f'Player {player_id} performed ' \
             f'action {"CHECK_FOLD" if acted == "FOLD" else acted} ' \
             f'with cards {cards[0]} ' \
             f'in {round}'
    print(result)
    return result


def get_reset_config(player_hands: List[str],
                     board=None) -> Dict:
    """
    In: ['[6s 6d]', '[9s 9d]', '[Jd Js]']
    Out: state_dict used to initialize PokerRL environment with the specified player
    and board cards via env.reset(config)
    """
    board = board if board else '[6h Ts Td 9c Jc]'
    board_cards = make_board_cards(board)
    deck = np.empty(shape=(13 * 4, 2), dtype=np.int8)
    deck[:len(board_cards)] = board_cards
    player_hands = [card_tokens(cards) for cards in player_hands]
    hands = []
    for cards in player_hands:
        hand = [card(token) for token in cards]
        hands.append(hand)
    initial_board = np.full((5, 2), Poker.CARD_NOT_DEALT_TOKEN_1D, dtype=np.int8)
    return {'deck_state_dict': {'deck': {'deck_remaining': deck},  # np.ndarray(shape=(52-n_cards*num_players, 2))
            'board': initial_board,  # np.ndarray(shape=(n_cards, 2))
            'hand': hands}}


def get_default_env(num_players, starting_stack_size=None):
    starting_stack_size = 20000 if not starting_stack_size else starting_stack_size
    stack_sizes = [starting_stack_size for _ in range(num_players)]
    args = NoLimitHoldem.ARGS_CLS(n_seats=len(stack_sizes),
                                  starting_stack_sizes_list=stack_sizes,
                                  use_simplified_headsup_obs=False)
    # return wrapped env instance
    env = NoLimitHoldem(is_evaluating=True,
                        env_args=args,
                        lut_holder=NoLimitHoldem.get_lut_holder())
    return AugmentObservationWrapper(env)


def print_player_actions(obs):
    # todo consider adding card info* to past actions to make it easier to connect
    #  only for player 0 as the other cards are invisible
    pass


def get_player_stacks(obs, normalization_sum=1):
    stacks = {0: obs[cols.Stack_p0] * normalization_sum,
              1: obs[cols.Stack_p1] * normalization_sum,
              2: obs[cols.Stack_p2] * normalization_sum,
              3: obs[cols.Stack_p3] * normalization_sum,
              4: obs[cols.Stack_p4] * normalization_sum,
              5: obs[cols.Stack_p5] * normalization_sum}
    return stacks


def print_player_stacks(obs, normalization_sum=1):
    stacks = get_player_stacks(obs, normalization_sum)
    res = ""
    for k, v in stacks.items():
        res += f"Player {k} stack is: {v} chips.\n"
    print(res)
