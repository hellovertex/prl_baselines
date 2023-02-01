from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as cols
import numpy as np
from prl.environment.steinberger.PokerRL import Poker
from prl.environment.steinberger.PokerRL.game._.rl_env.game_rules import HoldemRules


def cards2str(cards_2d, seperator=", "):
    """
    Args:
        cards_2d:           2D representation of any amount of cards
        seperator (str):    token to put between cards when printing

    Returns:
        str
    """
    hand_as_str = ""
    for c in cards_2d:
        if not np.array_equal(c, Poker.CARD_NOT_DEALT_TOKEN_2D):
            hand_as_str += HoldemRules.RANK_DICT[c[0]]
            hand_as_str += HoldemRules.SUIT_DICT[c[1]]
            hand_as_str += seperator
    return hand_as_str


def print_player_cards(obs):
    """Returns human readable cards of players."""
    n_ranks = 13
    n_suits = 4
    ci = n_ranks + n_suits
    cards = {0: "", 1: "", 2: "", 3: "", 4: "", 5: ""}
    # first player
    p0_bits = obs[cols.First_player_card_0_rank_0:cols.Second_player_card_0_rank_0]
    c0 = p0_bits[:ci]
    c1 = p0_bits[ci:]
    r0, s0 = np.where(c0 == 1)[0]
    # second player
    p1_bits = obs[cols.Second_player_card_0_rank_0:cols.Third_player_card_0_rank_0]
    # third player
    p2_bits = obs[cols.Third_player_card_0_rank_0:cols.Fourth_player_card_0_rank_0]
    # fourth player
    p3_bits = obs[cols.Fourth_player_card_0_rank_0:cols.Fifth_player_card_0_rank_0]
    # fifth player
    p4_bits = obs[cols.Fifth_player_card_0_rank_0:cols.Sixth_player_card_0_rank_0]
    # sixth player
    p5_bits = obs[cols.Sixth_player_card_0_rank_0:cols.Sixth_player_card_1_suit_3 + 1]
    return cards
