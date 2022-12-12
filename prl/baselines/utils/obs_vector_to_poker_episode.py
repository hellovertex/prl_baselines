from datetime import datetime
from typing import Union, Optional, List

import numpy as np
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as COLS
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode, Blind, PlayerStack

# use vectorized representation of AugmentedObservationFeatureColumns to parse back to PokerEpisode object
MAX_PLAYERS = 6


def _get_num_players(obs) -> int:
    IS_ALLIN = [COLS.Is_allin_p0,
                COLS.Is_allin_p1,
                COLS.Is_allin_p2,
                COLS.Is_allin_p3,
                COLS.Is_allin_p4,
                COLS.Is_allin_p5]
    num_players = 0
    for pid, stack_index in enumerate([COLS.Stack_p0,
                                       COLS.Stack_p1,
                                       COLS.Stack_p2,
                                       COLS.Stack_p3,
                                       COLS.Stack_p4,
                                       COLS.Stack_p5]):
        # We check if player has a stack > 0 or is all in
        # We can ignore players previously (not in this episode) eliminated,
        # because an episode does not know about previous eliminations
        if obs[stack_index] > 0 or obs[IS_ALLIN[pid]]:
            num_players += 1
    return num_players


def _get_blinds(obs, num_players, normalization_sum) -> List[Blind]:
    """
    observing player sits relative to button. this offset is given by
    >>> obs[COLS.Btn_idx]
    the button position determines which players post the small and the big blind.
    For games with more than 2 pl. the small blind is posted by the player who acts after the button.
    When only two players remain, the button posts the small blind.
    """
    sb_name = "Player_1"
    bb_name = "Player_2"

    if num_players == 2:
        sb_name = "Player_0"
        bb_name = "Player_1"

    sb_amount = COLS.Small_blind * normalization_sum
    bb_amount = COLS.Big_blind * normalization_sum

    return [Blind(sb_name, 'small blind', sb_amount),
            Blind(bb_name, 'big_blind', bb_amount)]


def _get_player_stacks() -> List[PlayerStack]:
    player_stacks = []
    # Name the players, according to their offset to the button:
    # Button: Player_0
    # SB: Player_1
    # BB: Player_2
    # ...
    # Cut-off: Player_5
    for i in range(MAX_PLAYERS):
        seat_display_name = ''
        player_name = ''
        stack = ''
        player_stacks.append(PlayerStack(seat_display_name,
                                         player_name,
                                         stack))
    return player_stacks


def _get_board_cards() -> str:
    pass


def obs_vec_to_episode(obs: Union[list, np.ndarray],
                       normalization_sum: int) -> PokerEpisode:
    # PokerEpisode.date: dateime.now()
    # PokerEpisode.num_players -- positive stacks
    # PokerEpisode.blinds -- normalization_sum * sb, bb index

    date = str(datetime.now())
    hand_id = -1
    variant = "HUNL"
    currency_symbol = "$"
    num_players = _get_num_players(obs)
    blinds = _get_blinds(obs, num_players, normalization_sum)
    ante = obs[COLS.Ante] * normalization_sum
    player_stacks = _get_player_stacks()
    btn_idx = COLS.Btn_idx
    board_cards = _get_board_cards()


def run_tests():
    # todo create environment with 3 pl
    #  step using known acions
    #  compare with PokerEpisode
    pass


if __name__ == "__main__":
    run_tests()
