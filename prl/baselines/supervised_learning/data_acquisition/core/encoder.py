"""Module to consume from Parser classes. Encoders will make observations from PokerEpisodes."""
import enum
from typing import NamedTuple
from .parser import PokerEpisode


class Positions6Max(enum.IntEnum):
    """Positions as in the literature, for a table with at most 6 Players.
BTN for Button, SB for Small Blind, etc...
"""
    BTN = 0
    SB = 1
    BB = 2
    UTG = 3  # UnderTheGun
    MP = 4  # Middle Position
    CO = 5  # CutOff


class Positions9Max(enum.IntEnum):
    """Positions as in the literature, for a table with at most 9 Players.
BTN for Button, SB for Small Blind, etc...
"""
    BTN = 0
    SB = 1
    BB = 2
    UTG = 3  # UnderTheGun
    UTG1 = 4
    UTG2 = 5
    MP = 6  # Middle Position
    MP1 = 7
    CO = 8  # CutOff


class PlayerInfo(NamedTuple):
    """Player information as parsed from the textfiles.
For example: PlayerInfo(seat_number=1, position_index=0, position='BTN',
player_name='jimjames32', stack_size=82.0)
"""
    seat_number: int
    position_index: int  # 0 for BTN, 1 for SB, 2 for BB, etc.
    position: str  # c.f. Positions6Max or Positions9Max
    player_name: str
    stack_size: float


class Encoder:  # pylint: disable=too-few-public-methods
    """ Abstract Encoder Interface. All encoders should be derived from this base class
    and implement the method "encode_episode"."""

    def encode_episode(self, episode: PokerEpisode):
        """Encodes one PokerEpisode to a vector that can be used for machine learning.
            Args:
                episode: A PokerEpisode returned by a Parser object.
            Returns:
                A vectorized observation together with an action that is the observations label
                for supervised training.
        """
        raise NotImplementedError
