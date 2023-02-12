from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Union

from prl.environment.Wrappers.base import ActionSpace

from prl.baselines.supervised_learning.data_acquisition.core.encoder import Positions6Max


class UnparsableFileException(ValueError):
    """This is raised when the parser can not finish parsing a .txt file."""


class EmptyPokerEpisode:
    """Placeholder for an empty PokerEpisode"""
    pass


@dataclass
class Player:
    name: str
    seat_num_one_indexed: int
    stack: int
    position: Optional[Positions6Max] = None
    cards: Optional[str] = None
    is_showdown_player: Optional[bool] = None
    money_won_this_round: Optional[int] = None


@dataclass
class Action:
    who: str
    what: Union[Tuple, ActionSpace]
    how_much: int
    stage: Optional[str] = None  # 'preflop' 'flop' 'turn' 'river'
    info: Optional[Dict] = None


@dataclass
class PokerEpisodeV2:
    hand_id: int
    currency_symbol: str
    players: Dict[str, Player]
    blinds: Dict[str, int]  # 'sb'->25 # Dict[str, Dict[str, int]]  # sb/bb -> player -> amount
    board: Optional[str]
    actions: Dict[str, List[Action]]
    has_showdown: Optional[bool]
    showdown_players: Optional[List[Player]]
    winners: Optional[List[Player]]
    btn_seat_num_one_indexed: Optional[int]
