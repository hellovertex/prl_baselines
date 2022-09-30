"""Module to parse .txt databases that contain Poker episodes crawled from Pokerstars."""
from typing import NamedTuple, Iterable, List, Dict, Optional
import enum


class PlayerWithCards(NamedTuple):
    """Player with cards as string.
    For example: PlayerWithCards('HHnguyen15', '[Ah Jd]')"""
    name: str
    cards: str


class PlayerStack(NamedTuple):
    """Player Stacks as parsed from the textfiles.
    For example: PlayerStack('Seat 1', 'jimjames32', '$82 ')
    """
    seat_display_name: str
    player_name: str
    stack: str


class Blind(NamedTuple):
    """Blind('HHnguyen15', 'small blind', '$1')"""
    player_name: str
    type: str  # 'small blind' | 'big blind'
    amount: str  # '$1', '$0.25', '€1', '€0.25'


class ActionType(enum.IntEnum):
    """Actions as used in PokerEnv.prl.environment.SteinbergerPokerEnvironment.PokerRL.game.Poker"""
    FOLD = 0
    CHECK_CALL = 1
    RAISE = 2


class Action(NamedTuple):
    """If the current bet is 30, and the agent wants to bet 60 chips more, the action should
    be (2, 90). For Example: Action(stage='preflop',
                                    player_name='jimjames32',
                                    action_type=<ActionType.RAISE: 2>, raise_amount='6')
    """
    stage: str
    player_name: str
    action_type: ActionType
    raise_amount: float = -1


class PlayerWinningsCollected(NamedTuple):
    player_name: str
    collected: str  # €78.20
    rake: Optional[str] = ""  # how much rake e.g. '€0.70' from a total pot of 78.90


class PokerEpisode(NamedTuple):
    """Internal Representation of played hand.
    Used to initialize and step the RL-environment."""
    date: str
    hand_id: int
    variant: str
    currency_symbol: str  # '$' or '€'  or '￡'
    num_players: int
    blinds: List[Blind]
    ante: str  # usually equal to '$0.00' or '€0.00' or '￡0.00'
    player_stacks: List[PlayerStack]
    btn_idx: int
    board_cards: str  # e.g. '[6h Ts Td 9c Jc]'
    """
    Dictionary with actions per stage:
            {'preflop': actions_preflop,
                'flop': actions_flop,
                'turn': actions_turn,
                'river': actions_river,
                'as_sequence': as_sequence}
    """
    actions_total: Dict[str, List[Action]]
    winners: List[PlayerWithCards]
    showdown_hands: List[PlayerWithCards]
    money_collected: List[PlayerWinningsCollected]


class Parser:  # pylint: disable=too-few-public-methods
    """ Abstract Parser Interface. All parsers should be derived from this base class
    and implement the method "parse_file"."""

    def parse_file(self, file_path) -> Iterable[PokerEpisode]:
        """Reads file that stores played poker hands and returns and iterator over the played hands.
        Args:
          file_path: path to the database file that contains hands crawled from a specific poker
          website.
        Returns: An Iterable of PokerEpisodes.

        """
        raise NotImplementedError
