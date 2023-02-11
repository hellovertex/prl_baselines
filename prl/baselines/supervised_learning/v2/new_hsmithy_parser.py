"""the old one uses python regexes which is unbearably slow for 250k files, let alone more
i tried making it better with multiprocessing but its still i/o bound so I have to improve on the
string processing side

I want a parser that can handle so much string data.
I want a parser that can handle incomplete episodes. I.e. if no showdown happened.
I need a data pipeline that is not unnecessarily complex
"""
import glob
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Union

from prl.environment.Wrappers.base import ActionSpace

from prl.baselines.supervised_learning.data_acquisition.core.encoder import Positions6Max


# all the following functionality should be possible with only minimal parameterization (input_dir, output_dir, ...)
# 1. parse .txt files given list of players (only games containing players, or all if list is None)
# 2.

class UnparsableFileException(ValueError):
    """This is raised when the parser can not finish parsing a .txt file."""


class EmptyPokerEpisode:
    """Placeholder for an empty PokerEpisode"""
    pass


@dataclass
class Player:
    name: str
    seat_num: int
    stack: int
    position: Optional[Positions6Max] = None


@dataclass
class Action:
    who: str
    what: Union[Tuple, ActionSpace]
    how_much: Optional[int]


@dataclass
class PokerEpisode:
    players: List[Player]
    blinds: Dict[str, int]  # player_name -> amount
    actions: Dict[str, List[Action]]


class ParseHsmithyTextToPokerEpisode:
    def __init__(self,
                 preflop_sep="*** HOLE CARDS ***",
                 flop_sep="*** FLOP ***",
                 turn_sep="*** TURN ***",
                 river_sep="*** RIVER ***",
                 showdown_sep="*** SHOW DOWN ***",
                 summary_sep="*** SUMMARY ***"):
        self.preflop_sep = preflop_sep
        self.flop_sep = flop_sep
        self.turn_sep = turn_sep
        self.river_sep = river_sep
        self.summary_sep = summary_sep
        self.showdown_sep = showdown_sep

    def strip_next_round(self, strip_round, episode_str):
        return episode_str.split(strip_round)[0]

    def split_at_round(self, round, episode_str):
        try:
            return episode_str.split(round)[1]
        except IndexError:
            # index 1 cannot be accessed -> there is no `round`
            return ""

    def rounds(self, current_episode: str) -> Dict[str, str]:
        hole_cards = self.split_at_round(self.preflop_sep, current_episode)
        flop = self.split_at_round(self.flop_sep, current_episode)
        turn = self.split_at_round(self.turn_sep, current_episode)
        river = self.split_at_round(self.river_sep, current_episode)

        # strip each round from the other rounds, so we isolate them for stat updates
        # flop, turn, river may be empty string, so we have to find out where to
        # strip each round
        # 1. strip hole cards
        if flop:
            next_round = self.flop_sep
            hole_cards = self.strip_next_round(self.flop_sep, hole_cards)
        else:
            hole_cards = self.strip_next_round(self.summary_sep, hole_cards)
        # 2. strip flop cards
        if turn:
            # split at flop and strip from turn onwards
            flop = self.strip_next_round(self.turn_sep, flop)
        else:
            flop = self.strip_next_round(self.summary_sep, flop)
        # 3. strip turn cards
        if river:
            # split at turn and strip from river onwards
            turn = self.strip_next_round(self.river_sep, turn)
        else:
            turn = self.strip_next_round(self.summary_sep, turn)
        # 4. strip river cards
        river = self.strip_next_round(self.showdown_sep, river)
        summary \
            = self.split_at_round(self.summary_sep, current_episode)

        # Assertions
        # PREFLOP
        assert not self.flop_sep in hole_cards
        assert not self.turn_sep in hole_cards
        assert not self.river_sep in hole_cards
        assert not self.summary_sep in hole_cards
        # FLOP
        assert not self.preflop_sep in flop
        assert not self.turn_sep in flop
        assert not self.river_sep in flop
        assert not self.summary_sep in flop
        # TURN
        assert not self.preflop_sep in turn
        assert not self.flop_sep in turn
        assert not self.river_sep in turn
        assert not self.summary_sep in turn
        # RIVER
        assert not self.preflop_sep in river
        assert not self.flop_sep in river
        assert not self.turn_sep in river
        assert not self.summary_sep in river

        return {'preflop': hole_cards,
                'flop': flop,
                'turn': turn,
                'river': river,
                'summary': summary}

    def get_players_and_blinds(self, hand_str) -> Tuple[Dict[str, Player], Dict[str, Dict[str, int]]]:
        players = {}  # don't make this ordered, better to rely on names
        blinds = {}
        table = hand_str.split("*** HOLE CARDS ***")[0]
        lines = table.split('\n')
        for line in lines:
            if "Table \'" in line:
                continue
            if 'Seat' in line:
                seat, player = line.split(':')
                seat_num = seat[-1]
                pname, stack = player.split('(')
                pname: str = pname[1:-1]
                stack = stack[1:]
                stack = float(stack.split(' ')[0])
                player = Player(name=pname,
                                seat_num=seat_num,
                                stack=round(stack * 100))
                players[pname] = player
            elif 'posts small blind' in line:
                sb_name: str = line.split(":")[0]
                sb_amt = round(float(line.split("$")[1]) * 100)
                players[sb_name].position = Positions6Max.SB
                blinds['sb'] = {sb_name: sb_amt}
            elif 'posts big blind' in line:
                bb_name: str = line.split(":")[0]
                bb_amt = round(float(line.split("$")[1]) * 100)
                players[bb_name].position = Positions6Max.BB
                blinds['bb'] = {bb_name: bb_amt}
        num_players = len(players)
        return players, blinds

    def get_action(self, line):
        pname, action = line.split(':')
        if 'folds' in action:
            return Action(who=pname, what=ActionSpace.FOLD, how_much=-1)
        elif 'checks' in action:
            return Action(who=pname, what=ActionSpace.CHECK_CALL, how_much=-1)
        elif 'calls' in action:
            amt = round(float(action.split('$')[1]) * 100)
            return Action(who=pname, what=ActionSpace.CHECK_CALL, how_much=amt)
        elif 'bets' in action:
            amt = round(float(action.split('$')[1]) * 100)
            return Action(who=pname, what=ActionSpace.RAISE_MIN_OR_3BB, how_much=amt)
        elif 'raises' in action:
            amt = round(float(action.split('to ')[1].split('$')[1]) * 100)
            return Action(who=pname, what=ActionSpace.RAISE_MIN_OR_3BB, how_much=amt)
        else:
            raise ValueError(f"Unknown action in {line}.")

    def _get_actions(self, lines):
        lines = lines.split('\n')
        actions = []
        for line in lines:
            if not line:
                continue
            if not ':' in line:
                continue
            action = self.get_action(line)
            actions.append(action)
        return actions

    def get_actions(self, info):
        actions_preflop = self._get_actions(info['preflop'])
        actions_flop = self._get_actions(info['flop'])
        actions_turn = self._get_actions(info['turn'])
        actions_river = self._get_actions(info['river'])
        as_sequence = []

        for actions in [actions_preflop, actions_flop, actions_turn, actions_river]:
            for action in actions:
                as_sequence.append(action)
        return {'actions_preflop': actions_preflop,
                'actions_flop': actions_flop,
                'actions_turn': actions_turn,
                'actions_river': actions_river,
                'as_sequence': as_sequence}

    def parse_hand(self, hand_str):
        players, blinds = self.get_players_and_blinds(hand_str)
        info = self.rounds(hand_str)
        actions = self.get_actions(info)
        board_cards = None

        return PokerEpisode()

    def parse_file(self, f: str,
                   out: str,
                   filtered_players: Optional[Tuple[str]],
                   only_showdowns: bool) -> List[PokerEpisode]:
        """
        :param f: Absolute path to .txt file containing human-readable hhsmithy-export.
        :param out: Absolute path to .txt file containing
        :param filtered_players: If provided, the result will only contain games where
        a player in `filtered_players` participated.
        :param only_showdowns: If True, will only generate episodes that finished.
        As a consequence, if set to false, this returns PokerEpisodes where no player hands are visible.
        """
        # if filtered_players is None:
        # instead of using regex (which is slow) we must do it manually
        episodes = []
        with open(f, 'r', encoding='utf-8') as f:  # pylint: disable=invalid-name,unspecified-encoding
            hand_database = f.read()
            hands_played = re.split(r'PokerStars Hand #', hand_database)[1:]
            for hand in hands_played:
                episodes.append(self.parse_hand(hand))
        return episodes


if __name__ == "__main__":
    unzipped_dir = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data_test"
    out_dir = "example.txt"
    filenames = glob.glob(unzipped_dir + "/**/*.txt", recursive=True)
    parser = ParseHsmithyTextToPokerEpisode()
    for filename in filenames:
        parser.parse_file(filename, out_dir, None, True)
