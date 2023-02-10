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
from typing import Optional, List, Tuple

from prl.baselines.supervised_learning.data_acquisition.core.encoder import Positions6Max
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode


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


class ParseHsmithyTextToPokerEpisode:
    def __init__(self):
        pass

    def parse_hand(self, hand_str):
        players = {}
        blinds = {}
        table = hand_str.split("*** HOLE CARDS ***")[0]
        lines = table.split('\n')
        for line in lines:
            if 'Seat' in line:
                seat, player = line.split(':')
                seat_num = seat[-1]
                pname, stack = player.split('(')
                pname = pname[1:-1]
                stack = stack[1:]
                stack = float(stack.split(' ')[0])
                player = Player(name=pname, seat_num=seat_num, stack=round(stack * 100))
                players[pname] = player
            elif 'posts small blind' in line:
                sb_name = line.split(":")[0]
                sb_amt  = round(float(line.split("$"))*100)
                players[sb_name].position = Positions6Max.SB
                blinds['sb'] = {sb_name: sb_amt}
            elif 'posts big blind' in line:
                sb_name = line.split(":")[0]
                sb_amt  = round(float(line.split("$"))*100)
                players[sb_name].position = Positions6Max.BB
                blinds['bb'] = {sb_name: sb_amt}

    def parse_file(self, f: str, out: str, filtered_players: Optional[Tuple[str]]) -> List[PokerEpisode]:
        """
        :param f: Absolute path to .txt file containing human-readable hhsmithy-export.
        :param out: Absolute path to .txt file containing
        :param filtered_players: If provided, the result will only contain games where
        a player in `filtered_players` participated.
        """
        # if filtered_players is None:
        # instead of using regex (which is slow) we must do it manually
        episodes = []
        with open(f, 'r', encoding='utf-8') as f:  # pylint: disable=invalid-name,unspecified-encoding
            hand_database = f.read()
            hands_played = re.split(r'PokerStars Hand #', hand_database)[1:]
            for hand in hands_played:
                episodes.append(self.parse_hand(hand))


if __name__ == "__main__":
    unzipped_dir = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    out_dir = "example.txt"
    filenames = glob.glob(unzipped_dir + "/**/*.txt", recursive=True)
    parser = ParseHsmithyTextToPokerEpisode()
    for filename in filenames:
        parser.parse_file(filename, out_dir, None)
