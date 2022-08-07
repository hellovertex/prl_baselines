import ast
import glob
import json
import time
from dataclasses import dataclass
from functools import partial
from typing import Iterable, Dict, List

import pandas as pd

from prl.baselines.supervised_learning.config import DATA_DIR
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode
from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser


# PlayerStat = namedtuple("PlayerStat", ["n_hands_played", "n_showdowns", "n_won", "total_earnings"])


@dataclass
class PlayerStat:
    n_hands_played: int
    n_showdowns: int
    n_won: int
    total_earnings: float


player_dict = {}  # player: games_played, games_won
player_stats: Dict[str, PlayerStat] = {}  # player: {n_hands_played, n_showdowns, n_won, total_earnings}
BLIND_SIZES = "0.25-0.50"


def write(player_dict):
    """Writes player_dict to result.txt file"""
    with open('result.txt', 'w') as f:
        f.write(json.dumps(player_dict))


def update_player_dict(episodes: Iterable[PokerEpisode]):
    # for each played hand
    for episode in episodes:
        # for each showdown player
        for player in episode.showdown_hands:
            if not player.name in player_dict:
                player_dict[player.name] = 0
            else:
                player_dict[player.name] += 1


def run(cbs=None) -> Dict[str, int]:
    """Reads unzipped folder and returns dictionary with player names and number of games played"""
    filenames = glob.glob(str(DATA_DIR) + f'/01_raw/{BLIND_SIZES}/unzipped/' '/**/*.txt', recursive=True)
    parser = HSmithyParser()
    # parse, encode, vectorize and write the training data from .txt to disk
    for i, filename in enumerate(filenames):
        try:
            parsed_hands = parser.parse_file(filename)
            if cbs:
                [cb(parsed_hands) for cb in cbs]  # update_player_dict(parsed_hands)
            print(f'After {i}-th file we have {len(player_dict.keys())} different players')
        except UnicodeDecodeError:
            print('---------------------------------------')
            print(
                f'Skipping {filename} because it has invalid continuation byte...')
            print('---------------------------------------')
    return player_dict


def reduce() -> pd.DataFrame:
    import ast
    with open("result.txt", "r") as data:
        df = pd.DataFrame.from_dict(ast.literal_eval(data.read()), orient='index')
    df = df.sort_values(0, ascending=False)
    df = df[df > 100].dropna()
    return df


def update_stats_dict(top_players: List[str], episodes: Iterable[PokerEpisode]):
    # player: {n_hands_played, n_showdowns, n_won, total_earnings}

    # for each played hand
    for episode in episodes:
        for players in episode.player_stacks:
            for p in players:
                if not p.name in player_stats:
                    # {'n_hands_played': 1, 'n_showdowns': 0, 'n_won': 0, 'total_earnings': 0.0}
                    player_stats[p.name] = PlayerStat(1, 0, 0, 0.0)
                else:
                    player_stats[p.name].n_hands_played += 1


def get_stats():
    with open("result.txt", "r") as data:
        player_dict = ast.literal_eval(data.read())
        player_names = list(player_dict.keys())
        print(player_names)
    cb = partial(update_stats_dict, top_players=player_names)
    run(cbs=[cb])


if __name__ == "__main__":
    # *** First run ***
    # player_dict = run(cbs=[update_player_dict])  # 1.
    # write(player_dict)
    # df = reduce()  # 2.
    # write(df.to_dict()[0])
    time.sleep(0.5)
    # *** Second run ***
    get_stats()
