import ast
import glob
import json
import time
from pydantic import BaseModel
from functools import partial
from typing import Iterable, Dict, List

import pandas as pd

from prl.baselines.supervised_learning.config import DATA_DIR
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode, PlayerStack, ActionType
from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser


# PlayerStat = namedtuple("PlayerStat", ["n_hands_played", "n_showdowns", "n_won", "total_earnings"])



class PlayerStat(BaseModel):
    n_hands_played: int
    n_showdowns: int
    n_won: int
    total_earnings: float


player_dict = {}  # player: games_played, games_won
player_stats_dict: Dict[str, PlayerStat] = {}  # player: {n_hands_played, n_showdowns, n_won, total_earnings}
BLIND_SIZES = "0.25-0.50"


def write(player_dict, outfile='result.txt'):
    """Writes player_dict to result.txt file"""
    with open(outfile, 'w') as f:
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


def run(cbs=None):
    """Reads unzipped folder and updates dictionaries when proper callback fn is provided"""
    filenames = glob.glob(str(DATA_DIR) + f'/01_raw/{BLIND_SIZES}/unzipped/' '/**/*.txt', recursive=True)
    parser = HSmithyParser()
    # parse, encode, vectorize and write the training data from .txt to disk
    for i, filename in enumerate(filenames):
        try:
            parsed_hands = parser.parse_file(filename)
            if cbs:
                [cb(episodes=parsed_hands) for cb in cbs]  # update_player_dict(parsed_hands)
            if i%100==0:
                print(f'Reading {i}-th file...')
        except UnicodeDecodeError:
            print('---------------------------------------')
            print(
                f'Skipping {filename} because it has invalid continuation byte...')
            print('---------------------------------------')


def reduce() -> pd.DataFrame:
    import ast
    with open("result.txt", "r") as data:
        df = pd.DataFrame.from_dict(ast.literal_eval(data.read()), orient='index')
    df = df.sort_values(0, ascending=False)
    df = df[df > 100].dropna()
    return df


def _update_player_earnings(episode: PokerEpisode, top_player: PlayerStack):
    p_name = top_player.player_name
    earned = 0
    pot = 0
    # strip currency symbol and convert to number
    # go thorugh blinds and see if starting stack decreased
    for blind in episode.blinds:
        amt = float(blind.amount[1:])
        if blind.player_name == p_name:
            earned -= amt
        pot += amt
    # go through actions and increase/decrease starting stacks
    for action in episode.actions_total['as_sequence']:
        amt = 0
        if action.action_type != ActionType.FOLD:
            amt = float(action.raise_amount)
        # update pot
        pot += amt
        if action.player_name == p_name:
            if action.action_type == ActionType.FOLD:
                break
            else:
                if action.raise_amount != -1:
                    # subtract chips for top_player
                    earned -= amt
    for winner in episode.winners:
        if winner.name == p_name:
            earned += pot
    player_stats_dict[p_name].total_earnings += earned


def _update_stats(episode: PokerEpisode, top_player: PlayerStack):
    # create new entry in player_stats_dict if necessary
    top_player_name = top_player.player_name
    if top_player_name not in player_stats_dict:
        # {'n_hands_played': 1, 'n_showdowns': 0, 'n_won': 0, 'total_earnings': 0.0}
        player_stats_dict[top_player_name] = PlayerStat(n_hands_played=1,
                                                        n_showdowns=0,
                                                        n_won=0,
                                                        total_earnings=0.0)
    else:
        player_stats_dict[top_player_name].n_hands_played += 1
    # update showdowns and wins
    for showdown_player in episode.showdown_hands:
        if top_player_name == showdown_player.name:
            player_stats_dict[top_player_name].n_showdowns += 1
    for winner in episode.winners:
        if top_player_name == winner.name:
            player_stats_dict[top_player_name].n_won += 1
    # update earnings
    _update_player_earnings(episode, top_player)


def update_stats_dict(top_players: List[str], episodes: Iterable[PokerEpisode]):
    # player: {n_hands_played, n_showdowns, n_won, total_earnings}
    # for each played hand
    for episode in episodes:
        for episode_player in episode.player_stacks:
            if episode_player.player_name in top_players:
                _update_stats(episode, episode_player)


def get_stats():
    with open("result.txt", "r") as data:
        player_dict = ast.literal_eval(data.read())
        player_names = list(player_dict.keys())
        print(player_names)
    cb = partial(update_stats_dict, top_players=player_names)
    run(cbs=[cb])


if __name__ == "__main__":
    # *** First run ***
    # run(cbs=[update_player_dict])  # 1.
    # write(player_dict)
    # df = reduce()  # 2.
    # write(df.to_dict()[0])
    time.sleep(0.5)
    # *** Second run ***
    get_stats()
    print(player_stats_dict)
    serialized_dict = {}
    for k,v in player_stats_dict.items():
        serialized_dict[k] = v.dict()
    write(serialized_dict, outfile='result2.txt')
