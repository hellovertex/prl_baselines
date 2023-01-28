# todo consider converting this to .ipynb and move to scripts/
import ast
import glob
import json
import time

import gdown
from pydantic import BaseModel
from functools import partial
from typing import Iterable, Dict, List

import pandas as pd

from prl.baselines.supervised_learning.config import DATA_DIR
from prl.baselines.supervised_learning.data_acquisition.core import utils
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode, PlayerStack, ActionType
from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser

# How many showdown hands must be available to generate a summary of the player
# the more data available, the higher it should be, e.g. try between 100 and 1000
MIN_SHOWDOWNS = 100


class PlayerStat(BaseModel):
    n_hands_played: int
    n_showdowns: int
    n_won: int
    total_earnings: float


player_dict = {}  # player: games_played, games_won
player_stats_dict: Dict[str, PlayerStat] = {}  # player: {n_hands_played, n_showdowns, n_won, total_earnings}
BLIND_SIZES = "0.25-0.50"


def write(player_dict, outfile):
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


def unzip(out_dir, from_gdrive_id):
    # try to download from_gdrive to out.zip
    zipfiles = [gdown.download(id=from_gdrive_id,
                               output=f"{out_dir}/bulkhands_.zip",
                               quiet=False)]
    out_dir = f'{out_dir}/unzipped'
    # creates out_dir if it does not exist
    # extracts zip file, only if extracted files with same name do not exist
    [utils.extract(f_zip, out_dir=out_dir) for f_zip in zipfiles]
    return out_dir


def run(cbs=None):
    """Reads unzipped folder and updates dictionaries when proper callback fn is provided"""
    unzipped_dir = unzip(out_dir='/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50',
                         from_gdrive_id="18GE6Xw4K1XE2PNiXSyh762mJ5ZCRl2SO")
    filenames = glob.glob(f'{unzipped_dir}/**/*.txt', recursive=True)

    parser = HSmithyParser()
    # parse, encode, vectorize and write the training data from .txt to disk
    for i, filename in enumerate(filenames):
        try:
            parsed_hands = parser.parse_file(filename)
            if cbs:
                [cb(episodes=parsed_hands) for cb in cbs]  # update_player_dict(parsed_hands)
            if i % 100 == 0:
                print(f'Reading {i}-th file...')
        except UnicodeDecodeError:
            print('---------------------------------------')
            print(
                f'Skipping {filename} because it has invalid continuation byte...')
            print('---------------------------------------')


def reduce() -> pd.DataFrame:
    import ast
    with open(abs_path + "/eda_players.txt", "r") as data:
        df = pd.DataFrame.from_dict(ast.literal_eval(data.read()), orient='index')
    df = df.sort_values(0, ascending=False)
    df = df[df > MIN_SHOWDOWNS].dropna()
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
    abs_path = str(DATA_DIR) + '/01_raw' + f'/{blind_sizes}' + "/eda_players.txt"
    with open(abs_path, "r") as data:
        player_dict = ast.literal_eval(data.read())
        player_names = list(player_dict.keys())
        print(player_names)
    cb = partial(update_stats_dict, top_players=player_names)
    run(cbs=[cb])


if __name__ == "__main__":
    blind_sizes = '0.25-0.50'
    abs_path = str(DATA_DIR) + '/01_raw' + f'/{blind_sizes}'
    # *** First run ***
    run(cbs=[update_player_dict])  # 1.
    write(player_dict, outfile=abs_path + "/eda_players.txt")
    df = reduce()  # 2.
    write(df.to_dict()[0], outfile=abs_path + "/eda_players.txt")
    time.sleep(0.5)
    # *** Second run ***
    get_stats()
    print(player_stats_dict)
    serialized_dict = {}
    for k, v in player_stats_dict.items():
        serialized_dict[k] = v.dict()
    write(serialized_dict, outfile=abs_path + "/eda_result.txt")

"""
# Todos

1. select 20 best players
2. extract their dataset 
train
 - 3. recompute action indices
 - 4. check if cards in non-terminal obs
 - 5. train again RF or NN

6. determine fold probability
 - read game log paper and use MC to compute +- EV moves

Test baseline mimic quality
- 7. using sklansky score
- 8. using pokersnowie error rate

9. Run Heads Up training of Baseline vs Rainbow 
"""