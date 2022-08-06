import glob
import json
import os
from typing import Iterable

from prl.baselines.supervised_learning.config import DATA_DIR
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode
from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser

player_dict = {}  # player: games_played, games_won
BLIND_SIZES = "0.25-0.50"


def run(episodes: Iterable[PokerEpisode]):
    # for each played hand
    for episode in episodes:
        # for each showdown player
        for player in episode.showdown_hands:
            # if player has not been added to player_dict
            if not player.name in player_dict:
                player_dict[player.name] = {'games_played': 0,
                                            'games_won': 0}
            # player exists in database
            else:
                player_dict[player.name]['games_played'] += 1
                for winner in episode.winners:
                    if winner.name == player.name:
                        player_dict[player.name]['games_won'] += 1


if __name__ == "__main__":
    filenames = glob.glob(str(DATA_DIR) + f'/01_raw/{BLIND_SIZES}/unzipped/' '/**/*.txt', recursive=True)
    parser = HSmithyParser()
    # parse, encode, vectorize and write the training data from .txt to disk
    for i, filename in enumerate(filenames):
        try:
            parsed_hands = parser.parse_file(filename)
            run(parsed_hands)
            print(f'After {i}-th file we have {len(player_dict.keys())} different players')
        except UnicodeDecodeError:
            print('---------------------------------------')
            print(
                f'Skipping {filename} because it has invalid continuation byte...')
            print('---------------------------------------')
    with open('result.txt', 'w') as f:
        f.write(json.dumps(player_dict))
