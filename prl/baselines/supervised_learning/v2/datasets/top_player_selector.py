import ast
import json
import logging
import os
import warnings
from collections import OrderedDict
import dataclasses
from typing import Dict

import pandas as pd
from tqdm import tqdm

from prl.baselines import DATA_DIR
from prl.baselines.supervised_learning.v2.fast_hsmithy_parser import \
    ParseHsmithyTextToPokerEpisode


@dataclasses.dataclass
class PlayerSelection:
    # don't need these to select top players
    # n_hands_dealt: int
    # n_hands_played: int
    n_showdowns: int
    n_won: int
    total_earnings: float


class TopPlayerSelector:
    def __init__(self,
                 parser: ParseHsmithyTextToPokerEpisode):
        self.parser = parser

    def get_players_showdown_stats(self):
        players: Dict[str, PlayerSelection] = {}

        num_files = self.parser.num_files
        for hand_histories in tqdm(self.parser.parse_hand_histories_from_all_players(),
                                   total=num_files):
            for episode in hand_histories:
                if not episode.has_showdown:
                    continue
                try:
                    # for each player count hands, showdowns and chips won
                    for player in episode.showdown_players:
                        if not player.name in players:
                            players[player.name] = PlayerSelection(n_showdowns=0,
                                                                   n_won=0,
                                                                   total_earnings=0)
                        else:
                            players[player.name].n_showdowns += 1
                            try:
                                # can be negative
                                players[
                                    player.name].total_earnings += player.money_won_this_round
                            except TypeError as e:
                                print(e)
                                logging.warning(
                                    f'Problems parsing showdown of game with ID'
                                    f' {episode.hand_id}. Players were {players}')
                                continue
                            for winner in episode.winners:
                                if winner.name == player.name:
                                    players[player.name].n_won += 1
                except Exception as e:
                    print(e)
                    continue
        return players

    def dir_top_n_players_dict(self, n, min_showdowns):
        raw_dir = os.path.join(DATA_DIR, '01_raw')
        return os.path.join(raw_dir,
                            self.parser.nl,  # replace with self.parser.opt.nl
                            f'top_{n}_players_min_showdowns={min_showdowns}')

    def _get_precomputed_from_disk(self, num_top_players, min_showdowns) -> Dict:
        logging.info(f'Loading top {num_top_players} players dictionary from disk.')
        filename = self.dir_top_n_players_dict(num_top_players, min_showdowns)
        with open(filename, "r") as data:
            top_n_player_dict = ast.literal_eval(data.read())
        return top_n_player_dict

    def get_top_n_players_min_showdowns(self, num_top_players, min_showdowns) -> Dict:
        # from disk
        if os.path.exists(self.dir_top_n_players_dict(num_top_players, min_showdowns)):
            return self._get_precomputed_from_disk(num_top_players, min_showdowns)

        # compute and flush to disk
        else:
            logging.info(f'Extracting hand history of top {num_top_players} players. '
                         f'This may take a few minutes up to several hours')
            # parse all files
            players = self.get_players_showdown_stats()
            # sort for winnings per showdown
            serialized = dict([(k, dataclasses.asdict(v)) for k, v in players.items()])
            df = pd.DataFrame.from_dict(serialized, orient='index')
            df = df.sort_values('n_showdowns', ascending=False).dropna()
            # only use players that went to more than 10_000 showdowns
            df = df[df['n_showdowns'] > min_showdowns]
            df = df['total_earnings'] / df['n_showdowns']
            df = df.sort_values(ascending=False).dropna()
            if len(df) < num_top_players:
                warnings.warn(f'{num_top_players} top players were '
                              f'requests, but only {len(df)} players '
                              f'are in database. Please decrease '
                              f'`num_top_players` or provide more '
                              f'hand histories.')
            d = df[:num_top_players].to_dict()
            # maintain rank from 1 to n
            ordered_dict = OrderedDict((k, d.get(k)) for k in df.index[:num_top_players])
            # flush to disk
            self.write(ordered_dict, self.dir_top_n_players_dict(num_top_players,
                                                                 min_showdowns))

            return ordered_dict

    @staticmethod
    def write(player_dict, outfile):
        """Writes player_dict to result.txt file"""
        with open(outfile, 'w') as f:
            f.write(json.dumps(player_dict))
