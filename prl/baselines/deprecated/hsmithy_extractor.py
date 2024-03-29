""" This module will
 - read .txt files inside ./data/
 - parse them to create corresponding PokerEpisode objects. """
import os
import re
from typing import List, Tuple, Dict, Iterator, Iterable, Generator

from prl.baselines.supervised_learning.data_acquisition.core.parser import Parser, PokerEpisode, Action, ActionType, \
    PlayerStack, Blind, PlayerWithCards, PlayerWinningsCollected
from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser

# REGEX templates
# PLAYER_NAME_TEMPLATE = r'([a-zA-Z0-9_.@#!-]+\s?[-@#!_.a-zA-Z0-9]*)'
# PLAYER_NAME_TEMPLATE = r'([óa-zA-Z0-9_.@#!-]+\s?[-@#!_.a-zA-Z0-9ó]*\s?[-@#!_.a-zA-Z0-9ó]*)'
# compile this with re.UNICODE to match any unicode char like é ó etc
PLAYER_NAME_TEMPLATE = r'([\w_.@#!-]+\s?[-@#!_.\w]*\s?[-@#!_.\w]*)'
STARTING_STACK_TEMPLATE = r'\(([$€￡Â£]+\d+.?\d*)\sin chips\)'
MATCH_ANY = r'.*?'  # not the most efficient way, but we prefer readabiliy (parsing is one time job)
POKER_CARD_TEMPLATE = r'[23456789TJQKAjqka][SCDHscdh]'
CURRENCY_SYMBOLS = ['$', '€', '￡', 'Â£']  # only these are currently supported, â‚¬ is € encoded


# ---------------------------- PokerStars-Parser ---------------------------------

class HSmithyExtractor:
    """Reads .txt files with poker games crawled from Pokerstars.com and looks for specific players.
     If found, writes them back to disk in a separate place.
     This is done to speed up parsing of datasets."""

    def _extract_hands(self, hands_played):
        for current in hands_played:  # c for current_hand
            # Only parse hands that went to Showdown stage, i.e. were shown
            if not '*** SHOW DOWN ***' in current:
                continue

            for target_player in self.target_players:
                if target_player in current:
                    result = "PokerStars Hand #" + current
                    if not os.path.exists(self.file_path_out):
                        os.makedirs(self.file_path_out)
                    with open(f'{self.file_path_out+"/"+target_player}.txt', 'a+', encoding='utf-8') as f:
                        f.write(result)

    def extract_file(self, file_path_in, file_path_out, target_players):
        self._variant = 'NoLimitHoldem'
        self.target_players = target_players
        self.file_path_out = file_path_out

        with open(file_path_in, 'r', encoding='utf-8') as f:  # pylint: disable=invalid-name,unspecified-encoding
            hand_database = f.read()
            hands_played = re.split(r'PokerStars Hand #', hand_database)[1:]
            self._extract_hands(hands_played)
            
    def extract_files(self,
                      fpaths,
                      file_path_out,
                      target_players):
        n_files = len(fpaths)
        n_files_skipped = 0
        for i, f in enumerate(fpaths[:-1]):
            print(f'Extracting file {i}/{n_files}....')
            try:
                self.extract_file(f, file_path_out, target_players)
            except UnicodeDecodeError:
                n_files_skipped += 1
        return f"Success. Skipped {n_files_skipped} / {n_files}."
