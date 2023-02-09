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
class SelectedPlayerStats:
    def __init__(self,
                 pname,
                 preflop_sep="HOLE CARDS",
                 flop_sep="FLOP",
                 turn_sep="TURN",
                 river_sep="RIVER",
                 summary_sep="SUMMARY"):
        self.pname = pname
        self.total_number_of_hands_seen = 0
        self.preflop_sep = preflop_sep
        self.flop_sep = flop_sep
        self.turn_sep = turn_sep
        self.river_sep = river_sep
        self.summary_sep = summary_sep
        # relevant stats for VPIP, PFR and AF
        # vpip: voluntarily put money into the preflop pot
        self.n_immediate_preflop_folds = 0
        self.n_big_blind_checked_preflop = 0
        # PFR: hands bet or raised preflop
        # the larger vpip - pfr the more passive a player
        self.n_raises_or_bets_preflop = 0
        # AF: total bets or raises / total calls
        self.times_bet_or_raised_pf = 0
        self.times_bet_or_raised_f = 0
        self.times_bet_or_raised_t = 0
        self.times_bet_or_raised_r = 0
        self.times_called_pf = 0
        self.times_called_f = 0
        self.times_called_t = 0
        self.times_called_r = 0

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
        # split at flop and strip from turn onwards
        flop = self.split_at_round(self.flop_sep, current_episode)
        flop = self.strip_next_round(self.turn_sep, flop)
        # split at turn and strip from river onwards
        turn = self.split_at_round(self.turn_sep, current_episode)
        turn = self.strip_next_round(self.river_sep, turn)
        # split at river and strip from summary onwards
        river = self.split_at_round(self.river_sep, current_episode)
        river = self.strip_next_round(self.summary_sep, river)
        summary = self.split_at_round(self.summary_sep, current_episode)

        return {'preflop': hole_cards,
                'flop': flop,
                'turn': turn,
                'river': river,
                'summary': summary}

    def update_vpip(self, preflop_str):
        has_bet = f'{self.pname}: bets' in preflop_str
        has_raised = f'{self.pname}: raises' in preflop_str
        has_called = f'{self.pname}: calls' in preflop_str
        has_folded = f'{self.pname}: folds' in preflop_str
        if not has_bet and not has_raised and not has_called:
            # if the big blind checks the vpip does not increase
            if has_folded:
                self.n_immediate_preflop_folds += 1
            if 'checks' in preflop_str:
                self.n_big_blind_checked_preflop += 1

    def update_pfr(self, preflop_str):
        has_bet = f'{self.pname}: bets' in preflop_str
        has_raised = f'{self.pname}: raises' in preflop_str
        if has_bet or has_raised:
            # even if player folds to 3bet,
            # an initial raise still counts towards pfr
            self.n_raises_or_bets_preflop += 1

    def update_preflop(self, preflop_str):
        assert not self.flop_sep in preflop_str
        assert not self.turn_sep in preflop_str
        assert not self.river_sep in preflop_str
        assert not self.summary_sep in preflop_str
        # VPIP: Voluntarily Put money In Pot
        self.update_vpip(preflop_str)
        # PFR: Preflop Raises
        self.update_pfr(preflop_str)
        # AF: Aggression factor: (N bets + N raises) / N calls
        af = self.update_af(preflop_str)
        self.times_called_pf += af['n_calls']
        self.times_bet_or_raised_pf += af['n_bets'] + af['n_raises']

    def update_af(self, round_str):
        return {'n_bets': round_str.count(f'{self.pname}: bets'),
                'n_raises': round_str.count(f'{self.pname}: raises'),
                'n_calls': round_str.count(f'{self.pname}: calls')}

    def update_flop(self, flop_str):
        assert not self.preflop_sep in flop_str
        assert not self.turn_sep in flop_str
        assert not self.river_sep in flop_str
        assert not self.summary_sep in flop_str
        # AF: Aggression factor: (N bets + N raises) / N calls
        af = self.update_af(flop_str)
        self.times_called_f += af['n_calls']
        self.times_bet_or_raised_f += af['n_bets'] + af['n_raises']

    def update_turn(self, turn_str):
        assert not self.preflop_sep in turn_str
        assert not self.flop_sep in turn_str
        assert not self.river_sep in turn_str
        assert not self.summary_sep in turn_str
        # AF: Aggression factor: (N bets + N raises) / N calls
        af = self.update_af(turn_str)
        self.times_called_t += af['n_calls']
        self.times_bet_or_raised_t += af['n_bets'] + af['n_raises']

    def update_river(self, river_str):
        assert not self.preflop_sep in river_str
        assert not self.flop_sep in river_str
        assert not self.turn_sep in river_str
        assert not self.summary_sep in river_str
        # AF: Aggression factor: (N bets + N raises) / N calls
        af = self.update_af(river_str)
        self.times_called_r += af['n_calls']
        self.times_bet_or_raised_r += af['n_bets'] + af['n_raises']

    def update(self, current_episode: str):
        self.total_number_of_hands_seen += 1
        rounds = self.rounds(current_episode)
        self.update_preflop(rounds['preflop'])

    def to_dict(self):
        vpiped = self.total_number_of_hands_seen - self.n_immediate_preflop_folds - self.n_big_blind_checked_preflop
        af = (self.times_bet_or_raised_pf +
              self.times_bet_or_raised_f +
              self.times_bet_or_raised_t +
              self.times_bet_or_raised_r) / (self.times_called_pf +
                                             self.times_called_f +
                                             self.times_called_t +
                                             self.times_called_r)
        return {
            'vpip': vpiped / self.total_number_of_hands_seen,
            'pfr': self.n_raises_or_bets_preflop / self.total_number_of_hands_seen,
            'af': af
        }


class HSmithyStats:
    """Reads .txt files with poker games crawled from Pokerstars.com and looks for specific players.
     If found, writes them back to disk in a separate place.
     This is done to speed up parsing of datasets."""

    def __init__(self, pname):
        self.pstats = SelectedPlayerStats(pname=pname)

    def split_next_round(self, stringval):
        return True

    def _compute_stats(self, hands_played):
        for current in hands_played:  # c for current_hand
            # Only parse hands that went to Showdown stage, i.e. were shown
            # skip hands without target player
            if not self.target_player in current:
                continue
            if f'{self.target_player}: sits out' in current:
                continue
            # accumulate stats
            self.pstats.update(current)

    def compute_from_file(self, file_path_in, file_path_out, target_player):
        self._variant = 'NoLimitHoldem'  # todo parse variant from filename
        self.target_player = target_player
        self.file_path_out = file_path_out
        with open(file_path_in, 'r', encoding='utf-8') as f:  # pylint: disable=invalid-name,unspecified-encoding
            hand_database = f.read()
            hands_played = re.split(r'PokerStars Hand #', hand_database)[1:]
            self._compute_stats(hands_played)
