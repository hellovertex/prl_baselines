import os
import re

import click

from prl.baselines.supervised_learning.v2.datasets.dataset_config import (
    arg_nl,
    arg_from_gdrive_id,
    arg_num_top_players,
    DatasetConfig
)
from typing import Dict, List

from prl.baselines.supervised_learning.v2.datasets.raw_data import \
    make_raw_data_if_not_exists_already
from prl.baselines.supervised_learning.v2.datasets.top_player_selector import \
    TopPlayerSelector
from prl.baselines.supervised_learning.v2.fast_hsmithy_parser import \
    ParseHsmithyTextToPokerEpisode


def player_name_to_alias_directory(selected_player_names: List[str],
                                   target_player_name: str,
                                   dataset_config: DatasetConfig):
    for i, name in enumerate(selected_player_names):
        if name == target_player_name:
            rank = i + 1
            return os.path.join(
                dataset_config.dir_raw_data_top_players,
                f'PlayerRank{str(rank).zfill(3)}')


# ---------------------------- PokerStars-Parser ---------------------------------
class _HudStats:
    def __init__(self,
                 pname,
                 preflop_sep="*** HOLE CARDS ***",
                 flop_sep="*** FLOP ***",
                 turn_sep="*** TURN ***",
                 river_sep="*** RIVER ***",
                 summary_sep="*** SUMMARY ***"):
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

        # WTSD: Went to showdown
        self.participated_in_showdown = 0
        self.won_showdown = 0

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
        river = self.strip_next_round(self.summary_sep, river)
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
        # AF: Aggression factor: (N bets + N raises) / N calls
        af = self.update_af(flop_str)
        self.times_called_f += af['n_calls']
        self.times_bet_or_raised_f += af['n_bets'] + af['n_raises']

    def update_turn(self, turn_str):
        # AF: Aggression factor: (N bets + N raises) / N calls
        af = self.update_af(turn_str)
        self.times_called_t += af['n_calls']
        self.times_bet_or_raised_t += af['n_bets'] + af['n_raises']

    def update_river(self, river_str):
        # AF: Aggression factor: (N bets + N raises) / N calls
        af = self.update_af(river_str)
        self.times_called_r += af['n_calls']
        self.times_bet_or_raised_r += af['n_bets'] + af['n_raises']

    def skip_invalid(self, current_episode: str):
        # Skip weird games
        if "*** SECOND FLOP ***" in current_episode:
            return True
        if "*** SECOND TURN ***" in current_episode:
            return True
        if "*** SECOND RIVER ***" in current_episode:
            return True
        return False

    def update_summary(self, summary_str):
        # todo: update wtsd
        lines = summary_str.split('\n')
        if 'share' in summary_str:
            a = 1
        for l in lines:
            if self.pname in l:
                if 'mucked' in l or 'showed' in l:
                    self.participated_in_showdown += 1
                if 'won' in l:
                    self.won_showdown += 1

    def update(self, current_episode: str):
        if self.skip_invalid(current_episode):
            return
        self.total_number_of_hands_seen += 1
        rounds = self.rounds(current_episode)
        self.update_preflop(rounds['preflop'])
        self.update_flop(rounds['flop'])
        self.update_turn(rounds['turn'])
        self.update_river(rounds['river'])
        self.update_summary(rounds['summary'])

    def to_dict(self):
        vpiped = self.total_number_of_hands_seen - self.n_immediate_preflop_folds - self.n_big_blind_checked_preflop
        n_calls = (self.times_called_pf +
                   self.times_called_f +
                   self.times_called_t +
                   self.times_called_r)
        n_bets_or_raises = (self.times_bet_or_raised_pf +
                            self.times_bet_or_raised_f +
                            self.times_bet_or_raised_t +
                            self.times_bet_or_raised_r)
        if n_calls == 0:
            af = n_bets_or_raises
        else:
            af = n_bets_or_raises / n_calls
        if self.total_number_of_hands_seen == 0:
            return {'vpip': -127,
                    'pfr': -127,
                    'af': -127,
                    'wtsd': -127,
                    'total_number_of_samples': 0}
        return {
            'vpip': vpiped / self.total_number_of_hands_seen,
            'pfr': self.n_raises_or_bets_preflop / self.total_number_of_hands_seen,
            'af': af,
            'total_number_of_samples': self.total_number_of_hands_seen,
            'wtsd': self.participated_in_showdown / self.total_number_of_hands_seen,
            'won': self.won_showdown
        }


class PlayerStats:
    """Reads .txt files with poker games crawled from Pokerstars.com and looks for specific players.
     If found, writes them back to disk in a separate place.
     This is done to speed up parsing of datasets."""

    def __init__(self, pname, hudstats=None):
        self.stats = _HudStats(pname=pname) if hudstats is None else hudstats
        assert hasattr(self.stats, 'update')

    def split_next_round(self, stringval):
        return True

    def _update_stats(self, hands_played):
        for current in hands_played:  # c for current_hand
            # Only parse hands that went to Showdown stage, i.e. were shown
            # skip hands without target player
            if not self.target_player in current:
                continue
            if f'{self.target_player}: sits out' in current:
                continue
            # accumulate stats
            self.stats.update(current)

    def update_from_episode(self, episode: str):
        if not self.target_player in episode:
            return
        if f'{self.target_player}: sits out' in episode:
            return
        # accumulate stats
        self.stats.update(episode)

    def update_from_file(self, file_path_in, target_player):
        self._variant = 'NoLimitHoldem'  # todo parse variant from filename
        self.target_player = target_player
        with open(file_path_in, 'r',
                  encoding='utf-8') as f:  # pylint: disable=invalid-name,unspecified-encoding
            hand_database = f.read()
            hands_played = re.split(r'PokerStars Hand #', hand_database)[1:]
            self._update_stats(hands_played)


class DatasetStats:

    def __init__(self,
                 dataset_config: DatasetConfig,
                 top_player_selector: TopPlayerSelector):
        self.dataset_config = dataset_config
        self.top_player_selector = top_player_selector
        self.total_hands = 0
        self.total_showdowns = 0
        self.n_showdowns_no_mucks = 0
        self.n_showdowns_with_mucks = 0
        # large lookup containing per player IDs of all hands played
        # useful to match what each player knows about its opponents,
        # for example when computing specific hud stats
        self.player_names_to_hand_ids: Dict[str, List[int]] = {}
        self.heroes_hud_stats_lookup_table: Dict[str, Dict[str, PlayerStats]] = {}
        self.pattern = re.compile(r'Seat \d:')
        self.selected_player_names = list(
            self.top_player_selector.get_top_n_players_min_showdowns(
                self.dataset_config.num_top_players,
                self.dataset_config.min_showdowns
            ).keys())

    def get_player_names_from_episode(self, episode: str) -> List[str]:
        """
        :param episode:
        PokerStars Hand #208958242124:  Hold'em No Limit ($0.25/$0.50 USD) - 2020/02/07 19:20:58 ET
        Table 'Aaltje III' 6-max Seat #6 is the button
        Seat 1: SWING BOLOO ($44.86 in chips)
        Seat 2: romixon36 ($50 in chips)
        Seat 3: supersimple2018 ($55.66 in chips)
        Seat 4: Flyyguyy403 ($49.46 in chips)
        Seat 5: Clamfish0 ($51.25 in chips)
        Seat 6: JuanAlmighty ($98 in chips), etc.
        """
        player_names = []
        for line in episode.split("\n"):
            if 'Seat' in line:
                for token in self.pattern.split(line):
                    if token:
                        name = token.split('(')[0].strip()
                        player_names.append(name)
        return player_names

    def update_from_file(self, file_path):
        hands_played = self.hands_histories(file_path)
        return

    @staticmethod
    def hands_histories(file_path):
        with open(file_path, 'r',
                  encoding='utf-8') as f:  # pylint: disable=invalid-name,unspecified-encoding
            hand_database = f.read()
            hands_played = re.split(r'PokerStars Hand #', hand_database)[1:]
        return hands_played

    def to_dict(self):
        return {'total_hands': self.total_hands,
                'total_showdowns': self.total_showdowns,
                'n_showdowns_no_mucks': self.n_showdowns_no_mucks,
                'n_showdowns_with_mucks': self.n_showdowns_with_mucks, }

    def _make_top_player_hud_stat_lookup_tables_if_missing(self):
        # todo: check if they exist already
        for hero in self.selected_player_names:
            filename = player_name_to_alias_directory(self.selected_player_names,
                                                      hero,
                                                      self.dataset_config)
            for episode in self.hands_histories(filename):
                for player in self.get_player_names_from_episode(episode):
                    if player != hero:
                        if player in self.heroes_hud_stats_lookup_table[hero]:
                            self.heroes_hud_stats_lookup_table[hero][
                                player].update_from_episode(episode)
                        else:
                            self.heroes_hud_stats_lookup_table[hero][player] = \
                                PlayerStats(player)

    def _make_dataset_summary_if_missing(self):
        # todo make stats and check if missing
        pass

    def generate_if_missing(self):
        # make_csv_files_with_dataset_summary_
        # need to sweep through
        # all files in self.dataset_config.dir_raw_data_top_players
        # twice -- first sweep computes dataset stats
        # for all player_names
        # update player stats
        output_dir = self.dataset_config.dir_data_summary
        self._make_top_player_hud_stat_lookup_tables_if_missing()
        self._make_dataset_summary_if_missing()


@click.command()
@arg_num_top_players
@arg_nl
@arg_from_gdrive_id
def main(num_top_players, nl, from_gdrive_id):
    dataset_config = DatasetConfig(num_top_players=num_top_players,
                                   nl=nl,
                                   from_gdrive_id=from_gdrive_id)
    make_raw_data_if_not_exists_already(dataset_config)
    parser_cls = ParseHsmithyTextToPokerEpisode
    selector = TopPlayerSelector(parser=parser_cls(
        dataset_config=dataset_config))
    stats = DatasetStats(dataset_config, selector)
    stats.generate_if_missing()


if __name__ == '__main__':
    main()
