import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from prl.baselines.analysis.core.experiment_runner import PlayerWithCardsAndPosition
from prl.baselines.evaluation.pokersnowie.converter_888 import Converter888
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode
from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import \
    HSmithyParser


def write_p_episodes_to_disk(pname):
    unzipped_dir = ''
    max_episodes_per_file = 2500
    filenames = glob.glob(
        f'{unzipped_dir}**/*.txt',
        recursive=True)
    current_hands = []
    # Update player stats file by file
    for fname in tqdm(filenames):
        try:
            with open(fname, 'r', encoding='utf-8') as f:  # pylint: disable=invalid-name
                hand_database = f.read()
                hands_played = re.split(r'PokerStars Hand #', hand_database)[1:]
                for hand in hands_played:
                    if pname in hand:
                        if len(current_hands) < max_episodes_per_file:
                            current_hands.append("PokerStars Hand #" + hand)
                        else:
                            # write current_hands to disk
                            outfile = os.path.join('./player_snowies', f'episodes.txt')
                            if not os.path.exists(outfile):
                                os.makedirs(Path(outfile).parent, exist_ok=True)
                            with open(outfile, 'a+') as f:
                                f.write(''.join(str(i) for i in current_hands))
                            return
        except UnicodeDecodeError:
            # skip few files with invalid continuation bytes
            pass


# # need to parse to pokerepisode only those games that have showdown
#    # pokerepisodes using 888 to disk and then to snowie
#    unzipped_dir = ''
#
#    # for p in best_players:
#    #     write_p_episodes_to_disk(p)
#    # for p in worst_players:
#    #     write_p_episodes_to_disk(p)


class SelectedPlayerStats:
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


class HSmithyStats:
    """Reads .txt files with poker games crawled from Pokerstars.com and looks for specific players.
     If found, writes them back to disk in a separate place.
     This is done to speed up parsing of datasets."""

    def __init__(self, pname):
        self.pstats = SelectedPlayerStats(pname=pname)
        self.target_player = pname

    def split_next_round(self, stringval):
        return True

    def compute_stats(self, hands_played):
        for current in hands_played:  # c for current_hand
            # Only parse hands that went to Showdown stage, i.e. were shown
            # skip hands without target player
            if not self.target_player in current:
                continue
            if f'{self.target_player}: sits out' in current:
                continue
            # accumulate stats
            self.pstats.update(current)

    def compute_from_file(self, file_path_in, target_player):
        self._variant = 'NoLimitHoldem'  # todo parse variant from filename
        self.target_player = target_player
        with open(file_path_in, 'r',
                  encoding='utf-8') as f:  # pylint: disable=invalid-name,unspecified-encoding
            hand_database = f.read()
            hands_played = re.split(r'PokerStars Hand #', hand_database)[1:]
            self.compute_stats(hands_played)


class PlayerSnowies:

    def __init__(self):
        self.parser = HSmithyParser()

    def make_episodes_from_file(self, filepath) -> List[PokerEpisode]:
        # parse file
        poker_episodes = self.parser.parse_file(filepath)
        # simulate environment
        # encode PokerEpisode
        # convert PokerEpisode to PokerSnowie
        # write to file
        pass

    def make_episodes_from_files(self, unzipped_dir):
        pass


best_players = ['ishuha',
                'Sakhacop',
                'nastja336',
                'Lucastitos',
                'I LOVE RUS34',
                'SerAlGog', ]

worst_players = ['Calabazaking',
                 'jenya86rus',
                 'podumci',
                 'Inas21',
                 'Fischkop993']


def is_showdown_player(target_player, episode):
    if not '*** SHOW DOWN ***' in episode:
        return False
    else:
        maybe_showdown = episode.split('*** SHOW DOWN ***')
        if len(maybe_showdown) == 1:
            # no showdown
            return False
        showdown = maybe_showdown[1].split('*** SUMMARY ***')[0]
        if target_player in showdown:
            return True
    return False


def parse_showdown_hands(ep: PokerEpisode):
    if not ep:
        return ep
    player_hands = []
    for hand in ep.showdown_hands:
        player_hands.append(PlayerWithCardsAndPosition(cards=hand.cards, name=hand.name))
    ep.info['player_hands'] = player_hands
    return ep


def run(files, target_player, max_episodes_per_file=2500):
    current_hands = []
    parser = HSmithyParser()
    converter = Converter888()
    for fname in tqdm(files):
        try:
            with open(fname, 'r', encoding='utf-8') as f:  # pylint: disable=invalid-name
                hand_database = f.read()
                hands_played = re.split(r'PokerStars Hand #', hand_database)[1:]
                for hand in hands_played:
                    if is_showdown_player(target_player, hand):
                        if len(current_hands) < max_episodes_per_file:
                            ep = parser.parse_episode(hand)
                            ep = parse_showdown_hands(ep)
                            if ep:
                                showdown_ep = converter.from_poker_episode(
                                    ep, [target_player])
                                for observer_relative in showdown_ep:
                                    current_hands.append(observer_relative)
                        else:
                            # write current_hands to disk
                            outfile = os.path.join('./player_snowies', f'episodes.txt')
                            if not os.path.exists(outfile):
                                os.makedirs(Path(outfile).parent, exist_ok=True)
                            with open(outfile, 'a+') as f:
                                f.write(''.join(str(i) for i in current_hands))
                            return
        except UnicodeDecodeError:
            # skip few files with invalid continuation bytes
            pass
    outfile = os.path.join(f'./player_snowies', f'{target_player}.txt')
    if not os.path.exists(outfile):
        os.makedirs(Path(outfile).parent, exist_ok=True)
    with open(outfile, 'a+') as f:
        f.write(''.join(str(i) for i in current_hands))


def main():
    unzipped_dir = '/home/sascha/Documents/github.com/prl_baselines/data/01_raw/NL50/all_players'
    filenames = glob.glob(f'{unzipped_dir}**/*.txt', recursive=True)
    for p in best_players:
        run(filenames, p)
    for p in worst_players:
        run(filenames, p)


if __name__ == '__main__':
    main()
