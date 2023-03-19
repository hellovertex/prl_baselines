import glob
import json
import os
import re
from pathlib import Path

from tqdm import tqdm

example_ep_without_showdown = """PokerStars Hand #208958099944:  Hold'em No Limit (
$0.25/$0.50 
USD) - 2020/02/07 19:15:13 ET
Table 'Aaltje III' 6-max Seat #2 is the button
Seat 1: SWING BOLOO ($36.51 in chips)
Seat 2: romixon36 ($50 in chips)
Seat 3: supersimple2018 ($52.97 in chips)
Seat 4: Flyyguyy403 ($49.52 in chips)
Seat 5: Clamfish0 ($52 in chips)
Seat 6: JuanAlmighty ($51.77 in chips)
supersimple2018: posts small blind $0.25
Flyyguyy403: posts big blind $0.50
*** HOLE CARDS ***
Clamfish0: folds
JuanAlmighty: raises $1 to $1.50
SWING BOLOO: calls $1.50
romixon36: folds
supersimple2018: raises $6 to $7.50
Flyyguyy403: folds
JuanAlmighty: calls $6
SWING BOLOO: calls $6
*** FLOP *** [5d Qd 7d]
supersimple2018: checks
JuanAlmighty: checks
SWING BOLOO: bets $4
supersimple2018: folds
JuanAlmighty: folds
Uncalled bet ($4) returned to SWING BOLOO
SWING BOLOO collected $21.85 from pot
*** SUMMARY ***
Total pot $23 | Rake $1.15
Board [5d Qd 7d]
Seat 1: SWING BOLOO collected ($21.85)
Seat 2: romixon36 (button) folded before Flop (didn't bet)
Seat 3: supersimple2018 (small blind) folded on the Flop
Seat 4: Flyyguyy403 (big blind) folded before Flop
Seat 5: Clamfish0 folded before Flop (didn't bet)
Seat 6: JuanAlmighty folded on the Flop
"""

example_ep_with_showdown = """PokerStars Hand #208958141851:  Hold'em No Limit ($0.25/$0.50 USD) - 2020/02/07 19:16:54 ET
Table 'Aaltje III' 6-max Seat #3 is the button
Seat 1: SWING BOLOO ($50.86 in chips)
Seat 2: romixon36 ($50 in chips)
Seat 3: supersimple2018 ($50 in chips)
Seat 4: Flyyguyy403 ($49.02 in chips)
Seat 5: Clamfish0 ($52 in chips)
Seat 6: JuanAlmighty ($50 in chips)
Flyyguyy403: posts small blind $0.25
Clamfish0: posts big blind $0.50
*** HOLE CARDS ***
JuanAlmighty: raises $1.25 to $1.75
SWING BOLOO: folds
romixon36: folds
supersimple2018: raises $4.25 to $6
Flyyguyy403: folds
Clamfish0: folds
JuanAlmighty: raises $6.50 to $12.50
supersimple2018: raises $37.50 to $50 and is all-in
JuanAlmighty: calls $37.50 and is all-in
*** FLOP *** [Tc 9h 6c]
*** TURN *** [Tc 9h 6c] [7c]
*** RIVER *** [Tc 9h 6c 7c] [Jd]
*** SHOW DOWN ***
JuanAlmighty: shows [As Ah] (a pair of Aces)
supersimple2018: shows [Qs Qd] (a pair of Queens)
JuanAlmighty collected $98.75 from pot
*** SUMMARY ***
Total pot $100.75 | Rake $2
Board [Tc 9h 6c 7c Jd]
Seat 1: SWING BOLOO folded before Flop (didn't bet)
Seat 2: romixon36 folded before Flop (didn't bet)
Seat 3: supersimple2018 (button) showed [Qs Qd] and lost with a pair of Queens
Seat 4: Flyyguyy403 (small blind) folded before Flop
Seat 5: Clamfish0 (big blind) folded before Flop
Seat 6: JuanAlmighty showed [As Ah] and won ($98.75) with a pair of Aces"""
example_ep_showdown_mucked_hand = """PokerStars Hand #208958463857:  Hold'em No Limit ($0.25/$0.50 USD) - 2020/02/07 19:29:42 ET
Table 'Aaltje III' 6-max Seat #6 is the button
Seat 1: SWING BOLOO ($44.46 in chips)
Seat 2: romixon36 ($56.39 in chips)
Seat 3: supersimple2018 ($52.60 in chips)
Seat 4: Flyyguyy403 ($53.97 in chips)
Seat 5: Clamfish0 ($50 in chips)
Seat 6: doselka ($50 in chips)
SWING BOLOO: posts small blind $0.25
romixon36: posts big blind $0.50
*** HOLE CARDS ***
supersimple2018: folds
Flyyguyy403: folds
supersimple2018 leaves the table
Clamfish0: raises $0.83 to $1.33
doselka: raises $2.42 to $3.75
SWING BOLOO: folds
romixon36: folds
Clamfish0: calls $2.42
*** FLOP *** [5d Ac Tc]
Flyyguyy403 is disconnected
Clamfish0: checks
doselka: bets $2.67
Clamfish0: calls $2.67
*** TURN *** [5d Ac Tc] [Js]
Clamfish0: checks
doselka: checks
*** RIVER *** [5d Ac Tc Js] [Ts]
Clamfish0: checks
doselka: checks
*** SHOW DOWN ***
Clamfish0: shows [Ad 8d] (two pair, Aces and Tens)
doselka: mucks hand
Clamfish0 collected $12.91 from pot
*** SUMMARY ***
Total pot $13.59 | Rake $0.68
Board [5d Ac Tc Js Ts]
Seat 1: SWING BOLOO (small blind) folded before Flop
Seat 2: romixon36 (big blind) folded before Flop
Seat 3: supersimple2018 folded before Flop (didn't bet)
Seat 4: Flyyguyy403 folded before Flop (didn't bet)
Seat 5: Clamfish0 showed [Ad 8d] and won ($12.91) with two pair, Aces and Tens
Seat 6: doselka (button) mucked"""


class DatasetStats:

    def __init__(self,
                 showdown_separator="*** SHOW DOWN ***",
                 summary_separator="*** SUMMARY ***"):
        self.showdown_separator = showdown_separator
        self.summary_separator = summary_separator
        self.total_hands = 0
        self.total_showdowns = 0
        self.n_showdowns_no_mucks = 0
        self.n_showdowns_with_mucks = 0
        # actions are part of dataset stats
        # but these are not computed here
        # but after vectorizing the dataset observations
        # and discretizing the action space

    def reset(self):
        self.total_hands = 0
        self.total_showdowns = 0
        self.n_showdowns_no_mucks = 0
        self.n_showdowns_with_mucks = 0

    def player_mucked(self, showdown: str):
        if 'mucked' in showdown or 'mucks' in showdown:
            return True
        return False

    def update_from_single_episode(self, ep: str):
        self.total_hands += 1
        maybe_showdown = ep.split(self.showdown_separator)
        if len(maybe_showdown) == 1:
            # no showdown
            return
        else:
            self.total_showdowns += 1
            # remove summary, because it in very care occasions contains mucks from
            # non showdown players, that we do not want to count as showdown mucks
            showdown = maybe_showdown[1].split(self.summary_separator)[0]
            if self.player_mucked(showdown):
                self.n_showdowns_with_mucks += 1
            else:
                self.n_showdowns_no_mucks += 1

    def update_from_file(self, file_path):
        with open(file_path, 'r',
                  encoding='utf-8') as f:  # pylint: disable=invalid-name,unspecified-encoding
            hand_database = f.read()
            hands_played = re.split(r'PokerStars Hand #', hand_database)[1:]
            for hand in hands_played:
                self.update_from_single_episode(hand)

    def to_dict(self):
        assert self.n_showdowns_no_mucks + self.n_showdowns_with_mucks == self.total_showdowns
        return {'total_hands': self.total_hands,
                'total_showdowns': self.total_showdowns,
                'n_showdowns_no_mucks': self.n_showdowns_no_mucks,
                'n_showdowns_with_mucks': self.n_showdowns_with_mucks, }


def main():
    unzipped_dir = ''
    filenames = glob.glob(
        f'{unzipped_dir}**/*.txt',
        recursive=True)
    stats = DatasetStats()
    # Update player stats file by file
    for f in tqdm(filenames):
        stats.update_from_file(f)

    # Flush to disk
    to_dict = stats.to_dict()
    outfile = os.path.join('./dataset_stats', f'dataset_stats.json')
    if not os.path.exists(outfile):
        os.makedirs(Path(outfile).parent, exist_ok=True)
    with open(outfile, 'a+') as f:
        f.write(json.dumps(to_dict))


if __name__ == '__main__':
    main()
    # ds = DatasetStats()

    # ds.update_from_single_episode(example_ep_showdown_mucked_hand)
    # assert ds.to_dict()['total_hands'] == 1
    # assert ds.to_dict()['total_showdowns'] == 1
    # assert ds.to_dict()['n_showdowns_with_mucks'] == 1
    # assert ds.to_dict()['n_showdowns_no_mucks'] == 0
    # ds.update_from_single_episode(example_ep_without_showdown)
    # assert ds.to_dict()['total_hands'] == 2
    # assert ds.to_dict()['total_showdowns'] == 1
    # assert ds.to_dict()['n_showdowns_with_mucks'] == 1
    # assert ds.to_dict()['n_showdowns_no_mucks'] == 0
    # ds.update_from_single_episode(example_ep_with_showdown)
    # assert ds.to_dict()['total_hands'] == 3
    # assert ds.to_dict()['total_showdowns'] == 2
    # assert ds.to_dict()['n_showdowns_with_mucks'] == 1
    # assert ds.to_dict()['n_showdowns_no_mucks'] == 1
    # print(ds.to_dict())