"""
player_name | Agression Factor | Tightness | acceptance level | Agression Factor NN | tightness NN
--------------------------------------------------------------------------------------------------
Agression Factor (AF): #raises / #calls
Tightness: % hands played (not folded immediately preflop)
"""
import glob

from prl.baselines.evaluation.analyzer import PlayerAnalyzer
from prl.baselines.evaluation.stats import PlayerStats
from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser

# [ ] 1. go into encoder and start building stats table while encoding from_dir = player_data with network loaded
# [x] 2. fix win_prob < ? condition - fix pot odds && fix whatif total_to_call=0
# --> if win_prob < total_to_call / (obs[cols.Pot_amt] + total_to_call):
# i: pot odds) this is correct. the player should win more than 1/(3+1) = 25% of the time.
# i.e. if he wins less often he shpould fold
# ii: whatif totaltocall=0) then we want to sometimes check sometimes raise
# we first see if we can raise which is determined by the acceptance level so it is a perfect hyperparameter
# to tune the AF and tightness of the baseline  % todo put this in the .tex file -- its our contrib to
#  todo: have made this a hyperparam to evolve from the game logs baseline paper

# fold probability is marginalized by tightness and by acceptance but we can fix the tightness to the players
# so the only remaining parameter is acceptance level. Again, todo move this to .tex

acceptance_levels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
# ppl and ppool filenames -- single file and globbed files
filenames = []
# implement parser, encoder, analyzer pipeline
unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data"
filenames = glob.glob(unzipped_dir.__str__() + '/**/*.txt', recursive=True)
parser = HSmithyParser()
baseline = None   # todo load baseline model
player_stats = []
for ifile, filename in enumerate(filenames):
    pname = ""
    player_stats.append(PlayerStats(pname=pname))

analyzer = PlayerAnalyzer(baseline=baseline, player_stats=player_stats)

for ifile, filename in enumerate(filenames):
    pname = ""
    parsed_hands = parser.parse_file(filename)
    for ihand, hand in enumerate(parsed_hands):
        analyzer.analyze_episode(hand, pname=pname)

stats_baseline = analyzer.baseline_stats.to_dict()