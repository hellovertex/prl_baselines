"""
player_name | Agression Factor | Tightness | acceptance level | Agression Factor NN | tightness NN
--------------------------------------------------------------------------------------------------
Agression Factor (AF): #raises / #calls
Tightness: % hands played (not folded immediately preflop)
"""
import glob
import json
import multiprocessing
import time
from pathlib import Path

from prl.environment.Wrappers.augment import AugmentObservationWrapper

from prl.baselines.analysis.core.analyzer import PlayerAnalyzer
from prl.baselines.analysis.core.stats import PlayerStats
from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser


def analysis(filename):
    parser = HSmithyParser()
    pname = Path(filename).stem
    player_stats = [PlayerStats(pname=pname)]
    analyzer = PlayerAnalyzer(baseline=None, player_stats=player_stats, env_wrapper_cls=AugmentObservationWrapper)

    t0 = time.time()
    parsed_hands = list(parser.parse_file(filename))
    print(f'Parsing file {filename} took {time.time() - t0} seconds.')
    num_parsed_hands = len(parsed_hands)
    print(f'num_parsed_hands = {num_parsed_hands}')
    for ihand, hand in enumerate(parsed_hands):
        print(f'Analysing hand {ihand} / {num_parsed_hands}')
        analyzer.analyze_episode(hand, pname=pname)
    with open(f'stats_{pname}.txt', 'a+') as f:
        for stat in analyzer.player_stats:
            f.write(json.dumps(stat.to_dict()))
    return f"Success. Wrote stats to {f'stats_{pname}.txt'}"


if __name__ == "__main__":
    acceptance_levels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    # ppl and ppool filenames -- single file and globbed files
    # implement parser, encoder, analyzer pipeline
    unzipped_dir = "/home/hellovertex/Documents/github.com/hellovertex/prl_baselines/data/player_data_test"
    filenames = glob.glob(unzipped_dir.__str__() + '/**/*.txt', recursive=True)

    start = time.time()
    p = multiprocessing.Pool()
    t0 = time.time()

    for x in p.imap_unordered(analysis, filenames):
        print(x + f'. Took {time.time() - t0} seconds')
    print(f'Finished job after {time.time() - start} seconds.')

    p.close()
