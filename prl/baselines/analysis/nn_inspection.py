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

from prl.baselines.agents.tianshou_agents import BaselineAgent
from prl.baselines.analysis.core.analyzer import PlayerAnalyzer
from prl.baselines.analysis.core.nn_inspector import Inspector
from prl.baselines.analysis.core.stats import PlayerStats
from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser

def inspection(model_ckpt_abs_path):

    unzipped_dir = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    filenames = glob.glob(unzipped_dir.__str__() + '/**/*.txt', recursive=True)

    parser = HSmithyParser()
    pname = Path(model_ckpt_abs_path).parent.stem
    hidden_dims = [256] if '256' in pname else [512]
    inspector = Inspector(baseline=None, env_wrapper_cls=AugmentObservationWrapper)
    baseline = BaselineAgent(model_ckpt_abs_path,  # MajorityBaseline
                             flatten_input=False,
                             model_hidden_dims=hidden_dims)
    for filename in filenames:
        t0 = time.time()
        parsed_hands = list(parser.parse_file(filename))
        print(f'Parsing file {filename} took {time.time() - t0} seconds.')
        num_parsed_hands = len(parsed_hands)
        print(f'num_parsed_hands = {num_parsed_hands}')
        for ihand, hand in enumerate(parsed_hands):
            print(f'Inspecting model on hand {ihand} / {num_parsed_hands}')
            inspector.inspect_episode(hand, pname=pname)
        with open(f'model_inspection_{pname}.txt', 'a+') as f:
            for stat in inspector.player_stats:
                f.write(json.dumps(stat.to_dict()))

    # return f"Success. Wrote stats to {f'stats_{pname}.txt'}"


if __name__ == "__main__":
    model_ckpt_abs_path = "/home/sascha/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_selected_players/with_folds_div_1/with_folds/ckpt_dir/ilaviiitech_[512]_1e-06/ckpt.pt"

