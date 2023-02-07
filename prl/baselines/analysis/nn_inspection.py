"""
player_name | Agression Factor | Tightness | acceptance level | Agression Factor NN | tightness NN
--------------------------------------------------------------------------------------------------
Agression Factor (AF): #raises / #calls
Tightness: % hands played (not folded immediately preflop)
"""
import glob
import time
from pathlib import Path

import pandas as pd
import seaborn
import torch.cuda
from matplotlib import pyplot as plt
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.base import ActionSpace

from prl.baselines.agents.tianshou_agents import BaselineAgent
from prl.baselines.analysis.core.nn_inspector import Inspector
from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser


def plot_heatmap(label_logits: dict, label_counts: dict) -> pd.DataFrame:
    detached = {}
    for label, logits in label_logits.items():
        normalize = label_counts[label]
        detached[label.value] = logits.detach().numpy()[0] / normalize
    # idx = cols = [i for i in range(len(ActionSpace))]
    df = pd.DataFrame(detached).T  # do we need , index=idx, columns=cols?
    plt.figure(figsize=(12, 7))
    seaborn.heatmap(df, annot=True)
# todo make path customizable
    plt.savefig('results/output.png')
    plt.show()

    return df


def inspection():
    model_ckpt_abs_path = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_all_players/with_folds_2NL_all_players/ckpt_dir_[512]_1e-06/ckpt.pt"
    # unzipped_dir = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/2.5NL/unzipped"
    filenames = glob.glob(unzipped_dir.__str__() + '/**/*.txt', recursive=True)

    parser = HSmithyParser()
    pname = Path(model_ckpt_abs_path).parent.stem
    hidden_dims = [256] if '256' in pname else [512]

    baseline = BaselineAgent(model_ckpt_abs_path,  # MajorityBaseline
                             device="cuda" if torch.cuda.is_available() else "cpu",
                             flatten_input=False,
                             model_hidden_dims=hidden_dims)
    inspector = Inspector(baseline=baseline, env_wrapper_cls=AugmentObservationWrapper)
    for filename in filenames[:500]:
        t0 = time.time()
        parsed_hands = list(parser.parse_file(filename))
        print(f'Parsing file {filename} took {time.time() - t0} seconds.')
        num_parsed_hands = len(parsed_hands)
        print(f'num_parsed_hands = {num_parsed_hands}')
        for ihand, hand in enumerate(parsed_hands):
            print(f'Inspecting model on hand {ihand} / {num_parsed_hands}')
            inspector.inspect_episode(hand, pname=pname)
    # todo add total counts of correct/wrong predictions
    # todo make this parallelizable for multiple networks
    # plots logits against true labels and saves csv with result to disk
    df = plot_heatmap(label_logits=inspector.false,
                      label_counts=inspector.label_counts_false)
    df.to_csv('./results/wrong.csv')
    print(df)
    df = plot_heatmap(label_logits=inspector.true,
                      label_counts=inspector.label_counts_true)
    df.to_csv('./results/true.csv')
    print(df)

    # todo write back model inspection
    # with open(f'model_inspection_{pname}.txt', 'a+') as f:
    #     for stat in inspector.player_stats:
    #         f.write(json.dumps(stat.to_dict()))

    # return f"Success. Wrote stats to {f'stats_{pname}.txt'}"


if __name__ == "__main__":
    # model_ckpt_abs_path = "/home/sascha/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_selected_players/with_folds_div_1/with_folds/ckpt_dir/ilaviiitech_[512]_1e-06/ckpt.pt"
    # inspection(model_ckpt_abs_path)
    inspection()