"""
player_name | Agression Factor | Tightness | acceptance level | Agression Factor NN | tightness NN
--------------------------------------------------------------------------------------------------
Agression Factor (AF): #raises / #calls
Tightness: % hands played (not folded immediately preflop)
"""
import glob
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch.cuda
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from prl.environment.Wrappers.augment import AugmentObservationWrapper

from prl.baselines.agents.tianshou_agents import BaselineAgent
from prl.baselines.analysis.core.nn_inspector import Inspector
from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser


def plot_heatmap(label_logits: dict,
                 label_counts: dict,
                 path_out_png: str) -> pd.DataFrame:
    detached = {}
    for label, logits in label_logits.items():
        normalize = label_counts[label]
        detached[label.value] = logits.detach().numpy()[0] / normalize
        # detached[label.value] = np.hstack([detached[label.value],[label_counts[label]]])
    # detached["Sum"] = [v for _, v in label_counts.items()]
    # idx = cols = [i for i in range(len(ActionSpace))]
    cols = ["Fold", "Check/Call", "Raise3BB", "Raise6BB", "Raise10BB", "Raise20BB",
            "Raise50BB", "RaiseALLIN", "Sum"]
    df = pd.DataFrame(detached).T  # do we need , index=idx, columns=cols?
    # plot the heatmap with annotations
    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())
    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   figsize=(16, 6),
                                   gridspec_kw={'width_ratios': [3, 1]})

    # sns.heatmap(df, annot=True, ax=ax1, cmap='coolwarm')
    cm = sns.heatmap(df, annot=True, ax=ax1, cmap='Purples')
    im = cm.collections[0]
    rgba = im.to_rgba(.2)
    rgb = tuple(map(int, 255 * rgba[:3]))
    hex_value = matplotlib.colors.rgb2hex(rgba, keep_alpha=True)
    ax1.set_title("Heatmap")
    # plot the magnitude of each number in l
    l = [114, 21, 16, 9, 5, 5, 1, 8]
    # plot the magnitude of each number in l in the second subplot
    ax2.bar(df.columns, l, color=[hex_value for _ in range(8)])
    ax2.set_xlabel("Keys")
    ax2.set_ylabel("Values")
    ax2.set_title("Magnitude of values in list 'l'")

    # adjust the subplots to occupy equal space
    # fig.tight_layout()
    plt.show()

    return df


def make_results(inspector, path_out):
    # WRONG PREDICTIONS
    df = plot_heatmap(label_logits=inspector.false,
                      label_counts=inspector.label_counts_false,
                      path_out_png=f'{path_out}/false.png')
    df.to_csv(f'./results/{path_out}/false_probabs.csv')
    df = pd.DataFrame(inspector.label_counts_false)
    df.to_csv(f'./results/{path_out}/false_labels.csv')
    print(df)
    # TRUE PREDICTIONS
    df = plot_heatmap(label_logits=inspector.true,
                      label_counts=inspector.label_counts_true,
                      path_out_png=f'{path_out}/correct.png')
    df.to_csv(f'./results/{path_out}/correct_probas.csv')
    df = pd.DataFrame(inspector.label_counts_true)
    df.to_csv(f'./results/{path_out}/true_labels.csv')
    print(df)

    # todo write back model inspection
    # with open(f'model_inspection_{pname}.txt', 'a+') as f:
    #     for stat in inspector.player_stats:
    #         f.write(json.dumps(stat.to_dict()))

    # return f"Success. Wrote stats to {f'stats_{pname}.txt'}"


def inspection(model_ckpt_abs_path,
               unzipped_dir,
               path_out):
    # unzipped_dir = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    filenames = glob.glob(unzipped_dir.__str__() + '/**/*.txt', recursive=True)

    parser = HSmithyParser()
    pname = Path(model_ckpt_abs_path).parent.stem
    hidden_dims = [256] if '256' in pname else [512]

    baseline = BaselineAgent(model_ckpt_abs_path,  # MajorityBaseline
                             device="cuda" if torch.cuda.is_available() else "cpu",
                             flatten_input=False,
                             model_hidden_dims=hidden_dims)
    inspector = Inspector(baseline=baseline, env_wrapper_cls=AugmentObservationWrapper)
    for filename in filenames[:2]:
        t0 = time.time()
        parsed_hands = list(parser.parse_file(filename))
        print(f'Parsing file {filename} took {time.time() - t0} seconds.')
        num_parsed_hands = len(parsed_hands)
        print(f'num_parsed_hands = {num_parsed_hands}')
        for ihand, hand in enumerate(parsed_hands):
            print(f'Inspecting model on hand {ihand} / {num_parsed_hands}')
            inspector.inspect_episode(hand, pname=pname)
    # plots logits against true labels and saves csv with result to disk
    make_results(inspector, path_out)


if __name__ == "__main__":
    # model_ckpt_abs_path = "/home/sascha/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_selected_players/with_folds_div_1/with_folds/ckpt_dir/ilaviiitech_[512]_1e-06/ckpt.pt"
    model_ckpt_abs_path = "/home/sascha/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_selected_players/with_folds_div_1/with_folds/ckpt_dir/ilaviiitech_[512]_1e-06/ckpt.pt"
    # unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/2.5NL/unzipped"
    unzipped_dir = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    path_out = './results/2NL'
    # todo make this parallelizable for multiple networks
    inspection(model_ckpt_abs_path=model_ckpt_abs_path,
               unzipped_dir=unzipped_dir,
               path_out=path_out)
