"""
player_name | Agression Factor | Tightness | acceptance level | Agression Factor NN | tightness NN
--------------------------------------------------------------------------------------------------
Agression Factor (AF): #raises / #calls
Tightness: % hands played (not folded immediately preflop)
"""
import glob
import multiprocessing
import os
import time
from functools import partial
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch.cuda
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.base import ActionSpace

from prl.baselines.agents.tianshou_agents import BaselineAgent, MajorityBaseline
from prl.baselines.analysis.core.nn_inspector import Inspector
from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser


def plot_heatmap(label_logits: dict,
                 label_counts: dict,
                 path_out_png: str) -> pd.DataFrame:
    detached = {}
    l = []
    for label, logits in label_logits.items():
        normalize = label_counts[label]
        l.append(normalize)
        detached[label.value] = logits.detach().numpy()[0] / normalize
        # detached[label.value] = np.hstack([detached[label.value],[label_counts[label]]])
    # detached["Sum"] = [v for _, v in label_counts.items()]
    # idx = cols = [i for i in range(len(ActionSpace))]
    rows = ["Fold", "Check/Call", "Raise3BB", "Raise6BB", "Raise10BB", "Raise20BB",
            "Raise50BB", "RaiseALLIN"]

    df = pd.DataFrame(detached).T  # do we need , index=idx, columns=cols?

    # plot the heatmap with annotations
    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())
    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   figsize=(16, 6),
                                   gridspec_kw={'width_ratios': [3, 1]})

    # sns.heatmap(df, annot=True, ax=ax1, cmap='coolwarm')
    cm = sns.heatmap(df, annot=True, ax=ax1, cmap='viridis')
    im = cm.collections[0]
    rgba = im.to_rgba(.1)
    # rgb = tuple(map(int, 255 * rgba[:3]))
    hex_value = matplotlib.colors.rgb2hex(rgba, keep_alpha=True)
    what = 'mis-' if 'false' in path_out_png else 'correctly '
    ax1.set_title(f"Average probabilities of the network on {what}predicting label")
    ax2.bar(df.columns,
            l,
            color=[hex_value for _ in range(8)])
    ax2.set_xlabel("Which Action")
    ax2.set_ylabel("Number of times label was present")
    ax2.set_title("Number of actions taken")
    df['Sum'] = l
    df.columns = [f'Predicted {ActionSpace(i)}' for i in range(len(ActionSpace))] + ['Sum']
    df.index = rows
    # adjust the subplots to occupy equal space
    fig.tight_layout()
    # fpath = os.path.join(os.getcwd(), 'result')
    # fpath = os.path.join(fpath, Path(path_out_png).stem)

    if not os.path.exists(Path(path_out_png).parent):
        os.makedirs(Path(path_out_png).parent)
    plt.savefig(os.path.join(Path(path_out_png).parent,
                             Path(path_out_png).stem),
                bbox_inches='tight')
    plt.show()

    return df


def make_results(inspector, path_out):
    # WRONG PREDICTIONS
    df = plot_heatmap(label_logits=inspector.false,
                      label_counts=inspector.label_counts_false,
                      path_out_png=f'{path_out}/false.png')
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    df.to_csv(f'{path_out}/false_probabs.csv')
    df = pd.DataFrame(inspector.label_counts_false,
                      index=[k.name for k in list(inspector.label_counts_false.keys())])
    df.to_csv(f'{path_out}/false_labels.csv')
    print(df)
    # TRUE PREDICTIONS
    df = plot_heatmap(label_logits=inspector.true,
                      label_counts=inspector.label_counts_true,
                      path_out_png=f'{path_out}/correct.png')
    df.to_csv(f'{path_out}/correct_probas.csv')
    df = pd.DataFrame(inspector.label_counts_true,
                      index=[k.name for k in list(inspector.label_counts_true.keys())])
    df.to_csv(f'{path_out}/true_labels.csv')
    print(df)

    # todo write back model inspection
    # with open(f'model_inspection_{pname}.txt', 'a+') as f:
    #     for stat in inspector.player_stats:
    #         f.write(json.dumps(stat.to_dict()))

    # return f"Success. Wrote stats to {f'stats_{pname}.txt'}"


def inspection(filename,
               model_ckpt_abs_path,
               path_out,
               max_files=5):
    parser = HSmithyParser()
    pname = Path(model_ckpt_abs_path).parent.stem

    if type(model_ckpt_abs_path) == str or type(model_ckpt_abs_path) == Path:
        hidden_dims = [256] if '256' in pname else [512]
        baseline = BaselineAgent(model_ckpt_abs_path,  # MajorityBaseline
                                 device="cpu",  # "cuda" if torch.cuda.is_available() else "cpu",
                                 flatten_input=False,
                                 model_hidden_dims=hidden_dims)
    else:  # list of checkpoints
        hidden_dims = [[256] if '[256]' in pname else [512] for pname in model_ckpt_abs_path]
        baseline = MajorityBaseline(model_ckpt_paths=model_ckpt_abs_path,  # MajorityBaseline
                                    model_hidden_dims=hidden_dims,
                                    device="cpu",  # "cuda" if torch.cuda.is_available() else "cpu",
                                    flatten_input=False)
    inspector = Inspector(baseline=baseline, env_wrapper_cls=AugmentObservationWrapper)
    # for filename in filenames[:max_files]:
    t0 = time.time()
    parsed_hands = list(parser.parse_file(filename))
    print(f'Parsing file {filename} took {time.time() - t0} seconds.')
    num_parsed_hands = len(parsed_hands)
    print(f'num_parsed_hands = {num_parsed_hands}')
    for ihand, hand in enumerate(parsed_hands[:10000]):
        print(f'Inspecting model on hand {ihand} / {num_parsed_hands}')
        inspector.inspect_episode(hand, pname=pname)
    # plots logits against true labels and saves csv with result to disk
    make_results(inspector, path_out + Path(filename).stem)
    return f"Succes. Wrote file to {path_out + '/' + Path(filename).stem}"


if __name__ == "__main__":
    # model_ckpt_abs_path = "/home/sascha/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_selected_players/with_folds_div_1/with_folds/ckpt_dir/ilaviiitech_[512]_1e-06/ckpt.pt"
    # model_ckpt_abs_path = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_all_players/with_folds_2NL_all_players/ckpt_dir_[512]_1e-06/ckpt.pt"
    model_ckpt_abs_path = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_all_players/randomized_folds_with_downsamplingv1_0_25NL_all_players/ckpt_dir_[512]_1e-06/ckpt.pt"
    unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data"
    # unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/2.5NL/unzipped"
    # unzipped_dir = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    path_out = './results/dprime_rand_folds'
    max_files = 1000
    # unzipped_dir = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    filenames = glob.glob(unzipped_dir.__str__() + '/**/*.txt', recursive=True)

    start = time.time()
    p = multiprocessing.Pool()
    t0 = time.time()

    inspect_fn = partial(inspection, model_ckpt_abs_path=model_ckpt_abs_path,
                         path_out=path_out,
                         max_files=max_files)
    for x in p.imap_unordered(inspect_fn, filenames):
        print(x + f'. Took {time.time() - t0} seconds')
    print(f'Finished job after {time.time() - start} seconds.')

    p.close()

# if __name__ == "__main__":
#     # model_ckpt_abs_path = "/home/sascha/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_selected_players/with_folds_div_1/with_folds/ckpt_dir/ilaviiitech_[512]_1e-06/ckpt.pt"
#     model_ckpt_abs_path = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_all_players/with_folds_2NL_all_players/ckpt_dir_[512]_1e-06/ckpt.pt"
#     unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/2.5NL/unzipped"
#
#     # 1. load checkpoints subdirs
#     # 2.
#
#     # unzipped_dir = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
#     path_out = './results/2NL'
#     max_files = 20
#     filenames = glob.glob(unzipped_dir.__str__() + '/**/*.txt', recursive=True)
#
#     start = time.time()
#     p = multiprocessing.Pool()
#     t0 = time.time()
#     inspect_fn = partial(inspection,
#                          model_ckpt_abs_path=model_ckpt_abs_path,
#                          path_out=path_out,
#                          max_files=max_files)
#     for x in p.imap_unordered(inspect_fn, filenames):
#         print(x + f'. Took {time.time() - t0} seconds')
#     print(f'Finished job after {time.time() - start} seconds.')
#
#     p.close()
#     inspection(model_ckpt_abs_path=model_ckpt_abs_path,
#                unzipped_dir=unzipped_dir,
#                path_out=path_out,
#                max_files=5)
