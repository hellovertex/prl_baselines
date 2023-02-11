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
from typing import Dict, Union

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


def plot_heatmap(input_dict,
                 path_out_png):
    means = input_dict['means']
    l = input_dict['l']
    rows = ["Fold", "Check/Call", "Raise3BB", "Raise6BB", "Raise10BB", "Raise20BB",
            "Raise50BB", "RaiseALLIN"]

    df = pd.DataFrame(means).T  # do we need , index=idx, columns=cols?

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


def flush(results, path_out, inspector):
    pass
    # df = results['means']
    # if not os.path.exists(path_out):
    #     os.makedirs(path_out)
    # df.to_csv(f'{path_out}/false_probabs.csv')
    # df = pd.DataFrame(inspector.label_counts_false,
    #                   index=[k.name for k in list(inspector.label_counts_false.keys())])
    # df.to_csv(f'{path_out}/false_labels.csv')
    # print(df)
    # # TRUE PREDICTIONS
    # df = plot_heatmap(results,
    #                   path_out_png=f'{path_out}/correct.png')
    # df.to_csv(f'{path_out}/correct_probas.csv')
    # df = pd.DataFrame(inspector.label_counts_true,
    #                   index=[k.name for k in list(inspector.label_counts_true.keys())])
    # df.to_csv(f'{path_out}/true_labels.csv')
    # print(df)


def compute(label_logits, label_counts) -> Dict[str, Union[torch.Tensor, Dict]]:
    means = {}
    maas = {}
    mins = {}
    maxs = {}
    percentile_10 = {}
    percentile_25 = {}
    percentile_50 = {}
    percentile_75 = {}
    percentile_90 = {}

    l = []
    # from dictionary of action to stacked tensors with list of action probability vectors
    # we want to compute statistics min, max, 25, 50, 75 percentile
    # mean, std
    for label, probas in label_logits.items():
        normalize = label_counts[label]
        l.append(normalize)
        if probas is None:
            continue
        assert sum(normalize) == probas.shape[0]

        # means[label.value] = logits.detach().numpy()[0] / normalize
        m = torch.mean(probas, dim=0)
        # mean action probabilities per action label
        means[label.value] = m
        ma = probas - m
        ma_abs = ma.abs()
        # mean absolute deviations per action label
        maas[label.value] = ma_abs
        # min, max
        mins[label.value] = torch.min(probas, dim=0)
        maxs[label.value] = torch.max(probas, dim=0)
        # percentiles
        num_samples = probas.shape[0]
        p = probas.reshape(num_samples, len(ActionSpace))
        p = p.sort(dim=0)[0]
        # x % of the data falls below the VALUE of the x-percentile so
        # percentile_50[ActionSpace.FOLD][ActionSpace.MIN_RAISE] = .8
        #  --> True action was fold, the predicted probability to call is below .8 in exactly 50% of the data
        percentile_10_index = int(0.10 * num_samples)
        percentile_10[label.value] = p[percentile_10_index, :]
        percentile_25_index = int(0.25 * num_samples)
        percentile_25[label.value] = p[percentile_25_index, :]
        percentile_50_index = int(0.50 * num_samples)
        percentile_50[label.value] = p[percentile_50_index, :]
        percentile_75_index = int(0.75 * num_samples)
        percentile_75[label.value] = p[percentile_75_index, :]
        percentile_90_index = int(0.90 * num_samples)
        percentile_90[label.value] = p[percentile_90_index, :]
        # detached[label.value] = np.hstack([detached[label.value],[label_counts[label]]])
    # detached["Sum"] = [v for _, v in label_counts.items()]
    # idx = cols = [i for i in range(len(ActionSpace))]
    results = {'means': means,
               'maas': maas,
               'mins': mins,
               'maxs': maxs,
               'percentile_10': percentile_10,
               'percentile_25': percentile_25,
               'percentile_50': percentile_50,
               'percentile_75': percentile_75,
               'percentile_90': percentile_90,
               'l': l  # todo rename
               }

    return results


def collect(filename,
            model_ckpt_abs_path):
    parser = HSmithyParser()

    if type(model_ckpt_abs_path) == str or type(model_ckpt_abs_path) == Path:
        pname = Path(model_ckpt_abs_path).parent.stem
        hidden_dims = [256] if '256' in pname else [512]
        baseline = BaselineAgent(model_ckpt_abs_path,  # MajorityBaseline
                                 device="cpu",  # "cuda" if torch.cuda.is_available() else "cpu",
                                 flatten_input=False,
                                 model_hidden_dims=hidden_dims)
    else:  # list of checkpoints
        pname = "MajorityVoting"
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
    for ihand, hand in enumerate(parsed_hands[:5000]):
        print(f'Inspecting model on hand {ihand} / {num_parsed_hands}')
        inspector.inspect_episode(hand, pname=pname)
    return inspector


def run(filename,
        model_ckpt_abs_path,
        path_out,
        max_files=5):
    # run neural network on .txt file dataset to compute labels and collect statistics
    inspector = collect(filename, model_ckpt_abs_path)

    # compute mean, abs-std, min, max, percentiles [10,25,50,75,90] of predictions
    results_wrong = compute(inspector.logits_when_wrong,
                            inspector.label_counts_false)
    results_correct = compute(inspector.logits_when_correct,
                              inspector.label_counts_true)
    # plot heatmaps
    plot_heatmap(results_wrong, path_out_png=path_out + '/wrong.png')
    plot_heatmap(results_correct, path_out_png=path_out + '/correct.png')

    # write files to disk using df.to_csv()
    flush(results_wrong, path_out, inspector)
    flush(results_correct, path_out, inspector)
    # plots logits against true labels and saves csv with result to disk
    return f"Succes. Wrote file to {path_out + '/' + Path(filename).stem}"


def start():
    # model_ckpt_abs_path = "/home/sascha/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_selected_players/with_folds_div_1/with_folds/ckpt_dir/ilaviiitech_[512]_1e-06/ckpt.pt"
    # model_ckpt_abs_path = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_all_players/with_folds_2NL_all_players/ckpt_dir_[512]_1e-06/ckpt.pt"

    # Multiple Baseline checkpoints --> Creates Majority Agent in inspect function
    debug = True
    model_ckpt_abs_path = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_selected_players/with_folds_rand_cards/ckpt_dir"
    player_dirs = [x[0] for x in
                   os.walk(model_ckpt_abs_path)][1:]
    player_dirs = [pdir for pdir in player_dirs if not Path(pdir).stem == 'ckpt']
    ckpts = [pdir + '/ckpt.pt' for pdir in player_dirs]
    model_ckpt_abs_path = ckpts

    # Single Baseline checkpoint
    # model_ckpt_abs_path = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_all_players/randomized_folds_with_downsamplingv1_0_25NL_all_players/ckpt_dir_[512]_1e-06/ckpt.pt"
    model_ckpt_abs_path = "/home/hellovertex/Documents/github.com/prl_baselines/data/no_folds_selected_players_70%/ckpt_dir_[512]_1e-06/ckpt.pt"
    model_ckpt_abs_path = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_all_players/no_folds_selected_players/ckpt_dir_[512]_1e-06/ckpt.pt"
    # ctd
    unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data"
    unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data_10"
    unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    # unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/2.5NL/unzipped"
    path_out = './results/selected_players_no_fold'
    max_files = 1000
    # unzipped_dir = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    filenames = glob.glob(unzipped_dir.__str__() + '/**/*.txt', recursive=True)

    run_fn = partial(run,
                     model_ckpt_abs_path=model_ckpt_abs_path,
                     path_out=path_out,
                     max_files=max_files)
    if debug:
        run_fn(filenames[0])
    else:
        start = time.time()
        p = multiprocessing.Pool()
        t0 = time.time()
        for x in p.imap_unordered(run_fn, filenames):
            print(x + f'. Took {time.time() - t0} seconds')
        print(f'Finished job after {time.time() - start} seconds.')
        p.close()


if __name__ == "__main__":
    start()

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
