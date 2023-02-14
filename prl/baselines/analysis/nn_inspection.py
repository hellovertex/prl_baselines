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
from typing import Dict, Union, List

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch.cuda
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.base import ActionSpace
from prl.environment.Wrappers.utils import init_wrapped_env

from prl.baselines.agents.tianshou_agents import BaselineAgent, MajorityBaseline
from prl.baselines.analysis.core.nn_inspector import Inspector
from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser
from prl.baselines.supervised_learning.v2.config import top_20
from prl.baselines.supervised_learning.v2.fast_hsmithy_parser import ParseHsmithyTextToPokerEpisode
from prl.baselines.supervised_learning.v2.inspectorv2 import InspectorV2

rows = ["Fold",
        "Check/Call",
        "Raise3BB",
        "Raise6BB",
        "Raise10BB",
        "Raise20BB",
        "Raise50BB",
        "RaiseALLIN"]


def plot_heatmap(input_dict: Dict[ActionSpace, torch.Tensor],
                 action_freqs,
                 title,
                 path_out_png):
    detached = {}
    for action, probas in input_dict.items():
        if probas is not torch.nan:
            detached[action] = probas.detach().numpy()[0]
        else:
            detached[action] = probas
    df = pd.DataFrame(detached, index=rows).T  # do we need , index=idx, columns=cols?
    # todo: fix this
    action_freqs = torch.sum(torch.row_stack(action_freqs), dim=1)
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
    ax1.set_title(title)
    ax2.bar(df.index,
            action_freqs,
            color=[hex_value for _ in range(8)])
    ax2.set_xlabel("Which Action")
    ax2.set_ylabel("Number of actions")
    ax2.set_title("Number of actions taken")
    df['Sum'] = action_freqs
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


def flush_to_disk(df,
                  label_counts,
                  path_out):
    if not os.path.exists(Path(path_out).parent):
        os.makedirs(Path(path_out).parent)
    df.to_csv(path_out)


def compute(label_logits, label_counts) \
        -> Dict[str, Dict[ActionSpace, torch.Tensor]]:
    means = {}
    maas = {}
    mins = {}
    maxs = {}
    percentile_10 = {}
    percentile_25 = {}
    percentile_50 = {}
    percentile_75 = {}
    percentile_90 = {}

    action_freqs = []
    # from dictionary of action to stacked tensors with list of action probability vectors
    # we want to compute statistics min, max, 25, 50, 75 percentile
    # mean, std
    for label, probas in label_logits.items():
        normalize = label_counts[label]
        action_freqs.append(normalize)
        if probas is None:
            means[label.value] = torch.nan
            maas[label.value] = torch.nan
            mins[label.value] = torch.nan
            maxs[label.value] = torch.nan
            percentile_10[label.value] = torch.nan
            percentile_25[label.value] = torch.nan
            percentile_50[label.value] = torch.nan
            percentile_75[label.value] = torch.nan
            percentile_90[label.value] = torch.nan
            continue
        assert sum(normalize) == probas.shape[0]

        # means[label.value] = logits.detach().numpy()[0] / normalize
        m = torch.mean(probas, dim=0)
        # mean action probabilities per action label
        means[label.value] = m
        ma = probas - m
        ma_abs = ma.abs()
        # mean absolute deviations per action label
        # mean of the absolute deviation of each feature-probability from its mean probability
        maas[label.value] = torch.mean(ma_abs, dim=0)
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
               'abs_std': maas,
               'mins': mins,
               'maxs': maxs,
               'percentile_10': percentile_10,
               'percentile_25': percentile_25,
               'percentile_50': percentile_50,
               'percentile_75': percentile_75,
               'percentile_90': percentile_90,
               'action_freqs': action_freqs  # todo re-name
               }

    return results


def collect(filename,
            model_ckpt_abs_path):
    # parser = HSmithyParser()
    parser = ParseHsmithyTextToPokerEpisode()

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
    # inspector = Inspector(baseline=baseline, env_wrapper_cls=AugmentObservationWrapper)
    env = init_wrapped_env(AugmentObservationWrapper,
                           [5000 for _ in range(6)],
                           blinds=(25, 50),
                           multiply_by=1, )
    inspector = InspectorV2(env=env, baseline=baseline)
    # for filename in filenames[:max_files]:
    t0 = time.time()
    parsed_hands = list(parser.parse_file(filename))
    print(f'Parsing file {filename} took {time.time() - t0} seconds.')
    num_parsed_hands = len(parsed_hands)
    print(f'num_parsed_hands = {num_parsed_hands}')
    for ihand, hand in enumerate(parsed_hands[:2000]):
        print(f'Inspecting model on hand {ihand} / {num_parsed_hands}')
        inspector.inspect_episode(hand,
                                  drop_folds=False,
                                  randomize_fold_cards=True,
                                  selected_players=top_20,
                                  verbose=True)
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
    path_out_png = path_out + '/' + Path(filename).stem + '/plots'
    df_wrong_means = plot_heatmap(results_wrong['means'],
                                  action_freqs=results_wrong['action_freqs'],
                                  title=f"Average probabilities of the network on mis-predicting label",
                                  path_out_png=path_out_png + '/means_false.png')
    df_wrong_abs_std = plot_heatmap(results_wrong['abs_std'],
                                    action_freqs=results_wrong['action_freqs'],
                                    title=f"Absolute mean-deviation of the network probabilities on mis-predicting label",
                                    path_out_png=path_out_png + '/abs_std_false.png')
    # df_wrong_percentile_10 = plot_heatmap(results_wrong['percentile_10'],
    #                                       action_freqs=results_wrong['action_freqs'],
    #                                       title=f"10% Percentile of the networks action probabilities on mis-predicting label",
    #                                       path_out_png=path_out_png + '/percentile_10_false.png')
    # df_wrong_percentile_25 = plot_heatmap(results_wrong['percentile_25'],
    #                                       action_freqs=results_wrong['action_freqs'],
    #                                       title=f"25% Percentile of the networks action probabilities on mis-predicting label",
    #                                       path_out_png=path_out_png + '/percentile_25_false.png')
    # df_wrong_percentile_50 = plot_heatmap(results_wrong['percentile_50'],
    #                                       action_freqs=results_wrong['action_freqs'],
    #                                       title=f"50% Percentile of the networks action probabilities on mis-predicting label",
    #                                       path_out_png=path_out_png + '/percentile_50_false.png')
    # df_wrong_percentile_75 = plot_heatmap(results_wrong['percentile_75'],
    #                                       action_freqs=results_wrong['action_freqs'],
    #                                       title=f"75% Percentile of the networks action probabilities on mis-predicting label",
    #                                       path_out_png=path_out_png + '/percentile_75_false.png')
    # df_wrong_percentile_90 = plot_heatmap(results_wrong['percentile_90'],
    #                                       action_freqs=results_wrong['action_freqs'],
    #                                       title=f"90% Percentile of the networks action probabilities on mis-predicting label",
    #                                       path_out_png=path_out_png + '/percentile_90_false.png')

    df_correct_means = plot_heatmap(results_correct['means'],
                                    action_freqs=results_correct['action_freqs'],
                                    title=f"Average probabilities of the network on correctly predicting label",
                                    path_out_png=path_out_png + '/means_correct.png')
    df_correct_abs_std = plot_heatmap(results_correct['abs_std'],
                                      action_freqs=results_correct['action_freqs'],
                                      title=f"Absolute mean-deviation of the network probabilities on correctly predicting label",
                                      path_out_png=path_out_png + '/abs_std_correct.png')
    # df_correct_percentile_10 = plot_heatmap(results_correct['percentile_10'],
    #                                         action_freqs=results_correct['action_freqs'],
    #                                         title=f"10% Percentile of the networks action probabilities on mis-correctly icting label",
    #                                         path_out_png=path_out_png + '/percentile_10_correct.png')
    # df_correct_percentile_25 = plot_heatmap(results_correct['percentile_25'],
    #                                         action_freqs=results_correct['action_freqs'],
    #                                         title=f"25% Percentile of the networks action probabilities on mis-correctly icting label",
    #                                         path_out_png=path_out_png + '/percentile_25_correct.png')
    # df_correct_percentile_50 = plot_heatmap(results_correct['percentile_50'],
    #                                         action_freqs=results_correct['action_freqs'],
    #                                         title=f"50% Percentile of the networks action probabilities on mis-correctly icting label",
    #                                         path_out_png=path_out_png + '/percentile_50_correct.png')
    # df_correct_percentile_75 = plot_heatmap(results_correct['percentile_75'],
    #                                         action_freqs=results_correct['action_freqs'],
    #                                         title=f"75% Percentile of the networks action probabilities on mis-correctly icting label",
    #                                         path_out_png=path_out_png + '/percentile_75_correct.png')
    # df_correct_percentile_90 = plot_heatmap(results_correct['percentile_90'],
    #                                         action_freqs=results_correct['action_freqs'],
    #                                         title=f"90% Percentile of the networks action probabilities on mis-correctly icting label",
    #                                         path_out_png=path_out_png + '/percentile_90_correct.png')

    # write files to disk using df.to_csv()
    path_out_csv = path_out + '/' + Path(filename).stem + '/csv_files'
    # mis-predicted labels
    flush_to_disk(df_wrong_means, inspector.label_counts_false, path_out_csv + '/means_wrong.csv')
    flush_to_disk(df_wrong_abs_std, inspector.label_counts_false, path_out_csv + '/abs_std_wrong.csv')
    # flush_to_disk(df_wrong_percentile_10, inspector.label_counts_false, path_out_csv + '/percentile_10_wrong.csv')
    # flush_to_disk(df_wrong_percentile_25, inspector.label_counts_false, path_out_csv + '/percentile_25_wrong.csv')
    # flush_to_disk(df_wrong_percentile_50, inspector.label_counts_false, path_out_csv + '/percentile_50_wrong.csv')
    # flush_to_disk(df_wrong_percentile_75, inspector.label_counts_false, path_out_csv + '/percentile_75_wrong.csv')
    # flush_to_disk(df_wrong_percentile_90, inspector.label_counts_false, path_out_csv + '/percentile_90_wrong.csv')

    # correctly predicted labels
    flush_to_disk(df_correct_means, inspector.label_counts_true, path_out_csv + '/means_correct.csv')
    flush_to_disk(df_correct_abs_std, inspector.label_counts_true, path_out_csv + '/abs_std_correct.csv')
    # flush_to_disk(df_correct_percentile_10, inspector.label_counts_true, path_out_csv + '/percentile_10_correct.csv')
    # flush_to_disk(df_correct_percentile_25, inspector.label_counts_true, path_out_csv + '/percentile_25_correct.csv')
    # flush_to_disk(df_correct_percentile_50, inspector.label_counts_true, path_out_csv + '/percentile_50_correct.csv')
    # flush_to_disk(df_correct_percentile_75, inspector.label_counts_true, path_out_csv + '/percentile_75_correct.csv')
    # flush_to_disk(df_correct_percentile_90, inspector.label_counts_true, path_out_csv + '/percentile_90_correct.csv')
    print(f'Summary for file {filename}: \nPredictions (Correct, Wrong) per label: {inspector.summary_predictions}')
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
    model_ckpt_abs_path = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_all_players/all_games_and_folds_rand_cards_selected_players/ckpt_dir_[512]_1e-06/ckpt.pt"
    # ctd
    unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data"
    unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    # unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/2.5NL/unzipped"
    unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data_test"
    unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data_5"
    path_out = './results/selected_players_only_fold_random_cards'
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
