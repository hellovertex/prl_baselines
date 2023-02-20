"""
player_name | Agression Factor | Tightness | acceptance level | Agression Factor NN | tightness NN
--------------------------------------------------------------------------------------------------
Agression Factor (AF): #raises / #calls
Tightness: % hands played (not folded immediately preflop)
"""
import os
from pathlib import Path
from typing import Dict

import matplotlib
import pandas as pd
import seaborn as sns
import torch.cuda
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from prl.environment.Wrappers.base import ActionSpace

from prl.baselines.agents.tianshou_agents import BaselineAgent, MajorityBaseline
from prl.baselines.supervised_learning.v2.inspectorv2 import InspectorV2Vectorized

rows = ["Fold",
        "Check/Call",
        "RaiseThirdPot",
        "RaiseTwoThirdsPot",
        "RaisePot",
        "Raise2xPot",
        "Raise3xPot",
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
    rows = ["Fold",
            "Check/Call",
            "RaiseThirdPot",
            "RaiseTwoThirdsPot",
            "RaisePot",
            "Raise2xPot"]
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
    # df.columns = [f'Predicted {ActionSpace(i)}' for i in range(len(ActionSpace))] + ['Sum']
    df.columns = [f'Predicted {ActionSpace(i)}' for i in range(6)] + ['Sum']
    df.index = ["Fold",
                "Check/Call",
                "RaiseThirdPot",
                "RaiseTwoThirdsPot",
                "RaisePot",
                "Raise2xPot",
                "Raise3xPot",
                "RaiseALLIN"]
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
        # p = probas.reshape(num_samples, len(ActionSpace))
        p = probas.reshape(num_samples, 6)
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
    inspector = InspectorV2Vectorized(baseline=baseline)
    inspector.inspect_from_file(filename=filename, verbose=True)
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
    model_ckpt_abs_path = "/home/hellovertex/Documents/github.com/prl_baselines/data/ckpt.pt"
    debug = True
    csv_dir = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/v2/preflop_all_players/data.csv.bz2"
    path_out = './results/top100_preflop_no_fold'
    run(filename=csv_dir,
        model_ckpt_abs_path=model_ckpt_abs_path,
        path_out=path_out)


if __name__ == "__main__":
    start()
