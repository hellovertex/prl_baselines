import glob
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt

json_dir_test_accs = '/home/sascha/Documents/github.com/prl_baselines/data/accuracies' \
                     '/test_accuracies_with_fold'
json_dir_train_accs = '/home/sascha/Documents/github.com/prl_baselines/data/accuracies/train_accuracies_with_fold'


def read_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    steps = []
    accs = []
    for d in data:
        steps.append(d[1])
        accs.append(d[2])
    return steps, accs


def plot_acc(steps, accs, label, ax):
    ax.plot(steps, accs, label=label)


pseudonyms = {
    'zMukeha': 'Lindsay',
    'SerAlGog': 'Frank',
    'LuckyJO777': 'Ryan',
    'SoLongRain': 'Kayla',
    'ishuha': 'Joe',
}


def plot_all(dirpath1, dirpath2):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), dpi=200)
    # fig = plt.figure(figsize=(8, 6), dpi=300)
    axes = [ax1, ax2]
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Accuracy')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Accuracy')
    plt.set_cmap('viridis')
    # plt.grid()
    plt.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})
    ax1.set_title('Train accuracies')  # ) from Game Logs with randomized Folds')
    ax2.set_title('Test accuracies ')  # from Game Logs with randomized Folds')
    labels = []
    for i, dirpath in enumerate([dirpath1, dirpath2]):
        axes[i].grid()
        # t = 'Test' if i ==0 else 'Train'
        # axes[i].set_title(f'{t} accuracies from Game Logs with randomized Folds')
        for filename in os.listdir(dirpath):
            if filename.endswith('.json'):
                match = re.search(r'\[(\d+)\]', filename)
                if match:
                    hidden_dim = str(int(match.group(1)))
                filepath = os.path.join(dirpath, filename)
                steps, accs = read_json(filepath)
                name = Path(filename).stem.split('_')[0]
                if not name in pseudonyms:
                    continue
                label = pseudonyms[name] + f'_hidden_dim={hidden_dim}'  #
                labels.append(label)
                # remove the
                # extension from
                # the
                # filename

                plot_acc(steps, accs, label, axes[i])
    plt.legend(loc='lower right',
               labels=labels,
               bbox_to_anchor=(1.0, 0.0),
               fontsize=10)
    plt.show()


def plot_single(dirpath):
    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), dpi=200)
    fig = plt.figure(figsize=(8, 6), dpi=300)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.set_cmap('viridis')
    plt.grid()
    plt.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})
    plt.title('F1-scores after training from Game Logs')  # ) from Game Logs with
    # randomized Folds')
    labels = []
    i = 0
    for filename in os.listdir(dirpath):
        if filename.endswith('.json'):
            match = re.search(r'\[(\d+)\]', filename)
            if match:
                hidden_dim = str(int(match.group(1)))
            filepath = os.path.join(dirpath, filename)
            steps, accs = read_json(filepath)
            steps = [step * 10 for step in steps]
            label = list(pseudonyms.values())[i] + f'_hidden_dim={512}'  #
            labels.append(label)
            i += 1
            # remove the
            # extension from
            # the
            # filename

            plot_acc(steps, accs, label, plt)
    plt.legend(loc='lower right',
               labels=labels,
               bbox_to_anchor=(1.0, 0.0),
               fontsize=10)
    plt.show()


if __name__ == '__main__':
    dir_f1_scores = '/home/sascha/Documents/github.com/prl_baselines/data/accuracies' \
                    '/f1_testscores_without_fold'
    plot_single(dir_f1_scores)
    # plot_all(json_dir_train_accs, json_dir_test_accs)
