from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def make_annotations():
    labels = [
        'A',
        'K',
        'Q',
        'J',
        'T',
        '9',
        '8',
        '7',
        '6',
        '5',
        '4',
        '3',
        '2'
    ]
    annot = np.zeros((13, 13), dtype=object)
    for row, c1 in enumerate(labels):
        for col, c2 in enumerate(labels):
            suited = 'o' if row > col else 's'
            if row == col:
                suited = ''
            if suited == 'o':
                ann = c2 + c1 + suited
            else:
                ann = c1 + c2 + suited
            annot[col][row] = ann
    print(annot)


def plot_ranges(positions: Dict[str, np.ndarray]):
    labels = ["AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33",
              "22"]
    card_labels = [['AA', 'AKo', 'AQo', 'AJo', 'ATo', 'A9o', 'A8o', 'A7o', 'A6o', 'A5o',
                    'A4o', 'A3o',
                    'A2o'],
                   ['AKs', 'KK', 'KQo', 'KJo', 'KTo', 'K9o', 'K8o', 'K7o', 'K6o', 'K5o',
                    'K4o', 'K3o',
                    'K2o'],
                   ['AQs', 'KQs', 'QQ', 'QJo', 'QTo', 'Q9o', 'Q8o', 'Q7o', 'Q6o', 'Q5o',
                    'Q4o', 'Q3o',
                    'Q2o'],
                   ['AJs', 'KJs', 'QJs', 'JJ', 'JTo', 'J9o', 'J8o', 'J7o', 'J6o', 'J5o',
                    'J4o', 'J3o',
                    'J2o'],
                   ['ATs', 'KTs', 'QTs', 'JTs', 'TT', 'T9o', 'T8o', 'T7o', 'T6o', 'T5o',
                    'T4o', 'T3o',
                    'T2o'],
                   ['A9s', 'K9s', 'Q9s', 'J9s', 'T9s', '99', '98o', '97o', '96o', '95o',
                    '94o', '93o',
                    '92o'],
                   ['A8s', 'K8s', 'Q8s', 'J8s', 'T8s', '98s', '88', '87o', '86o', '85o',
                    '84o', '83o',
                    '82o'],
                   ['A7s', 'K7s', 'Q7s', 'J7s', 'T7s', '97s', '87s', '77', '76o', '75o',
                    '74o', '73o',
                    '72o'],
                   ['A6s', 'K6s', 'Q6s', 'J6s', 'T6s', '96s', '86s', '76s', '66', '65o',
                    '64o', '63o',
                    '62o'],
                   ['A5s', 'K5s', 'Q5s', 'J5s', 'T5s', '95s', '85s', '75s', '65s', '55',
                    '54o', '53o',
                    '52o'],
                   ['A4s', 'K4s', 'Q4s', 'J4s', 'T4s', '94s', '84s', '74s', '64s', '54s',
                    '44', '43o',
                    '42o'],
                   ['A3s', 'K3s', 'Q3s', 'J3s', 'T3s', '93s', '83s', '73s', '63s', '53s',
                    '43s', '33',
                    '32o'],
                   ['A2s', 'K2s', 'Q2s', 'J2s', 'T2s', '92s', '82s', '72s', '62s', '52s',
                    '42s', '32s',
                    '22']]

    # Create the figure and subplots
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 20))

    # Flatten the subplots array for easier indexing
    axs = axs.flatten()

    # Iterate over the starting positions and corresponding numpy arrays
    for i, (position, data) in enumerate(positions.items()):
        # Create a heatmap of the data
        im = axs[i].imshow(data, cmap="Reds", vmin=0, vmax=1)

        # Set the title and axis labels
        axs[i].set_title(position)
        axs[i].set_xticks(np.arange(len(labels)))
        axs[i].set_yticks(np.arange(len(labels)))
        axs[i].set_xticklabels(labels)
        axs[i].set_yticklabels(labels)
        axs[i].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        for (j, k), _ in np.ndenumerate(data):
            label = card_labels[j][k]
            axs[i].text(j, k, label, ha='center', va='center')
            axs[i].text(j, k, label, ha='center', va='center')
    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Percentage")

    # Set the overall title and adjust the layout
    fig.suptitle("Hand Range Chart", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Show the plot
    plt.show()


if __name__ == '__main__':
    # Define the positions and labels for the x and y axis
    # positions = ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
    positions = {'UTG': np.random.rand(13, 13), 'MP': np.random.rand(13, 13),
                 'CO': np.random.rand(13, 13), 'BTN': np.random.rand(13, 13),
                 'SB': np.random.rand(13, 13), 'BB': np.random.rand(13, 13)}
    plot_ranges(positions)
