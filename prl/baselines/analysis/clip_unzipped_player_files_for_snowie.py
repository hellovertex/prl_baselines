import os
import re
from functools import partial

import glob
import time
import multiprocessing
from pathlib import Path


def write_shortened_version_for_snowie_analysis(filename, n_hands, path_out):
    # read file and write back first n_hands
    with open(filename, 'r', encoding='utf-8') as f:  # pylint: disable=invalid-name,unspecified-encoding
        hand_database = f.read()
        hands_played = re.split(r'PokerStars Hand #', hand_database)[1:]
    if not os.path.exists(path_out):
        os.makedirs(os.path.abspath(path_out))
    with open(os.path.join(path_out, Path(filename).stem)+".txt", 'a+') as f:
        for e in hands_played[:n_hands]:
            f.write("PokerStars Hand #" + e)



if __name__ == '__main__':
    # load player dirs via multiprocessing and write back stripped down files for pokersnowie eval
    # 1k showdown hands

    #unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data"
    #unzipped_dir = "/home/hellovertex/Documents/github.com/hellovertex/prl_baselines/data/player_data"
    unzipped_dir = "/home/hellovertex/Documents/github.com/hellovertex/prl_baselines/data/player_data_test"
    filenames = glob.glob(unzipped_dir.__str__() + '/**/*.txt', recursive=True)
    path_out = './snowie_unzipped'
    write_shortened_version_for_snowie_analysis(filenames[0], 5, path_out)
    # reduce_to_hands = 1000
    # start = time.time()
    # p = multiprocessing.Pool()
    # t0 = time.time()
    # clip_fn = partial(write_shortened_version_for_snowie_analysis, n_hands=reduce_to_hands)
    # for x in p.imap_unordered(clip_fn, filenames):
    #     print(x + f'. Took {time.time() - t0} seconds')
    # print(f'Finished job after {time.time() - start} seconds.')
    #
    # p.close()
