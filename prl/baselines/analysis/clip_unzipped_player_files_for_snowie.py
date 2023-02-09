from functools import partial

import glob
import time
import multiprocessing


def write_shortened_version_for_snowie_analysis(filename, n_hands, out_path):
    # read file and write back first n_hands

    pass


if __name__ == '__main__':
    # load player dirs via multiprocessing and write back stripped down files for pokersnowie eval
    # 1k showdown hands

    unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data"
    filenames = glob.glob(unzipped_dir.__str__() + '/**/*.txt', recursive=True)
    out_path = './snowie_unzipped'
    write_shortened_version_for_snowie_analysis(filenames[0], 1, out_path)
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
