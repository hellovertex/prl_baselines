"""
Goal is to set up a comparison pipeline

take each individual player

for each observation, compare actions taken

/home/sascha/Documents/github.com/prl_baselines/data/baseline_model_ckpt.pt

"""
import glob
import json
import multiprocessing
import time
from functools import partial

from prl.baselines.analysis.core.dataset_stats import DatasetStats


def run(datasetname, top_lvl_folder) -> str:
    stats = DatasetStats()

    filenames = glob.glob(
        f'{top_lvl_folder}**/*.txt',
        recursive=True)
    n_files = len(filenames)
    n_files_skipped = 0

    # Update player stats file by file
    for i, f in enumerate(filenames):
        # print(f'Extracting file {i} / {n_files}')
        try:
            stats.update_from_file(file_path=f)
        except UnicodeDecodeError:
            # should not occur too often, less than 1% of files have some continuation byte errs
            n_files_skipped += 1

    # Flush to disk
    to_dict = stats.to_dict()
    with open(f'stats_{datasetname}.txt', 'a+') as f:
        f.write(json.dumps(to_dict))

    return f"Done. Extracted {n_files - n_files_skipped}/ {n_files}. {n_files_skipped} files were skipped."


# def main():
#     debug = False
#     if not debug:
#         start = time.time()
#         p = multiprocessing.Pool()
#         t0 = time.time()
#         for x in p.imap_unordered(filter_games_for_player, best_players):
#             print(x + f'. Took {time.time() - t0} seconds')
#         print(f'Finished job after {time.time() - start} seconds.')
#         p.close()
#     else:
#         for p in best_players:
#             filter_games_for_player(p)


if __name__ == '__main__':
    # use with unzipped and player_data simultaneously
    folder_in__unzipped_txt_files = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    folder_in__player_data_txt_files = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    datasetnames = ['D', 'Dprime']
    folders = [folder_in__unzipped_txt_files,
               folder_in__player_data_txt_files]
    for d, f in zip(datasetnames, folders):
        run(datasetname=d, top_lvl_folder=f)


