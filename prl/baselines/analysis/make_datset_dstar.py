"""
Goal is to set up a comparison pipeline

take each individual player

for each observation, compare actions taken

/home/sascha/Documents/github.com/prl_baselines/data/baseline_model_ckpt.pt

"""
import glob
import multiprocessing
import time

from prl.baselines.supervised_learning.data_acquisition import hsmithy_selector

# todo using mutliprocessing on player names
# rewrite _select_hands(...) such that it
# counts hands where player has folded
# count hands where player reached showdown even if mucked
# how are we gonna compute stats when we dont have observations
# todo with the goal of re-running eval_analyzer.py - like evaluation of selected players
#  but this time with the correct numbers
# this will only be done once so it does not have to have a perfect api
# todo then compute baseline stats from 1M games vs random agents
best_players = ['ishuha',
                'Sakhacop',
                'nastja336',
                'Lucastitos',
                'I LOVE RUS34',
                'SerAlGog',
                'Ma1n1',
                'zMukeha',
                'SoLongRain',
                'LuckyJO777',
                'Nepkin1',
                'blistein',
                'ArcticBearDK',
                'Creator_haze',
                'ilaviiitech',
                'm0bba',
                'KDV707']


def filter_games_for_player(pname: str) -> str:
    folder_out = f"/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/dstar_with_folded_hands/{pname}"
    folder_in__unzipped_txt_files = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    extr = hsmithy_selector.HSmithySelector()
    filenames = glob.glob(
        f'{folder_in__unzipped_txt_files}**/*.txt',
        recursive=True)
    n_files = len(filenames)
    n_files_skipped = 0
    for i, f in enumerate(filenames):
        # print(f'Extracting file {i} / {n_files}')
        try:
            extr.select_from_file(file_path_in=f,
                                  file_path_out=folder_out,
                                  target_player=pname)
        except UnicodeDecodeError:
            n_files_skipped += 1
    return f"Done. Extracted {n_files - n_files_skipped}. {n_files_skipped} files were skipped."


def main():
    start = time.time()
    p = multiprocessing.Pool()
    t0 = time.time()
    for x in p.imap_unordered(filter_games_for_player, best_players):
        print(x + f'. Took {time.time() - t0} seconds')
    print(f'Finished job after {time.time() - start} seconds.')
    p.close()


if __name__ == '__main__':
    main()
