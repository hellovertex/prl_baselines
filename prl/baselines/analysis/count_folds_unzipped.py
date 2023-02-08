"""
Goal is to set up a comparison pipeline

take each individual player

for each observation, compare actions taken

/home/sascha/Documents/github.com/prl_baselines/data/baseline_model_ckpt.pt

"""
import glob

from prl.baselines.supervised_learning.data_acquisition import hsmithy_selector

folder_out = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data"
extr = hsmithy_selector.HSmithySelector()
filenames = glob.glob("/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped/**/*.txt",
                      recursive=True)
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
n_files = len(filenames)
n_files_skipped = 0
for i, f in enumerate(filenames):
    print(f'Extractin file {i} / {n_files}')
    for pname in best_players:
        try:
            extr.select_from_file(file_path_in=f,
                              file_path_out=folder_out,
                              target_player=pname)
        except UnicodeDecodeError:
            n_files_skipped += 1
print(f"Done. Extracted {n_files - n_files_skipped}. {n_files_skipped} files were skipped.")

