"""
Goal is to set up a comparison pipeline

take each individual player

for each observation, compare actions taken

/home/sascha/Documents/github.com/prl_baselines/data/baseline_model_ckpt.pt

"""
import glob

from prl.baselines.supervised_learning.data_acquisition import hsmithy_extractor

target_player = "Clamfish0"
fpath = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped/Aaltje III-0.25-0.50-USD-NoLimitHoldem-PokerStars-2-8-2020.txt"
folder_out = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50"
extr = hsmithy_extractor.HSmithyExtractor()
filenames = glob.glob("/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped/**/*.txt", recursive=True)
# todo: loop extract file for all 60gb of unzipped files
#  loop target players for all of
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
for f in filenames:
    for pname in best_players:
        extr.extract_file(file_path_in=f,
                          file_path_out=folder_out,
                          target_player=pname)

gains = {"ishuha": {"n_hands_played": 59961, "n_showdowns": 18050, "n_won": 10598, "total_earnings": 37425.32000000013},
         "Sakhacop": {"n_hands_played": 54873, "n_showdowns": 14718, "n_won": 9235,
                      "total_earnings": 43113.86000000007},
         "nastja336": {"n_hands_played": 48709, "n_showdowns": 11231, "n_won": 7303,
                       "total_earnings": 35729.6900000002},
         "Lucastitos": {"n_hands_played": 37898, "n_showdowns": 11117, "n_won": 6811,
                        "total_earnings": 20171.329999999984},
         "I LOVE RUS34": {"n_hands_played": 36457, "n_showdowns": 10985, "n_won": 6441,
                          "total_earnings": 21504.00999999993},
         "SerAlGog": {"n_hands_played": 50103, "n_showdowns": 10850, "n_won": 6613,
                      "total_earnings": 25631.720000000074},
         "Ma1n1": {"n_hands_played": 40296, "n_showdowns": 9792, "n_won": 6188, "total_earnings": 25016.60999999976},
         "zMukeha": {"n_hands_played": 34991, "n_showdowns": 9104, "n_won": 6003, "total_earnings": 21469.710000000083},
         "SoLongRain": {"n_hands_played": 33826, "n_showdowns": 8722, "n_won": 5381,
                        "total_earnings": 22247.390000000087},
         "LuckyJO777": {"n_hands_played": 33201, "n_showdowns": 8118, "n_won": 5283,
                        "total_earnings": 26579.860000000004},
         "Nepkin1": {"n_hands_played": 28467, "n_showdowns": 8032, "n_won": 5501, "total_earnings": 21739.989999999976},
         "blistein": {"n_hands_played": 34620, "n_showdowns": 7966, "n_won": 5134, "total_earnings": 20824.44000000001},
         "ArcticBearDK": {"n_hands_played": 24449, "n_showdowns": 6849, "n_won": 4292,
                          "total_earnings": 24626.509999999973},
         "Creator_haze": {"n_hands_played": 23882, "n_showdowns": 6737, "n_won": 4172,
                          "total_earnings": 20679.31000000002},
         "ilaviiitech": {"n_hands_played": 29527, "n_showdowns": 6401, "n_won": 4213,
                         "total_earnings": 22407.82999999994},
         "m0bba": {"n_hands_played": 24384, "n_showdowns": 6349, "n_won": 4325, "total_earnings": 25772.450000000015},
         "KDV707": {"n_hands_played": 25570, "n_showdowns": 6241, "n_won": 3492, "total_earnings": 23929.970000000063}}
