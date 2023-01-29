"""
Goal is to set up a comparison pipeline

take each individual player

for each observation, compare actions taken

/home/sascha/Documents/github.com/prl_baselines/data/baseline_model_ckpt.pt

"""


from prl.baselines.supervised_learning.data_acquisition import hsmithy_extractor

target_player = "Clamfish0"
fpath = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped/Aaltje III-0.25-0.50-USD-NoLimitHoldem-PokerStars-2-8-2020.txt"
folder_out = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/"
extr = hsmithy_extractor.HSmithyExtractor()
extr.extract_file(file_path_in=fpath,
                  file_path_out=folder_out,
                  target_player="Clamfish0")