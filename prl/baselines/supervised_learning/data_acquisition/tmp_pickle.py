# Parse all example files and dump using pickle
# Load all dumps and verify that each hand is contained
import pickle
from prl.baselines.supervised_learning.config import LOGFILE, DATA_DIR
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode
from prl.baselines.supervised_learning.data_acquisition import core


def test_all_poker_episodes_are_kept():
    # load pickle file and use PokerEpisodes
    filepath = str(DATA_DIR) + "/01_raw" + "/0.25-0.50" + "/pickled/"
    filename = "poker_episodes_1"
    with open(filepath + filename, "rb") as f:
        data = pickle.load(f)
    print(type(data))


test_all_poker_episodes_are_kept()
