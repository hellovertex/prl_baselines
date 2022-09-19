# Parse all example files and dump using pickle
# Load all dumps and verify that each hand is contained
from prl.environment.Wrappers.prl_wrappers import AugmentObservationWrapper

from prl.baselines.supervised_learning.config import LOGFILE
from prl.baselines.supervised_learning.data_acquisition.csv_writer import CSVWriter
from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser
from prl.baselines.supervised_learning.data_acquisition.rl_state_encoder import RLStateEncoder
from prl.baselines.supervised_learning.data_acquisition.runner import Runner


def test_all_poker_episodes_are_kept():
    # load pickle file and use PokerEpisodes
    pass


