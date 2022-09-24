import pickle

from prl.baselines.supervised_learning.config import DATA_DIR
from prl.baselines.supervised_learning.data_acquisition import core
from prl.baselines.supervised_learning.data_acquisition.core.parser import Blind, PlayerStack, Action


def deprecated():
    """    n_players = 6
    dummy_blind_0 = Blind('p0', 'small blind', "$1")
    dummy_blind_1 = Blind('p1', 'big blind', "$2")
    blinds = [dummy_blind_0, dummy_blind_1]
    startstack = 100
    player_stacks = [PlayerStack(f'Seat{i + 1}', f'p{i}', f'${str(startstack)}') for i in range(n_players)]
    a = Action()
# poker_episode = PokerEpisode(date='',
#                                hand_id=1234567,
#                                variant='NoLimit',
#                                currency_symbol="$",
#                                num_players=6,
#                                blinds=blinds,
#                                ante='$0.00',
#                                player_stacks=player_stacks,
#                                btn_idx=5,
#                                board_cards='[6h Ts Td 9c Jc]',
#                                actions_total=actions_total,
#                                )"""
    pass


def get_dummy_episode():
    """Unpickle poker episode"""
    # load pickle file and use PokerEpisodes
    filepath = str(DATA_DIR) + "/01_raw" + "/0.25-0.50" + "/pickled/"
    filename = "poker_episodes_1"
    with open(filepath + filename, "rb") as f:
        while True:
            try:
                data = pickle.load(f)
                print(type(data))
                return data
            except Exception as e:
                print(e)
                break


get_dummy_episode()
