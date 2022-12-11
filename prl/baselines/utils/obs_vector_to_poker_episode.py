from datetime import datetime
from typing import Union

import numpy as np
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode
# use vectorized representation of AugmentedObservationFeatureColumns to parse back to PokerEpisode object


def obs_vec_to_episode(obs: Union[list, np.ndarray],
                       normalization_sum: int) -> PokerEpisode:
    # PokerEpisode.date: dateime.now()
    # PokerEpisode.num_players -- positive stacks
    # PokerEpisode.blinds -- normalization_sum * sb, bb index

    date = str(datetime.now())
    hand_id = -1
    variant = "HUNL"
    currency_symbol = "$"
    num_players = None
    pass