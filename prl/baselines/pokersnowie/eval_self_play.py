# need game loop
# need card eval
# need policy sampler that injects fold prob given hand strength
# need component that is able to build PokerEpiosde from series of `obs`
import enum
from typing import List, Union

import numpy as np
import torch
from prl.environment.Wrappers.prl_wrappers import ActionSpace

from prl.baselines.pokersnowie.eighteighteight import EightEightEightConverter
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode
from prl.baselines.supervised_learning.models.nn_model import MLP

N_FEATURES = 564


class AgentModelType(enum.IntEnum):
    MLP_2x512 = 10
    RANDOM_FOREST = 20


class Agent:
    def __init__(self):
        self._model = None
        self._card_evaluator = None
        self._policy = None

    def load_model(self, model_type: AgentModelType):
        if model_type == AgentModelType.MLP_2x512:
            input_dim = N_FEATURES
            classes = [ActionSpace.FOLD,
                       ActionSpace.CHECK_CALL,  # CHECK IS INCLUDED
                       ActionSpace.RAISE_MIN_OR_3BB,
                       ActionSpace.RAISE_HALF_POT,
                       ActionSpace.RAISE_POT,
                       ActionSpace.ALL_IN]
            hidden_dim = [512, 512]
            output_dim = len(classes)
            net = MLP(input_dim, output_dim, hidden_dim)
            # if running on GPU and we want to use cuda move model there
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                net = net.cuda()
            self._model = net
            return net
        else:
            raise NotImplementedError

    def act(self, obs: Union[np.arary, List]):
        # from obs, get cards
        # from cards get ranking
        # todo: preflop - n player lookup (equities do)
        #  flop: compute [Flop, 1326] and rank hero vs 1325 => equity estimate
        #  test this using known hand and board combinations
        #  assume Rank(.,.) is quick and correct for now
        # from ranking get fold prob
        # from fold prob get policy
        # from policy, return action
        pass


def create_wrapped_environment():
    wrapped_env = False
    return wrapped_env


# Consider using this:
# class Runner:
#     def __init__(self):
#         self._agent = None
#         self._env = None
#         self._converter = None
#
#     def reset(self):
#         self._agent = Agent()  # is stateless
#         self._env = create_wrapped_environment()
#         self._converter = EightEightEightConverter()


def run_games(n_episodes=100):
    # setup
    agent = Agent()  # is stateless
    env = create_wrapped_environment()
    converter = EightEightEightConverter()
    converted = []
    # env_config = None
    for i in range(n_episodes):
        # obs, _, _, _ = env.reset(env_config)
        done = False
        # episode = init_poker_episode(env_config, obs)
        while not done:
            # todo query agent that queries model [card_eval + fold_prob]
            # todo update_poker_episode
            break
        # converted.append(converter.from_episode(episode))

    return converted


def init_poker_episode(env_conf, init_obs) -> PokerEpisode:
    pass


def update_poker_episode(env, obs):
    pass


def export(episodes: List[PokerEpisode]):
    pass


if __name__ == "__main__":
    result = run_games()
    export(result)
