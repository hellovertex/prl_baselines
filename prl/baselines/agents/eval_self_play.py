# need game loop
# need card eval
# need policy sampler that injects fold prob given hand strength
# need component that is able to build PokerEpiosde from series of `obs`
import enum
from typing import List, Union, Tuple

import numpy as np
import torch
from numba import njit
from numba.typed import Dict
from prl.environment.Wrappers.prl_wrappers import ActionSpace, AugmentObservationWrapper

from prl.baselines.cpp_hand_evaluator.experiments.hand_strength import mc
from prl.baselines.pokersnowie.eighteighteight import EightEightEightConverter
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode
from prl.baselines.supervised_learning.data_acquisition.environment_utils import init_wrapped_env
from prl.baselines.supervised_learning.models.nn_model import MLP

def create_wrapped_environment(stacks):
    wrapped_env: AugmentObservationWrapper = init_wrapped_env(env_wrapper_cls=AugmentObservationWrapper,
                                                              stack_sizes=stacks)
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


def run_games(starting_stack_sizes: List[int], n_episodes=100):
    # setup
    env = create_wrapped_environment(starting_stack_sizes)
    agent = Agent(env=env)
    snowie_converter = EightEightEightConverter()
    converted = []

    for i in range(n_episodes):
        obs, _, done, _ = env.reset()
        # episode = init_poker_episode(env_config, obs)
        while not done:
            # query agent that queries model [card_eval + fold_prob]

            obs, _, done, _ = env.step((1,-1))
            obs, _, done, _ = env.step((1,-1))
            obs, _, done, _ = env.step((1,-1))
            obs, _, done, _ = env.step((1,-1))
            obs, _, done, _ = env.step((1,-1))
            obs, _, done, _ = env.step((1,-1))
            obs, _, done, _ = env.step((1,-1))
            obs, _, done, _ = env.step((1,-1))
            obs, _, done, _ = env.step((1,-1))
            obs, _, done, _ = env.step((1,-1))
            obs, _, done, _ = env.step((1,-1))
            obs, _, done, _ = env.step((1,-1))
            action = agent.act(obs)
            obs, _, done, _ = env.step(action)
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
    result = run_games(starting_stack_sizes=[100, 110, 120, 130, 140, 150])
    export(result)
