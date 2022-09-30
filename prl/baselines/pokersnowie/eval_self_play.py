# need game loop
# need card eval
# need policy sampler that injects fold prob given hand strength
# need component that is able to build PokerEpiosde from series of `obs`
from typing import List

from prl.baselines.pokersnowie.eighteighteight import EightEightEightConverter
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode


class Agent:
    def __init__(self):
        self._model = None
        self._card_evaluator = None
        self._policy = None

    def load_model(self):
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
