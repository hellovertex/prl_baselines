# need game loop
# need card eval
# need policy sampler that injects fold prob given hand strength
# need component that is able to build PokerEpiosde from series of `obs`
from typing import List

from prl.baselines.pokersnowie.eighteighteight import EightEightEightConverter
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode


def create_wrapped_environment():
    pass


def load_model():
    pass


def run_games(n_episodes=100):
    converter = EightEightEightConverter()
    converted = []
    # episode = init_poker_episode()
    # converter.from_poker_episode()
    done = False
    for i in range(n_episodes):
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
