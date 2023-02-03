from random import random
from typing import List

import numpy as np
from numba import njit
from prl.environment.Wrappers.base import ActionSpace

from prl.baselines.agents.mc_agent import MCAgent
from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env


def xavier(input_size, output_size):
    var = 2. / (input_size + output_size)
    bound = np.sqrt(3.0 * var)
    return np.random.uniform(-bound, bound, size=(input_size, output_size))


sigmoid = lambda x: 1 / (1 + np.exp(-x))
relu = lambda x: np.maximum(x, 0)


class Player:
    def __init__(self,
                 name,
                 ckpt_to_mc_agent,
                 num_players=6):
        self.name = name
        self.w0 = xavier(564, 512)
        self.b0 = np.zeros(512)
        self.w1 = xavier(512, 512)
        self.b1 = np.zeros(512)
        self.w2 = xavier(512, 1)
        self.mc_agent = MCAgent(ckpt_to_mc_agent, num_players)
        self.wnb = [self.w0, self.b0, self.w1, self.b1, self.w2]
        self.collected_rewards = 0

    def save(self, filename):
        np.savez(filename,
                 w0=self.w0, w1=self.w1, w2=self.w2,
                 b0=self.b0, b1=self.b1)

    def load(self, filename):
        data = np.load(filename)
        self.w0 = data['w0']
        self.w1 = data['w1']
        self.w2 = data['w2']
        self.b0 = data['b0']
        self.b1 = data['b1']
        return self

    @njit
    def _mutate(self, weights, mutation_rate, mutation_std):
        # Normalize weights to apply noise
        norm = np.linalg.norm(weights)
        weights = weights / norm

        # Apply noise with a standard deviation of mutation_std
        noise = np.random.normal(0, mutation_std, size=weights.shape)

        # Apply mutation by adding noise to weights with a prob mutation_rate
        mutation = np.random.binomial(1, mutation_rate, size=weights.shape)
        weights = weights + mutation * noise

        # Renormalize the weights to have the same norm as before
        weights = weights * norm

        return weights

    def mutate(self, mutation_rate=0.1, mutation_std=0.1):
        for params in self.wnb:
            self._mutate(params, mutation_rate, mutation_std)

    @njit
    def compute_fold_probability(self, obs):
        x = obs
        x = np.dot(x, self.w0) + self.b0
        x = relu(x)
        x = np.dot(x, self.w1) + self.b1
        x * relu(x)
        x = np.dot(x, self.w2)
        return sigmoid(x)

    def act(self, obs, legal_moves):
        action, probas = self.mc_agent.act(obs,
                                           legal_moves,
                                           report_probas=True)
        # concatenate action probabilities with obs
        if type(obs) == list:
            obs = obs + probas
        elif type(obs) == np.ndarray:
            obs = np.concatenate([obs, probas])

        # use concatenated new obs to compute fold_prob
        fold_prob = self.compute_fold_probability(obs, probas)

        if fold_prob < random():
            return ActionSpace.FOLD
        else:
            return action


"""
    Has a .ckpt model from game log supervised learning
    self.base_model = base_model
    FrameWork requirements:

    - MARL: acts together with other agents in a MARL fashion:
    - VectorEnv?
    - PBT?
    # algo:
    # [x] 0. define fitness function -- rewards
    # [x] 1. init 6 players with random weights
    # [x] 2. play M games -- collect rewards per player
    # [x] 3. evaluate players using rewards collected
    # 4. select the best player P1 using fitness function -- i) mutate ii) save weights to current best
    # 5. repeat 2,3,4
    It is enough to have a 569 x

    todo: figure out how we can use the trained .ckpt MLP quickly(vectorized/mp) with this numpy based approach
    """


def play_game(players, env):
    # the environment takes care of moving the button, see eval_tianshou_env.py
    # we can assume player positions are fixed while in the backend they are not
    obs = env.reset()
    agent_id = obs['agent_id']
    legal_moves = obs['mask']
    obs = obs['obs']
    while True:
        i = player_names.index(agent_id)
        action = player_names[i].act(obs, legal_moves)
        obs_dict, cum_rewards, terminated, truncated, info = env.step(action)
        agent_id = obs_dict['agent_id']
        players[i].collected_rewards += cum_rewards
        obs = obs_dict['obs']
        if terminated:
            break


def play_games(n_games, players, env):
    for i in range(n_games):
        play_game(players, env)


def select_best_player(players: List[Player]):
    max_reward = 0
    best_player = 0
    for pid, p in enumerate(players):
        if p.collected_rewards > max_reward:
            max_reward = p.collected_rewards
            best_player = p
    return best_player, max_reward


if __name__ == "__main__":
    n_games = 10
    #n_games = 10000
    evolution_steps = 100
    # evolution_steps = 100_000_000
    ckpt_to_mc_agent = "/home/sascha/Documents/github.com/prl_baselines/data/ckpt/ckpt.pt"

    # 1. init 6 players with random params
    players = [Player(name=f"Player_{i}",
                      ckpt_to_mc_agent=ckpt_to_mc_agent) for i in range(6)]
    player_names = [p.name for p in players]

    # 1.1 Create poker environment that automatically moves btn after reset, see `eval_tianshou_env.py`
    env = make_default_tianshou_env(ckpt_to_mc_agent,
                                    agents=player_names,
                                    num_players=2)
    epoch = 0
    while True:
        # 2. Play many games to let each player accumulate rewards over all games
        play_games(n_games=10000, players=players, env=env)

        # 3. evaluate players using collected rewards
        best_player, collected_reward = select_best_player

        # 4. mutate weights to generate new generation
        new_gen = [best_player for _ in range(6)]
        [p.mutate(mutation_rate=.2) for p in new_gen]

        # 5. terminate after
        epoch += 1
        if epoch > evolution_steps:
            break



