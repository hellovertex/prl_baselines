from random import random

import numpy as np
from numba import njit
from prl.environment.Wrappers.base import ActionSpace

from prl.baselines.agents.mc_agent import MCAgent
from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env

mc_model_ckpt_path = "/home/sascha/Documents/github.com/prl_baselines/data/ckpt/ckpt.pt"
env = make_default_tianshou_env(mc_model_ckpt_path, num_players=2)


def xavier(input_size, output_size):
    var = 2. / (input_size + output_size)
    bound = np.sqrt(3.0 * var)
    return np.random.uniform(-bound, bound, size=(input_size, output_size))


sigmoid = lambda x: 1 / (1 + np.exp(-x))
relu = lambda x: np.maximum(x, 0)


class Player:
    def __init__(self, ckpt_to_mc_agent, num_players=6):
        self.w0 = xavier(564, 512)
        self.b0 = np.zeros(512)
        self.w1 = xavier(512, 512)
        self.b1 = np.zeros(512)
        self.w2 = xavier(512, 1)
        self.mc_agent = MCAgent(ckpt_to_mc_agent, num_players)
        self.wnb = [self.w0, self.b0, self.w1, self.b1, self.w2]

    @njit
    def _mutate(self, weights, mutation_rate, mutation_std):
        # Normalize the weights to have a unit norm
        norm = np.linalg.norm(weights)
        weights = weights / norm

        # Generate random noise with a standard deviation of mutation_std
        noise = np.random.normal(0, mutation_std, size=weights.shape)

        # Apply the mutation by adding the noise to the weights with a probability of mutation_rate
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
        fold_prob = self.compute_fold_probability(obs)
        if fold_prob < random():
            return ActionSpace.FOLD
        else:
            return self.mc_agent.act(obs, legal_moves)


class BaselineAgent:
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
    # 2. play M games -- collect rewards per player
    # 3. evaluate players using rewards collected
    # 4. select the best player P1 using fitness function -- i) mutate ii) save weights to current best
    # 5. repeat 2,3,4
    It is enough to have a 569 x

    todo: figure out how we can use the trained .ckpt MLP quickly(vectorized/mp) with this numpy based approach
    """


# 1. init 6 players
players = [Player(mc_model_ckpt_path) for _ in range(6)]
# 2. play M games
M = 1000


def play_game(players, env):
    obs, rews, done, info = env.reset()
    while True:
        players


def play_games(n_games, players, env):
    for i in range(n_games):
        play_game(players, env)


#
action_probs = np.random.random(5)
obs = np.random.random(564)
w0 = xavier(564, 512)
b0 = np.zeros(512)
w1 = xavier(512, 512)
b1 = np.zeros(512)
w2 = xavier(512, 1)
x = obs
x = np.dot(x, w0) + b0
x = relu(x)
x = np.dot(x, w1) + b1
x * relu(x)
x = np.dot(x, w2)
fold_prob = sigmoid(x)
# np.dot()
a = 1
print(min(obs), max(obs), obs.shape)
