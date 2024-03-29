import time
from random import random
from typing import List

import numpy as np
import torch
from numba import njit
from prl.environment.Wrappers.base import ActionSpace

from prl.baselines.agents.mc_agent import MCAgent
from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env


def xavier(m, n):
    return torch.nn.init.xavier_uniform_(torch.empty(m, n))


class Player(torch.nn.Module):
    def __init__(self,
                 name,
                 ckpt_to_mc_agent,
                 device,
                 num_players=6):
        super(Player, self).__init__()
        self.device = device
        self.name = name
        self.w0 = torch.nn.Parameter(xavier(570, 512).to(device))
        self.b0 = torch.nn.Parameter(torch.zeros(512).to(device))
        self.w1 = torch.nn.Parameter(xavier(512, 512).to(device))
        self.b1 = torch.nn.Parameter(torch.zeros(512).to(device))
        self.w2 = torch.nn.Parameter(xavier(512, 1).to(device))
        # self.w0 = torch.nn.Linear(570, 512)
        # self.w1 = torch.nn.Linear(512, 512)
        # self.w2 = torch.nn.Linear(512, 1)
        self.mc_agent = MCAgent(ckpt_to_mc_agent, num_players)
        self.collected_rewards = 0

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def compute_fold_probability(self, obs):
        obs = obs @ self.w0 + self.b0
        obs = torch.relu(obs)
        obs = obs @ self.w1 + self.b1
        obs = torch.relu(obs)
        obs = obs @ self.w2
        return torch.sigmoid(obs)

    @staticmethod
    def _mutate(weights, mutation_rate, mutation_std, device):
        weights = torch.tensor(weights)
        # Normalize weights to apply noise
        norm = torch.norm(weights)
        weights = weights / norm

        # Apply noise with a standard deviation of mutation_std
        noise = torch.normal(mean=0., std=mutation_std, size=weights.shape).to(device)

        # Apply mutation by adding noise to weights with a prob mutation_rate
        mutation = torch.bernoulli(torch.full_like(weights, mutation_rate)).to(device)
        weights = weights + mutation * noise

        # Renormalize the weights to have the same norm as before
        weights = weights * norm

        return torch.nn.Parameter(weights)

    def mutate(self, mutation_rate=0.1, mutation_std=0.1):
        self.w0 = self._mutate(self.w0, mutation_rate, mutation_std, self.device)
        self.w1 = self._mutate(self.w1, mutation_rate, mutation_std, self.device)
        self.w2 = self._mutate(self.w2, mutation_rate, mutation_std, self.device)

    def act(self, obs, legal_moves):
        obs = torch.Tensor(np.array([obs])).to(self.device)
        action, probas = self.mc_agent.act(obs,
                                           legal_moves,
                                           report_probas=True)
        # concatenate action probabilities with obs
        obs = torch.concat([obs, probas], dim=1)

        # use concatenated new obs to compute fold_prob
        fold_prob = self.compute_fold_probability(obs)

        if fold_prob < random():
            return ActionSpace.FOLD
        else:
            return action


def play_game(players, env):
    # the environment takes care of moving the button, see eval_tianshou_env.py
    # we can assume player positions are fixed while in the backend they are not
    obs = env.reset()
    agent_id = obs['agent_id']
    legal_moves = obs['mask']
    obs = obs['obs']
    while True:
        i = player_names.index(agent_id)
        action = players[i].act(obs, legal_moves)
        obs_dict, rewards, terminated, truncated, info = env.step(action)
        agent_id = obs_dict['agent_id']
        for pid, rew in enumerate(rewards):
            players[pid].collected_rewards += rew
        obs = obs_dict['obs']
        if terminated:
            break


def play_games(n_games, players, env):
    t0 = time.time()
    for i in range(n_games):
        play_game(players, env)
    print(f'Playing {n_games} games took {time.time() - t0} seconds.')
    return players


def select_best_player(players: List[Player]):
    max_reward = 0
    best_player = 0
    for pid, p in enumerate(players):
        if p.collected_rewards > max_reward:
            max_reward = p.collected_rewards
            best_player = p
    return best_player, max_reward


if __name__ == "__main__":
    n_games = 50
    # n_games = 10000
    evolution_steps = 10000
    # evolution_steps = 100_000_000
    ckpt_to_mc_agent = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_selected_players/ckpt_dir/Lucastitos_[256]_1e-06/ckpt.pt"

    # 1. init 6 players with random params or from checkpoint
    players = [Player(name=f"Player_{i}", device="cuda",
                      ckpt_to_mc_agent=ckpt_to_mc_agent) for i in range(6)]
    player_names = [p.name for p in players]
    try:
        print(f'Loaded from checkpoint... Continuing evolution')
        [p.load_weights('./best_player.pt') for p in players]
    except FileNotFoundError:
        pass

    # 1.1 Create poker environment that automatically moves btn after reset, see `eval_tianshou_env.py`
    env = make_default_tianshou_env(ckpt_to_mc_agent,
                                    agents=player_names,
                                    num_players=6)
    epoch = 0
    # 2. Play many games to let each player accumulate rewards over all games
    while True:
        print(f'Playing {n_games} games at epoch {epoch}/{evolution_steps}')
        players = play_games(n_games=n_games, players=players, env=env)

        # 3. evaluate players using collected rewards
        best_player, collected_reward = select_best_player(players)

        # 4. mutate weights to generate new generation
        best_player.save_weights('./best_player.pt')
        players = [best_player for _ in range(6)]
        [p.mutate(mutation_rate=.2) for p in players]

        # 5. terminate when `evolution_steps` threshold reached
        epoch += 1
        if epoch > evolution_steps:
            break
