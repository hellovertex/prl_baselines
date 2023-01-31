from enum import IntEnum
from functools import partial
from functools import partial
from typing import Optional, Union, List, Tuple

import gym
import numpy as np
from gym.spaces import Box
from pettingzoo import AECEnv
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import ObsType
from prl.environment.Wrappers.utils import init_wrapped_env
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.env.venvs import SubprocVectorEnv

from prl.baselines.agents.tianshou_agents import MCAgent
from prl.baselines.agents.tianshou_policies import MultiAgentActionFlags


class RewardType(IntEnum):
    MBB = 0
    ABSOLUTE = 1


class TianshouEnvWrapper(AECEnv):
    """
    Multi Agent Environment that changes reset call such that
    observation, reward, done, info = reset()
        becomes
    observation = reset(),
    so that tianshou can parse observation properly.
    """

    def __init__(self,
                 env,
                 agents: List[str],
                 reward_type: RewardType):
        super().__init__()
        self.name = "env"
        self.reward_type = reward_type
        self.metadata = {'name': self.name}
        self.agents = agents
        self.possible_agents = self.agents[:]
        self.num_players = len(self.possible_agents)
        self.env_wrapped = env
        self.BIG_BLIND = self.env_wrapped.env.BIG_BLIND
        self._mc_agent = MCAgent(
            ckpt_path="/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/ckpt/ckpt.pt")
        self._last_player_id = -1

        obs_space = Box(low=0.0, high=6.0, shape=(564,), dtype=np.float64)

        # if 'mask_legal_moves' in env_config:
        #     if env_config['mask_legal_moves']:
        observation_space = gym.spaces.Dict({
            'obs': obs_space,  # do not change key-name 'obs' it is internally used by rllib (!)
            'action_mask': Box(low=0, high=1, shape=(3,), dtype=int)
            # one-hot encoded [FOLD, CHECK_CALL, RAISE]
        })
        self.observation_spaces = self._convert_to_dict(
            [observation_space for _ in range(self.num_agents)]
        )
        self.action_spaces = self._convert_to_dict(
            [self.env_wrapped.action_space for _ in range(self.num_agents)]
        )

    def seed(self, seed: Optional[int] = None) -> None:
        np.random.seed(seed)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent: str) -> Optional[ObsType]:
        return {"observation": self._last_obs, "action_mask": self.next_legal_moves}
        # raise NotImplementedError

    def render(self) -> Union[None, np.ndarray, str, list]:
        return self.env_wrapped.render()  # returns None

    def state(self) -> np.ndarray:
        raise NotImplementedError

    def _convert_to_dict(self, list_of_list):
        return dict(zip(self.possible_agents, list_of_list))

    def _scale_rewards(self, rewards):
        if self.reward_type == RewardType.MBB:
            return [r / self.BIG_BLIND for r in rewards]
        if self.reward_type == RewardType.ABSOLUTE:
            return rewards
        else:
            raise NotImplementedError(f"Reward Type {self.reward_type} "
                                      f"not implemented.")

    def _int_to_name(self, ind):
        return self.possible_agents[ind]

    def roll_rewards(self, names: List[str], went_first: int, rewards: List[float], num_players: int):
        """
        Rolls indices from relative to button to relative to agent_list.
        In Backend the button is always at index 0, in PettingZoo env however, any player could be
        the button.
        @param names: Player Names as in PettingZoo wrapped PokerRL env.
        @param went_first: index of player in `agent_list` who started the round after env.reset()
        @param rewards: Payouts gotten from PokerRL backend. Button is always at index 0
        @param num_players: Total number of players at the table. Determines who goes first
        according to the following rule: If number of players is less than four,
        the button starts, otherwise the UTG in [Button, SmallBlind, BigBlind, UTG, MP, CO]
        starts.
        @return: The payouts dictionary with indices relative to the PettingZoo player name list,
        instead of relative to the button. The backend has the button at index 0/
        """
        # player who goes first, either button at index 0 or UTG at index 3
        offset = 0 if num_players < 4 else 3
        # Example 1: "Alice" goes first
        # - ["Bob", "Alice", "Tina"] --> ["Alice", "Tina", "Bob]
        # Example 2:  "Hans" goes first (implies Tina is button)
        # - ["Bob", "Alice", "Hans", "Tina"] --> ["Tina", "Bob", "Alice", "Hans"]
        # Example 3:  "Tina" goes first (implies Alice is button)
        # - ["Bob", "Alice", "Hans", "Stacy", "Tina"] --> ["Alice", "Hans", "Stacy", "Tina", "Bob"]
        return list(np.roll(rewards, offset - went_first))

    def reset(self,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None) -> None:
        if seed is not None:
            self.seed(seed=seed)
        obs, rew, done, info = self.env_wrapped.reset()
        player_id = (self._last_player_id + 1) % self.num_players
        player = self._int_to_name(player_id)
        self.goes_first = player_id
        self.agents = self.possible_agents[:]
        self.agent_selection = player
        self.rewards = self._convert_to_dict([0 for _ in range(self.num_agents)])
        self._cumulative_rewards = self._convert_to_dict(
            [0 for _ in range(self.num_agents)]
        )
        self.terminations = self._convert_to_dict(
            [False for _ in range(self.num_agents)]
        )
        self.truncations = self._convert_to_dict(
            [False for _ in range(self.num_agents)]
        )
        self.infos = self._convert_to_dict(
            [{"legal_moves": []} for _ in range(self.num_agents)]
        )
        legal_moves = np.array([0, 0, 0, 0, 0, 0])
        legal_moves[self.env_wrapped.env.get_legal_actions()] += 1
        if legal_moves[2] == 1:
            legal_moves[[3, 4, 5]] = 1
        self.next_legal_moves = legal_moves
        self._last_obs = obs

    def step(self, action):
        if action == MultiAgentActionFlags.TriggerMC:
            action = self._mc_agent.compute_action(self._last_obs, self.next_legal_moves)
        if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        obs, rew, done, info = self.env_wrapped.step(action)
        self._last_obs = obs
        next_player_id = (self._last_player_id + 1) % self.num_players
        next_player = self._int_to_name(next_player_id)
        if done:
            self.rewards = self._convert_to_dict(
                self._scale_rewards(self.roll_rewards(names=self.possible_agents,
                                                      went_first=self.goes_first,
                                                      rewards=rew,
                                                      num_players=self.num_players))
            )
            self.terminations = self._convert_to_dict(
                [True for _ in range(self.num_agents)]
            )
            self.truncations = self._convert_to_dict(
                [False for _ in range(self.num_agents)]
            )
        else:
            legal_moves = np.array([0, 0, 0, 0, 0, 0])
            legal_moves[self.env_wrapped.env.get_legal_actions()] += 1
            if legal_moves[2] == 1:
                legal_moves[[3, 4, 5]] = 1
            self.next_legal_moves = legal_moves
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = next_player
        self._accumulate_rewards()
        self._deads_step_first()


class WrappedEnv(BaseWrapper):
    def seed(self, seed: Optional[int] = None) -> None:
        np.random.seed(seed)

    def __init__(self, env):
        super().__init__(env)
        self.env = env


def make_env(cfg):
    return init_wrapped_env(**cfg)


def make_vector_env(num_envs: int,
                    single_env_config: dict,
                    agent_names: List[str],
                    reward_type: RewardType = RewardType.MBB) -> Tuple[SubprocVectorEnv, PettingZooEnv]:
    assert len(agent_names) == len(single_env_config['stack_sizes'])
    env = TianshouEnvWrapper(env=make_env(single_env_config),
                             agents=agent_names,
                             reward_type=reward_type)
    wrapped_env_fn = partial(PettingZooEnv, WrappedEnv(env))
    wrapped_env = PettingZooEnv(WrappedEnv(env))
    venv = SubprocVectorEnv([wrapped_env_fn for _ in range(num_envs)])
    return venv, wrapped_env
