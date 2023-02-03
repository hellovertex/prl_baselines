from enum import IntEnum
from functools import partial
from typing import Tuple, List, Union, Optional

import gym
import numpy as np
from gym.spaces import Box
from pettingzoo import AECEnv
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import ObsType
from prl.environment.Wrappers.aoh import Positions6Max
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.env.venvs import SubprocVectorEnv
from prl.baselines.agents.mc_agent import MCAgent
from prl.baselines.agents.tianshou_policies import MultiAgentActionFlags
from prl.baselines.evaluation.core.experiment import ENV_WRAPPER


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
                 reward_type: RewardType,
                 mc_ckpt_path: str):
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
            ckpt_path=mc_ckpt_path, num_players=self.num_players)
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
        self.agent_map = {}
        for i in range(self.num_players):
            self.agent_map[i] = i

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

    def reset(self,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None) -> None:
        if seed is not None:
            self.seed(seed=seed)
        obs, rew, done, info = self.env_wrapped.reset()
        shifted_indices = {}
        for rel_btn, agent_idx in self.agent_map.items():
            shifted_indices[rel_btn] = (agent_idx + 1) % self.num_players
        self.agent_map = shifted_indices
        player_id = self.agent_map[self.env_wrapped.env.current_player.seat_id]
        player = self._int_to_name(player_id)

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
        next_player_id = self.env_wrapped.env.current_player.seat_id
        next_player_id = self.agent_map[next_player_id]
        next_player = self._int_to_name(next_player_id)
        # roll button to correct position [BTN,...,] to [,...,BTN,...]
        rewards = np.roll(rew, self.agent_map[Positions6Max.BTN])
        if done:
            self.rewards = self._convert_to_dict(
                self._scale_rewards(rewards)
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


def make_default_tianshou_env(mc_model_ckpt_path, num_players=2):
    starting_stack = 20000
    stack_sizes = [starting_stack for _ in range(num_players)]
    agents = [f'p{i}' for i in range(num_players)]
    env_config = {"env_wrapper_cls": AugmentObservationWrapper,
                  # "stack_sizes": [100, 125, 150, 175, 200, 250],
                  "stack_sizes": stack_sizes,
                  "multiply_by": 1,  # use 100 for floats to remove decimals but we have int stacks
                  "scale_rewards": False,  # we do this ourselves
                  "blinds": [50, 100]}
    # env = init_wrapped_env(**env_config)
    # obs0 = env.reset(config=None)

    env = TianshouEnvWrapper(env=make_env(env_config),
                             agents=agents,
                             mc_ckpt_path=mc_model_ckpt_path,
                             reward_type=RewardType.MBB)
    return env

def make_env(cfg):
    return init_wrapped_env(**cfg)


def make_vectorized_pettingzoo_env(num_envs: int,
                                   single_env_config: dict,
                                   agent_names: List[str],
                                   mc_model_ckpt_path: str,
                                   reward_type: RewardType = RewardType.MBB,
                                   ) -> Tuple[SubprocVectorEnv, PettingZooEnv]:
    assert len(agent_names) == len(single_env_config['stack_sizes'])
    env = TianshouEnvWrapper(env=make_env(single_env_config),
                             agents=agent_names,
                             reward_type=reward_type,
                             mc_ckpt_path=mc_model_ckpt_path)
    wrapped_env_fn = partial(PettingZooEnv, WrappedEnv(env))
    wrapped_env = PettingZooEnv(WrappedEnv(env))
    venv = SubprocVectorEnv([wrapped_env_fn for _ in range(num_envs)])
    return venv, wrapped_env


# class TianshouWrappedSingleEnv(AugmentObservationWrapper):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     @override
#     def observation_space(self):
#         obs_space = Box(low=0.0, high=6.0, shape=(564,), dtype=np.float64)
#
#         # if 'mask_legal_moves' in env_config:
#         #     if env_config['mask_legal_moves']:
#         return gym.spaces.Dict({
#             'obs': obs_space,  # do not change key-name 'obs' it is internally used by rllib (!)
#             'action_mask': Box(low=0, high=1, shape=(3,), dtype=int)
#             # one-hot encoded [FOLD, CHECK_CALL, RAISE]
#         })


def make_vectorized_prl_env(num_envs: int,
                            single_env_config: dict,
                            agent_names: List[str],
                            ) -> Tuple[SubprocVectorEnv, ENV_WRAPPER]:
    assert len(agent_names) == len(single_env_config['stack_sizes'])
    single_env_config['disable_info'] = True  # drop info we do not need it with tianshou
    wrapped_env_fn = partial(make_env, single_env_config)
    wrapped_env = wrapped_env_fn()
    venv = SubprocVectorEnv([wrapped_env_fn for _ in range(num_envs)])
    return venv, wrapped_env
