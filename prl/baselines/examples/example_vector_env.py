from enum import IntEnum
from typing import Optional, Union, List

import gym
import numpy as np
from gym.spaces import Box
from pettingzoo import AECEnv
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import ObsType
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env
from tianshou.data import Collector
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RainbowPolicy
from tianshou.env.venvs import SubprocVectorEnv


# todo implement this https://pettingzoo.farama.org/tutorials/tianshou/intermediate/
# todo: create one policy (actually zero because c51 already exists and the _mc_agent
#  does not have to be a policy after all
# todo: copy from policies.StakeLevelImitationPolicy.compute_action(obs:np.ndarray,...)
#  to create an agent that lives inside the environment


class MultiAgentActionFlags(IntEnum):
    """
    In the Baseline Agents, the Monte Carlo simulation takes a lot of time,
    so we want to parallelize its computations.
    Therefore, we include the MC simulator into the environment and let it be parallelized from
    tianshou by triggering the environment to compute the MC-based agent by itself, instead of
    getting it from the outside from an agent. This way when calling step with an MCTrigger - Action,
    the environment asks the BaselineAgent to compute its action as part of the env.step functionality.
    This parallelizes one MC-simulation per num_env. So we get maximum speedup by
    setting num_envs equal to the number of CPU cores available, e.g. 32.
    And can run 32 MC simulations at the same time -- the same number of env.step() we can call at
    the same time."""
    TriggerMC = 0


class MCAgent:
    def act(self, _):
        return MultiAgentActionFlags.TriggerMC


class TianshouEnvWrapper(AECEnv, BaseWrapper):
    """
    Multi Agent Environment that changes reset call such that
    observation, reward, done, info = reset()
        becomes
    observation = reset(),
    so that tianshou can parse observation properly.
    """

    def __init__(self, env, agents: List[str]):
        super().__init__()
        self.agents = agents
        self.env_wrapped = env
        # moved this to prl.baselines because I understand we need access to the baseline agents
        # which are not available from within prl_environment
        self._mc_agent = None
        obs_space = Box(low=0.0, high=6.0, shape=(564,), dtype=np.float64)
        self.observation_space = obs_space
        # if 'mask_legal_moves' in env_config:
        #     if env_config['mask_legal_moves']:
        self.observation_space = gym.spaces.Dict({
            'obs': obs_space,  # do not change key-name 'obs' it is internally used by rllib (!)
            'action_mask': Box(low=0, high=1, shape=(3,), dtype=int)
            # one-hot encoded [FOLD, CHECK_CALL, RAISE]
        })
        self.observation_spaces = self._convert_to_dict(
            [self.observation_space for _ in range(self.num_agents)]
        )
        self.action_spaces = self._convert_to_dict(
            [self.env_wrapped.action_space for _ in range(self.num_agents)]
        )
        self.possible_agents = self.agents[:]

    def seed(self, seed: Optional[int] = None) -> None:
        np.random.seed(seed)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent: str) -> Optional[ObsType]:
        # todo: add the following line but first check if it is needed by tianshou
        # return {"observation": self._last_obs, "action_mask": self.next_legal_moves}
        raise NotImplementedError

    def render(self) -> Union[None, np.ndarray, str, list]:
        return self.env_wrapped.render()  # returns None

    def state(self) -> np.ndarray:
        raise NotImplementedError

    def _convert_to_dict(self, list_of_list):
        return dict(zip(self.possible_agents, list_of_list))

    def _scale_rewards(self, reward):
        return reward

    def _int_to_name(self, ind):
        return self.possible_agents[ind]

    def reset(self,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None) -> None:
        if seed is not None:
            self.seed(seed=seed)
        obs, rew, done, info = self.env_wrapped.reset()
        player_id = self.env_wrapped.current_player.seat_id
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
        # todo double check these:
        legal_moves = np.array([0, 0, 0])
        legal_moves[self.env_wrapped.env.get_legal_actions()] += 1
        self.next_legal_moves = legal_moves
        self._last_obs = obs

    def step(self, action):
        # todo add the following code
        # if action == MultiAgentActionFlags.TriggerMC:
        #     # compute action here
        #     pass
        if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        obs, rew, done, info = self.env_wrapped.step(action)
        next_player_id = self.env_wrapped.current_player.seat_id
        next_player = self._int_to_name(next_player_id)
        if done:
            self.rewards = self._convert_to_dict(
                self._scale_rewards(info['payouts'])
            )
            self.terminations = self._convert_to_dict(
                [True for _ in range(self.num_agents)]
            )
            self.truncations = self._convert_to_dict(
                [False for _ in range(self.num_agents)]
            )
        else:
            legal_moves = np.array([0, 0, 0])
            legal_moves[self.env_wrapped.env.get_legal_actions()] += 1
            self.next_legal_moves = legal_moves
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = next_player
        self._accumulate_rewards()
        self._deads_step_first()


# class WrappedEnv(BaseWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.env = env


env_config = {"env_wrapper_cls": AugmentObservationWrapper,
              "stack_sizes": [100, 125, 150, 175, 200, 250],
              "blinds": [50, 100]}
# env = init_wrapped_env(**env_config)
# obs0 = env.reset(config=None)
num_envs = 3


def make_env(cfg):
    return init_wrapped_env(**cfg)


agents = ["p0", "p1"]
env = PettingZooEnv(TianshouEnvWrapper(make_env(env_config), agents))
venv = SubprocVectorEnv([env for _ in range(num_envs)])
# env_fn = partial(make_env, env_config)
# env_fns = [env_fn for _ in range(num_envs)]
# # venv = SubprocVectorEnv(env_fns, wait_num=None, timeout=None)
# venv = DummyVectorEnv(env_fns, wait_num=None, timeout=None)
# obs = venv.reset()  # returns the initial observations of each environment
# # todo get ready_id`s and reset only with ids of envs that signalled `done`
# # returns "wait_num" steps or finished steps after "timeout" seconds,
# # whichever occurs first.
# print(obs)
rainbow_config = {'model': None,
                  'optim': None,
                  'num_atmos': 51,
                  'v_min': -6,
                  'v_max': 6,
                  'estimation_step': 3,
                  'target_update_freq': 500  # training steps
                  }
policy = MultiAgentPolicyManager([
    RainbowPolicy(**rainbow_config),
    RainbowPolicy(**rainbow_config)], env)  # policy is made from PettingZooEnv

collector = Collector(policy, venv)
result = collector.collect(n_episode=1, render=.1)
