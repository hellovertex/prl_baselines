from enum import IntEnum
from functools import partial
from typing import Optional, Union, List

import gym
import numpy as np
import torch
from gym.spaces import Box
from pettingzoo import AECEnv
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import ObsType
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.base import ActionSpace
from prl.environment.Wrappers.utils import init_wrapped_env
from tianshou.data import Collector
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.env.venvs import SubprocVectorEnv
from tianshou.policy import MultiAgentPolicyManager, RainbowPolicy
from tianshou.utils.net.common import Net


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


class TianshouEnvWrapper(AECEnv):
    """
    Multi Agent Environment that changes reset call such that
    observation, reward, done, info = reset()
        becomes
    observation = reset(),
    so that tianshou can parse observation properly.
    """

    def __init__(self, env, agents: List[str]):
        super().__init__()
        self.name = "env"
        self.metadata = {'name': self.name}
        self.agents = agents
        self.possible_agents = self.agents[:]
        self.env_wrapped = env
        # moved this to prl.baselines because I understand we need access to the baseline agents
        # which are not available from within prl_environment
        self._mc_agent = None
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
        legal_moves = np.array([0, 0, 0, 0, 0, 0])
        legal_moves[self.env_wrapped.env.get_legal_actions()] += 1
        if legal_moves[2] == 1:
            legal_moves[[3, 4, 5]] = 1
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
            legal_moves = np.array([0, 0, 0, 0, 0, 0])
            legal_moves[self.env_wrapped.env.get_legal_actions()] += 1
            if legal_moves[2] == 1:
                legal_moves[[3,4,5]] = 1
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


env_config = {"env_wrapper_cls": AugmentObservationWrapper,
              # "stack_sizes": [100, 125, 150, 175, 200, 250],
              "stack_sizes": [100, 125],
              "blinds": [50, 100]}
# env = init_wrapped_env(**env_config)
# obs0 = env.reset(config=None)
num_envs = 3


def make_env(cfg):
    return init_wrapped_env(**cfg)


agents = ["p0", "p1"]
env = TianshouEnvWrapper(make_env(env_config), agents)
wrapped_env_fn = partial(PettingZooEnv, WrappedEnv(env))
wrapped_env = PettingZooEnv(WrappedEnv(env))
venv = SubprocVectorEnv([wrapped_env_fn for _ in range(num_envs)])


def get_rainbow_config():
    # network
    classes = [ActionSpace.FOLD,
               ActionSpace.CHECK_CALL,  # CHECK IS INCLUDED
               ActionSpace.RAISE_MIN_OR_3BB,
               ActionSpace.RAISE_HALF_POT,
               ActionSpace.RAISE_POT,
               ActionSpace.ALL_IN]
    hidden_dim = [512, 512]
    output_dim = len(classes)
    input_dim = 564  # hard coded for now -- very unlikely to be changed by me at any poiny in time
    device = "cpu"
    num_atoms = 51
    Q_dict = V_dict = {'input_dim': 564,
                       "output_dim": output_dim,
                       "hidden_sizes": hidden_dim,
                       "device": device,
                       }
    net = Net(state_shape=input_dim,
              action_shape=output_dim,
              hidden_sizes=hidden_dim,
              device=device,
              num_atoms=num_atoms,
              dueling_param=(Q_dict, V_dict)
              )
    optim = torch.optim.Adam(net.parameters(), lr=1e-6)
    # if running on GPU and we want to use cuda move model there
    return {'model': net,
            'optim': optim,
            'num_atoms': num_atoms,
            'v_min': -6,
            'v_max': 6,
            'estimation_step': 3,
            'target_update_freq': 500  # training steps
            }


rainbow_config = get_rainbow_config()
policy = MultiAgentPolicyManager([
    RainbowPolicy(**rainbow_config),
    RainbowPolicy(**rainbow_config)], wrapped_env)  # policy is made from PettingZooEnv

collector = Collector(policy, venv)
result = collector.collect(n_episode=1)
print(result)
