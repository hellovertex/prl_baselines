from enum import IntEnum
from typing import Optional, Union

import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils.env import ObsType
from tianshou.data import Collector
from tianshou.env.venvs import SubprocVectorEnv, DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from functools import partial
from pettingzoo.classic import tictactoe_v3
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env
# todo implement this https://pettingzoo.farama.org/tutorials/tianshou/intermediate/

# [x] implement tianshou env wrapper
# [x] implement seed() and render()
# [ ]


from pettingzoo.classic import texas_holdem_no_limit_v6
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


# todo: create one policy (actually zero because c51 already exists and the _mc_agent
#  does not have to be a policy after all


# todo: copy from policies.StakeLevelImitationPolicy.compute_action(obs:np.ndarray,...)
#  to create an agent that lives inside the environment
from tianshou.policy import base, MultiAgentPolicyManager, RainbowPolicy
from rlcard.envs import nolimitholdem
# env = PettingZooEnv(TianshoEnvWrapper(make_env(cfg)))
# venv = SubProcEnv(env)
class TianshouEnvWrapper(AECEnv):
    """
    Multi Agent Environment that changes reset call such that
    observation, reward, done, info = reset()
        becomes
    observation = reset(),
    so that tianshou can parse observation properly.
    """

    def seed(self, seed: Optional[int] = None) -> None:
        np.random.seed(seed)

    def observe(self, agent: str) -> Optional[ObsType]:
        raise NotImplementedError

    def render(self) -> Union[None, np.ndarray, str, list]:
        return self.env.render()  # returns None

    def state(self) -> np.ndarray:
        raise NotImplementedError

    def __init__(self, env):
        super().__init__()
        self.env = env
        # moved this to prl.baselines because I understand we need access to the baseline agents
        # which are not available from within prl_environment
        self._mc_agent = None

    def reset(self, config=None):
        observation, _, _, _ = super().reset(config)
        return observation

    def step(self, action):
        if action == MultiAgentActionFlags.TriggerMC:
            # compute action here
            pass

    def observation_space(self):
        pass

    def action_space(self):
        pass


env_config = {"env_wrapper_cls": TianshouEnvWrapper,
              "stack_sizes": [100, 125, 150, 175, 200, 250],
              "blinds": [50, 100]}
# env = init_wrapped_env(**env_config)
# obs0 = env.reset(config=None)
num_envs = 3


def make_env(cfg):
    return init_wrapped_env(**cfg)


env_fn = partial(make_env, env_config)
env_fns = [env_fn for _ in range(num_envs)]
# venv = SubprocVectorEnv(env_fns, wait_num=None, timeout=None)
venv = DummyVectorEnv(env_fns, wait_num=None, timeout=None)
obs = venv.reset()  # returns the initial observations of each environment
# todo get ready_id`s and reset only with ids of envs that signalled `done`
# returns "wait_num" steps or finished steps after "timeout" seconds,
# whichever occurs first.
print(obs)
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
    RainbowPolicy(**rainbow_config)], venv)  # policy is made from PettingZooEnv

collector = Collector(policy, venv)
