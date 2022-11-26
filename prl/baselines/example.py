import os

import numpy as np
from gym.spaces import Box
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from ray.rllib import MultiAgentEnv
from ray.rllib.algorithms.apex_dqn import ApexDQN, ApexDQNConfig
from ray.rllib.env import EnvContext
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import override
from ray.rllib.utils.typing import MultiAgentDict

from prl.baselines.agents.policies import RandomPolicy
from prl.baselines.supervised_learning.data_acquisition.environment_utils import init_wrapped_env

n_players = 3
starting_stack_size = 2000


def make_rl_env(env_config):
    class _RLLibSteinbergerEnv(MultiAgentEnv):
        """Single Env that will be passed to rllib.env.MultiAgentEnv
        which internally creates n copies and decorates each env-copy with @ray.remote."""

        def __init__(self, config: EnvContext):

            self._n_players = env_config['n_players']
            self._starting_stack_size = env_config['starting_stack_size']
            self._env_cls = env_config['env_wrapper_cls']
            self._num_envs = env_config['num_envs']
            self.envs = [self._single_env() for _ in range(self._num_envs)]
            self.action_space = self.envs[0].action_space  # not batched, rllib wants that to be for single env
            # self.observation_space = self.envs[
            #     0].observation_space  # not batched, rllib wants that to be for single env
            # self.observation_space.dtype = np.float32
            self.observation_space = Box(low=0.0, high=6.0, shape=(564, ), dtype=np.float64)
            self._agent_ids = set(range(self._num_envs))  # _agent_ids name is enforced by rllib

            MultiAgentEnv.__init__(self)
            self.dones = set()
            self.rewards = {}
            self.acting_seat=None

        def _single_env(self):
            return init_wrapped_env(self._env_cls, [self._starting_stack_size for _ in range(self._n_players)])

        @override(MultiAgentEnv)
        def reset(self):
            self.dones = set()
            self.rewards = {}
            # return only obs nothing else, for each env
            self.acting_seat = 0
            return {i: env.reset()[0] for i, env in enumerate(self.envs)}

        def cumulate_rewards(self, rew):
            # update per agent reward
            if not self.acting_seat in self.rewards:
                self.rewards[self.acting_seat] = rew
            else:
                # update rew of agent on every sub env
                for key in self.rewards[self.acting_seat].keys():
                    self.rewards[self.acting_seat][key] += rew[key]
        @override(MultiAgentEnv)
        def step(self, action_dict):
            # agent A acts a --> step(a) --> obs, rew;  rew to A, obs to B?
            obs, rew, done, info = {}, {}, {}, {}

            for i, action in action_dict.items():
                obs[i], rew[i], done[i], info[i] = self.envs[i].step(action)
                if i in self.rewards:
                    self.rewards[i] += rew[i]
                else:
                    self.rewards[i] = rew[i]
                if done[i]:
                    self.dones.add(i)
            done["__all__"] = len(self.dones) == len(self.envs)

            # do we have to do a vector of cumulative rewards and select the right one to return?
            # e.g.  return the added 20 from [10,20,-5,30] and reset it to [10,0,-5,30] etc?
            # todo: fix the rewarding of agents AFTER having debug setup ready
            # self.cumulate_rewards(rew)
            # self.acting_seat = (self.acting_seat + 1) % self._n_players
            # rew = self.rewards[self.acting_seat]
            # self.rewards[self.acting_seat]
            return obs, rew, done, info

        @override(MultiAgentEnv)
        def render(self, mode='human'):
            pass

        @override(MultiAgentEnv)
        def observation_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
            """The name 'agent_ids' is taken from rllib's core fn, although I think env_ids would be better"""
            if agent_ids is None:
                agent_ids = list(range(len(self.envs)))
            obs = {agent_id: self.observation_space.sample() for agent_id in agent_ids}

            return obs

        @override(MultiAgentEnv)
        def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
            """The name 'agent_ids' is taken from rllib's core fn, although I think env_ids would be better"""
            if agent_ids is None:
                agent_ids = list(range(len(self.envs)))
            actions = {agent_id: self.action_space.sample() for agent_id in agent_ids}
            return actions

        @override(MultiAgentEnv)
        def action_space_contains(self, x: MultiAgentDict) -> bool:
            """The name 'agent_ids' is taken from rllib's core fn, although I think env_ids would be better"""
            if not isinstance(x, dict):
                return False
            return all(self.action_space.contains(val) for val in x.values())

        @override(MultiAgentEnv)
        def observation_space_contains(self, x: MultiAgentDict) -> bool:
            """The name 'agent_ids' is taken from rllib's core fn, although I think env_ids would be better"""
            if not isinstance(x, dict):
                return False
            return all(self.observation_space.contains(val) for val in x.values())

    return _RLLibSteinbergerEnv


# todo update config with remaining rainbow hyperparams
config = ApexDQNConfig().to_dict()
config['num_atoms'] = 51
RAINBOW_POLICY = "ApexDqnRainbow"
BASELINE_POLICY = "RandomPolicy"

DistributedRainbow = ApexDQN


def run_rainbow_vs_baseline_example(env_cls):
    """Run heuristic policies vs a learned agent.

    The learned agent should eventually reach a reward of ~5 with
    use_lstm=False, and ~7 with use_lstm=True. The reason the LSTM policy
    can perform better is since it can distinguish between the always_same vs
    beat_last heuristics.
    """

    def select_policy(agent_id, episode, **kwargs):
        if agent_id == "player_0":
            return RAINBOW_POLICY
        else:
            return BASELINE_POLICY

    config = {
        "env": env_cls,
        "gamma": 0.9,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 0,
        "num_envs_per_worker": 4,
        "rollout_fragment_length": 10,
        "train_batch_size": 200,
        "metrics_num_episodes_for_smoothing": 200,
        "multiagent": {
            "policies_to_train": ["ApexDqnRainbow"],
            "policies": {
                BASELINE_POLICY: PolicySpec(policy_class=RandomPolicy),
                RAINBOW_POLICY: PolicySpec(
                    config={  # todo make this a complete rainbow policy
                        "model": {"use_lstm": False},
                        "framework": "torch",
                    }
                ),
            },
            "policy_mapping_fn": select_policy,
        },
        "framework": "torch",
    }

    algo = DistributedRainbow(config=config)
    for _ in range(100000):
        results = algo.train()
        # Timesteps reached.
        if "policy_always_same_reward" not in results["hist_stats"]:
            reward_diff = 0
            continue
        reward_diff = sum(results["hist_stats"]["policy_learned_reward"])
        if results["timesteps_total"] > 100000:
            break
        # Reward (difference) reached -> all good, return.
        elif reward_diff > 1000:
            return


# ray.tune.run(ApexTrainer,
#              # config=config,  # todo check whether bottom config overwrites ApexDqnConfig
#              config={
#                  "env": "CartPole-v0",  # todo check how to set our env
#                  "num_gpus": 0,
#                  "num_workers": 1,
#                  "lr": tune.grid_search([0.01, 0.001, 0.0001]),
#              },
#              )
if __name__ == '__main__':
    env_cfg = {'env_wrapper_cls': AugmentObservationWrapper,
               'n_players': 2,
               'starting_stack_size': 1000,
               'num_envs': 2
               }
    env_cls = make_rl_env(env_cfg)
    # dummy_ctx = EnvContext(env_config={},
    #                        worker_index=0,  # 0 for local worker, >0 for remote workers.
    #                        vector_index=0,  # uniquely identify env when there are multiple envs per worker
    #                        remote=False,  # individual sub-envvs should be @ray.remote actors
    #                        num_workers=0,  # 0 for only local
    #                        recreated_worker=False
    #                        )
    # env = env_cls(dummy_ctx)
    # print(env)
    run_rainbow_vs_baseline_example(env_cls)
