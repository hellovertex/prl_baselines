import os

import numpy as np
from gym.spaces import Box
from prl.environment.multi_agent.utils import make_multi_agent_env

from prl.baselines.agents.policies import RandomPolicy
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env
from ray.rllib import MultiAgentEnv
from ray.rllib.algorithms.apex_dqn import ApexDQN, ApexDQNConfig
from ray.rllib.env import EnvContext
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import override
from ray.rllib.utils.typing import MultiAgentDict

n_players = 3
starting_stack_size = 2000


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
    env_cls = make_multi_agent_env(env_cfg)
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
