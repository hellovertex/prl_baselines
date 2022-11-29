import os
from typing import Dict, Union, Optional

import numpy as np
from gym.spaces import Box
from prl.environment.multi_agent.utils import make_multi_agent_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.simple_q import SimpleQ
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2

from prl.baselines.agents.policies import RandomPolicy, CallingStation, AlwaysMinRaise
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env
from ray.rllib import MultiAgentEnv, RolloutWorker, BaseEnv
from ray.rllib.algorithms.apex_dqn import ApexDQN, ApexDQNConfig
from ray.rllib.env import EnvContext
from ray.rllib.policy.policy import PolicySpec, Policy
from ray.rllib.utils import override
from ray.rllib.utils.typing import MultiAgentDict, PolicyID

n_players = 3
starting_stack_size = 2000

# todo update config with remaining rainbow hyperparams
# config = ApexDQNConfig().to_dict()
# config['num_atoms'] = 51
# config['log_level'] = "DEBUG"
RAINBOW_POLICY = "SimpleQ"
BASELINE_POLICY = "AlwaysMinRaise"

DistributedRainbow = SimpleQ


class OurRllibCallbacks(DefaultCallbacks):
    def on_episode_end(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Union[Episode, EpisodeV2, Exception],
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        """Runs when an episode is done.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy
                objects. In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
                In case of environment failures, episode may also be an Exception
                that gets thrown from the environment before the episode finishes.
                Users of this callback may then handle these error cases properly
                with their custom logics.
            env_index: The index of the sub-environment that ended the episode
                (within the vector of sub-environments of the BaseEnv).
            kwargs: Forward compatibility placeholder.
        """
        print(f'FROM WITHIN EPISODE END CB')
        print(f"EPISODE = {episode}")

    def on_episode_step(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            episode: Union[Episode, EpisodeV2],
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        """Runs on each episode step.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects.
                In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index: The index of the sub-environment that stepped the episode
                (within the vector of sub-environments of the BaseEnv).
            kwargs: Forward compatibility placeholder.
        """
        print(f'FROM WITHIN EPISODE STEP CB')
        print(f"EPISODE = {episode}")


def run_rainbow_vs_baseline_example(env_cls):
    """Run heuristic policies vs a learned agent.

    The learned agent should eventually reach a reward of ~5 with
    use_lstm=False, and ~7 with use_lstm=True. The reason the LSTM policy
    can perform better is since it can distinguish between the always_same vs
    beat_last heuristics.
    """

    def select_policy(agent_id, episode, **kwargs):
        # if "player" not in agent_id:
        #     raise ValueError("WRONG AGENT ID")
        if agent_id == 0:
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
        "log_level": "DEBUG",
        "callbacks": OurRllibCallbacks,
        "replay_buffer_config": {**SimpleQ.get_default_config()["replay_buffer_config"],
                                 "capacity": 1000},
        "multiagent": {
            "policies_to_train": [RAINBOW_POLICY],
            "policies": {
                RAINBOW_POLICY: PolicySpec(
                    config={  # todo make this a complete rainbow policy
                        "model": {"use_lstm": False},
                        "framework": "torch",
                    }
                ),
                BASELINE_POLICY: PolicySpec(policy_class=AlwaysMinRaise),
            },
            "policy_mapping_fn": select_policy,
        },
        "framework": "torch",
    }

    algo = DistributedRainbow(config=config)

    for _ in range(3):
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
